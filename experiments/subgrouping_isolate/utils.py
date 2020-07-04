
from ..subvariant_isolate.merge_isolate import calculate_siml
from ..subvariant_tour.utils import RandomType
from dryadic.features.mutations import MuTree
from dryadic.features.cohorts.mut import BaseMutationCohort

import numpy as np
from scipy.stats import ks_2samp


def search_siml_pair(siml_dicts, mut, other_mut):
    simls = dict()

    for mut_lvls, siml_dfs in siml_dicts.items():
        for siml_df in siml_dfs:
            if mut in siml_df.columns and other_mut in siml_df.index:
                simls[mut_lvls] = siml_df.loc[other_mut, mut]
                break

    return simls

# TODO: clean this up / find another use for it
def calculate_pair_siml(mcomb1, mcomb2, siml_dicts, all_phn=None,
                        pheno_dict=None, pred_vals=None, mean_dict=None):

    run_tests = False
    pair_simls = []

    if len(pair_simls) == 0 or (run_tests and len(pair_simls) == 1):
        if mean_dict is None:
            none_mean = np.concatenate(pred_vals[~all_phn].values).mean()
            base_mean = np.concatenate(
                pred_vals[pheno_dict[mcomb1]].values).mean()
        else:
            none_mean, base_mean = mean_dict['none'], mean_dict['base']

        pair_siml = np.concatenate(
            pred_vals[pheno_dict[mcomb2]].values).mean() - none_mean
        pair_siml /= (base_mean - none_mean)

        if run_tests and len(pair_simls) == 1:
            test_list = mcomb1, mcomb2

            assert (
                pair_siml == tuple(pair_simls.values())[0]), (
                    "Similarity values are internally inconsistent!")

    elif len(pair_simls) == 1:
        pair_siml = tuple(pair_simls.values())[0]

    else:
        raise ValueError("Multiple similarity values found!")

    if run_tests:
        return pair_siml, test_list
    else:
        return pair_siml


def calculate_mean_siml(wt_vals, mut_vals, other_vals,
                        wt_mean=None, mut_mean=None, other_mean=None):
    if wt_mean is None:
        wt_mean = wt_vals.mean()

    if mut_mean is None:
        mut_mean = mut_vals.mean()

    if other_mean is None:
        other_mean = other_vals.mean()

    return (other_mean - wt_mean) / (mut_mean - wt_mean)


def calculate_ks_siml(wt_vals, mut_vals, other_vals,
                      base_dist=None, wt_dist=None, mut_dist=None):
    if base_dist is None:
        base_dist = ks_2samp(wt_vals, mut_vals,
                             alternative='greater').statistic
        base_dist -= ks_2samp(wt_vals, mut_vals, alternative='less').statistic

    if wt_dist is None:
        wt_dist = ks_2samp(wt_vals, other_vals,
                           alternative='greater').statistic
        wt_dist -= ks_2samp(wt_vals, other_vals, alternative='less').statistic

    if mut_dist is None:
        mut_dist = ks_2samp(mut_vals, other_vals,
                            alternative='greater').statistic
        mut_dist -= ks_2samp(mut_vals, other_vals,
                             alternative='less').statistic

    return (base_dist + wt_dist + mut_dist) / (2 * base_dist)


class IsoMutationCohort(BaseMutationCohort):

    def mtrees_status(self, mtype, samps=None):
        for lvls, mtree in self.mtrees.items():
            if ((not isinstance(mtype, RandomType)
                 and mtree.match_levels(mtype))
                    or (isinstance(mtype, RandomType)
                        and mtree.match_levels(mtype.base_mtype))):
                phn = mtree.status(samps, mtype)
                break

        else:
            if mtype.cur_level == 'Gene':
                phns = []

                for lbls, subtype in mtype.child_iter():
                    for lvls, mtree in self.mtrees.items():
                        if mtree.match_levels(subtype):
                            phns += [mtree.status(samps, subtype)]
                            break

                phn = [any(phn_vals) for phn_vals in zip(*phns)]

            else:
                raise ValueError("Unable to retrieve phenotype data "
                                 "for `{}`!".format(mtype))

        return phn

    def data_hash(self):
        return ({gene: round(expr_val, 2)
                 for gene, expr_val in sorted(super().data_hash()[0])},
                hash(tuple(sorted(self.mtrees.items()))))

    def merge(self, other, use_genes=None):
        """Assimilates another cohort object from the same dataset."""

        if not isinstance(other, IsoMutationCohort):
            return NotImplemented

        if other.data_hash()[0] != self.data_hash()[0]:
            raise ValueError("Cohorts have mismatching expression datasets!")
        if other.gene_annot != self.gene_annot:
            raise ValueError("Cohorts have mismatching genomic annotations!")
        if other.leaf_annot != self.leaf_annot:
            raise ValueError("Cohorts have mismatching variant annotations!")

        # for each mutation tree in the cohort to be merged, check if there is
        # already a tree for the same mutation attributes in this cohort...
        for mut_lvls, mtree in other.mtrees.items():
            if mut_lvls in self.mtrees:
                if hash(mtree) != hash(self.mtrees[mut_lvls]):
                    raise ValueError("Cohorts have mismatching mutation "
                                     "trees at levels `{}`!".format(mut_lvls))

            # ...if there isn't, just add the tree to this cohort
            else:
                self.mtrees[mut_lvls] = mtree

        # this defines a unique merge key for each mutation in the cohorts
        merge_cols = {'Sample', 'Feature', 'Location', 'VarAllele',
                      'Gene', 'Scale', 'Copy'}

        # removes the columns in the mutation data to be merged that are
        # neither in the merge key nor novel to the mutations to be merged
        other_cols = set(other.muts.columns)
        other_cols -= set(self.muts.columns & other.muts.columns) - merge_cols

        if use_genes:
            self.muts = self.muts.loc[self.muts.Gene.isin(set(use_genes))]
            other.muts = other.muts.loc[other.muts.Gene.isin(set(use_genes))]

        self.muts = self.muts.merge(other.muts[other_cols],
                                    how='outer', on=tuple(merge_cols),
                                    validate='one_to_one')

        # check if we can recreate the mutation trees in the assimilated
        # cohort from scratch using the merged mutation data in this cohort
        for mut_lvls, mtree in other.mtrees.items():
            test_tree = MuTree(self.muts, mut_lvls,
                               leaf_annot=other.leaf_annot)

            for gene, gene_tree in mtree:
                if not use_genes or gene in set(use_genes):
                    if hash(gene_tree) != hash(test_tree[gene]):
                        raise ValueError(
                            "Cohorts have internally inconsistent mutation "
                            "datasets for gene {} at levels `{}`!".format(
                                gene, mut_lvls)
                            )

