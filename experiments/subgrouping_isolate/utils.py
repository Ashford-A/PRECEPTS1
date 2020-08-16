
from dryadic.features.cohorts.mut import BaseMutationCohort
from dryadic.features.mutations import MuTree
import numpy as np


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


def remove_pheno_dups(muts, pheno_dict):
    mut_phns = set()
    mut_list = set()

    for mut in muts:
        mut_phn = tuple(pheno_dict[mut].tolist())

        if mut_phn not in mut_phns:
            mut_phns |= {mut_phn}
            mut_list |= {mut}

    return mut_list


class IsoMutationCohort(BaseMutationCohort):
    """This class introduces new cohort features in isolation experiments."""

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
        if other._leaf_annot != self._leaf_annot:
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
        other_cols = set(other._muts.columns)
        other_cols -= set(self._muts.columns
                          & other._muts.columns) - merge_cols

        if use_genes:
            self._muts = self._muts.loc[
                self._muts.Gene.isin(set(use_genes))]
            other._muts = other._muts.loc[
                other._muts.Gene.isin(set(use_genes))]

        self._muts = self._muts.merge(other._muts[other_cols],
                                      how='outer', on=tuple(merge_cols),
                                      validate='one_to_one')

        # check if we can recreate the mutation trees in the assimilated
        # cohort from scratch using the merged mutation data in this cohort
        for mut_lvls, mtree in other.mtrees.items():
            test_tree = MuTree(self._muts, mut_lvls,
                               leaf_annot=other._leaf_annot)

            for gene, gene_tree in mtree:
                if not use_genes or gene in set(use_genes):
                    if hash(gene_tree) != hash(test_tree[gene]):
                        raise ValueError(
                            "Cohorts have internally inconsistent mutation "
                            "datasets for gene {} at levels `{}`!".format(
                                gene, mut_lvls)
                            )

