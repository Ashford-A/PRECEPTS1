
from ..subvariant_isolate.merge_isolate import calculate_siml
from dryadic.features.mutations import MuTree
from dryadic.features.cohorts.mut import BaseMutationCohort
import numpy as np


def search_siml_pair(siml_dict, mut, other_mut):
    return {mut_lvls: siml_df.loc[mut, other_mut]
            for mut_lvls, siml_df in siml_dict.items()
            if mut in siml_df.index and other_mut in siml_df.columns}


def calculate_pair_siml(mcomb1, mcomb2, all_mtype, siml_dfs,
                        pheno_dict, pred_df, ex_lbl, cdata, test_list=None):
    pair_simls = search_siml_pair(siml_dfs, mcomb1, mcomb2)

    if len(pair_simls) == 0 or (len(pair_simls) == 1
                                and test_list is not None):
        pheno_dict[ex_lbl, mcomb1] = np.array(cdata.train_pheno(all_mtype))
        use_phns = {lbl: phn_vals for lbl, phn_vals in pheno_dict.items()
                    if lbl in {mcomb1, (ex_lbl, mcomb1), mcomb2}}

        pair_siml = calculate_siml(
            mcomb1, use_phns, (ex_lbl, mcomb1),
            pred_df.loc[mcomb1, cdata.get_train_samples()]
            )[mcomb2]

        if test_list is not None and len(pair_simls) == 1:
            test_list += [(mcomb1, mcomb2)]

            assert (
                pair_siml == tuple(pair_simls.values())[0]), (
                    "Similarity values are internally inconsistent!")

    elif len(pair_simls) == 1:
        pair_siml = tuple(pair_simls.values())[0]

    else:
        raise ValueError("Multiple similarity values found!")

    if test_list is not None:
        return pair_siml, test_list
    else:
        return pair_siml


class IsoMutationCohort(BaseMutationCohort):

    def mtrees_status(self, mtype, samps=None):
        for lvls, mtree in self.mtrees.items():
            if mtree.match_levels(mtype):
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

