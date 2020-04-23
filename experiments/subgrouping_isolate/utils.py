
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_isolate.utils import ExMcomb
from dryadic.features.mutations import *
from dryadic.features.cohorts.mut import BaseMutationCohort

from functools import reduce
from operator import or_


def get_mtype_gene(mtype):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is not None:
            mtype_genes = mtype.base_mtype.get_labels()

        else:
            mtype_genes = [None]

    elif isinstance(mtype, ExMcomb):
        mtype_genes = mtype.all_mtype.get_labels()

    elif isinstance(mtype, MutComb):
        mtype_genes = list(reduce(or_, [set(mtp.get_labels())
                                        for mtp in mtype.mtypes]))

    elif isinstance(mtype, MuType):
        mtype_genes = mtype.get_labels()

    else:
        raise ValueError("Cannot retrieve gene for something that is "
                         "not a mutation!")

    assert isinstance(mtype_genes, list)

    if len(mtype_genes) > 1:
        raise ValueError("Cannot retrieve gene for a mutation associated "
                         "with multiple genes!")

    if len(mtype_genes) == 0:
        raise ValueError("Cannot retrieve gene for a mutation associated "
                         "with no genes!")

    return mtype_genes[0]


class IsoMutationCohort(BaseMutationCohort):

    def data_hash(self):
        return ({gene: round(expr_val, 2)
                 for gene, expr_val in sorted(super().data_hash()[0])},
                hash(tuple(sorted(self.mtrees.items()))))

    def merge(self, other):
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

        self.muts = self.muts.merge(other.muts[other_cols],
                                    how='left', on=tuple(merge_cols),
                                    validate='one_to_one')

        # check if we can recreate the mutation trees in the assimilated
        # cohort from scratch using the merged mutation data in this cohort
        for mut_lvls, mtree in other.mtrees.items():
            test_tree = MuTree(self.muts, mut_lvls,
                               leaf_annot=other.leaf_annot)

            for gene, gene_tree in mtree:
                if hash(gene_tree) != hash(test_tree[gene]):
                    raise ValueError("Cohorts have internally inconsistent "
                                     "mutation datasets for gene {} at "
                                     "levels `{}`!".format(gene, mut_lvls))

        #TODO: should this return `None`?
        return self

