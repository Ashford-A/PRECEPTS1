
"""Consolidating -omics datasets.

This module contains classes for grouping expression datasets with other
-omic features such as variants, copy number alterations, and drug response
data so that the former can be used to predict the latter using machine
learning pipelines.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>
        Hannah Manning <manningh@ohsu.edu>

"""

from .expression import get_expr_bmeg
from .variants import get_variants_mc3, MuTree
from .copies import get_copies_firehose
from .pathways import parse_sif
from .annot import get_gencode
from .drugs import get_expr_ioria, get_drug_ioria, get_drug_bmeg

import numpy as np
from scipy.stats import fisher_exact
import random

from functools import reduce
from abc import abstractmethod


class OmicCohort(object):
    """A matched pair of expression and feature datasets for use in learning.

    Attributes:
        cohort (str): The source of the datasets.
        cv_seed (int): A seed used for random sampling from the datasets.

    """

    def __init__(self, omic_mat, train_samps, test_samps, cohort, cv_seed):

        if test_samps is not None and set(train_samps) & set(test_samps):
            raise ValueError("Training sample set and testing sample set must"
                             "be disjoint!")

        if test_samps is not None:
            self.samples = train_samps | test_samps
            self.train_samps = frozenset(train_samps)
            self.test_samps = frozenset(test_samps)

        else:
            self.samples = train_samps.copy()
            self.train_samps = frozenset(train_samps)


        self.omic_mat = omic_mat.loc[self.samples, :]
        self.genes = frozenset(self.omic_mat.columns)

        self.cohort = cohort
        self.cv_seed = cv_seed

    def omic_dims(self,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None,
                  use_test=False):
        """Gets the dimensions of the -omics dataset of the cohort.

        """

        # get samples and genes from the specified cohort as specified
        if use_test:
            samps = self.test_samps.copy()
        else:
            samps = self.train_samps.copy()
        genes = self.genes.copy()

        # remove samples and/or genes as necessary
        if include_samps is not None:
            samps &= set(include_samps)
        if exclude_samps is not None:
            samps -= set(exclude_samps)

        if include_genes is not None:
            genes &= set(include_genes)
        if exclude_genes is not None:
            genes -= set(exclude_genes)

        return samps, genes

    def train_omics(self,
                    include_samps=None, exclude_samps=None,
                    include_genes=None, exclude_genes=None):
        samps, genes = self.omic_dims(include_samps, exclude_samps,
                                      include_genes, exclude_genes,
                                      use_test=False)

        return self.omic_mat.loc[samps, genes]

    def test_omics(self,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):
        samps, genes = self.omic_dims(include_samps, exclude_samps,
                                      include_genes, exclude_genes,
                                      use_test=True)

        return self.omic_mat.loc[samps, genes]

    @abstractmethod
    def train_pheno(self, pheno):
        """Returns the training values of a phenotype."""

    @abstractmethod
    def test_pheno(self, pheno):
        """Returns the testing values of a phenotype."""


class LabelCohort(OmicCohort):
    """A matched pair of omics and discrete phenotypic data."""

    def __init__(self, omic_mat, train_samps, test_samps, cohort, cv_seed):
        super().__init__(omic_mat, train_samps, test_samps, cohort, cv_seed)


class ValueCohort(OmicCohort):
    """A matched pair of omics and continuous phenotypic data."""

    def __init__(self, omic_mat, train_samps, test_samps, cohort, cv_seed):
        super().__init__(omic_mat, train_samps, test_samps, cohort, cv_seed)


class VariantCohort(LabelCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        mut_genes (list of str): Which genes' variants to include.
        mut_levels (list of str): What variant annotation level to consider.
        cv_prop (float): Proportion of samples to use for cross-validation.

    Attributes:
        train_expr (pandas DataFrame of floats)
        test_expr (pandas DataFrame of floats)
        train_mut (MuTree)
        test_mut (MuTree)
        path (dict)

    Examples:
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>> cdata = VariantCohort(
        >>>     syn, cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon']
        >>>     )

    """

    def __init__(self,
                 syn, cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 cv_seed=None, cv_prop=2.0/3):
        # TODO: double-check how Python handles random seeds
        if cv_prop <= 0 or cv_prop > 1:
            raise ValueError("Improper cross-validation ratio that is "
                             "not > 0 and <= 1.0")
        self.mut_genes = mut_genes
        self.cv_prop = cv_prop

        # loads gene expression and mutation data
        expr = get_expr_bmeg(cohort)
        variants = get_variants_mc3(syn)

        # loads the pathway neighbourhood of the variant genes, as well as
        # annotation data for all genes
        self.path = parse_sif(mut_genes)
        annot = get_gencode()

        # filters out genes that don't have any variation across the samples
        # or are not included in the annotation data
        expr = expr.loc[:, expr.apply(lambda x: np.var(x) > 0.005)].dropna()
        annot = {g: a for g,a in annot.items()
                 if a['gene_name'] in expr.columns}
        annot_genes = [a['gene_name'] for g, a in annot.items()]
        expr = expr.loc[:, annot_genes]

        # gets set of samples shared across expression and mutation datasets,
        # subsets these datasets to use only these samples
        use_samples = set(variants['Sample']) & set(expr.index)
        variants = variants.loc[variants['Gene'].isin(mut_genes), :]

        # gets annotation data for the genes whose mutations
        # are under consideration
        annot_data = {a['gene_name']: {'ID': g, 'Chr': a['chr'],
                                       'Start': a['Start'], 'End': a['End']}
                      for g, a in annot.items()
                      if a['gene_name'] in mut_genes}
        self.annot = annot
        self.mut_annot = annot_data

        # gets subset of samples to use for training, and split the expression
        # and variant datasets accordingly into training/testing cohorts
        random.seed(a=cv_seed)
        if cv_prop < 1:
            train_samps = set(
                random.sample(population=use_samples,
                              k=int(round(len(use_samples) * cv_prop)))
                )
            test_samps = use_samples - train_samps

            self.test_mut = MuTree(
                muts=variants.loc[variants['Sample'].isin(test_samps), :],
                levels=mut_levels
                )

        else:
            train_samps = use_samples.copy()
            test_samps = set()

        self.train_mut = MuTree(
            muts=variants.loc[variants['Sample'].isin(train_samps), :],
            levels=mut_levels)

        super().__init__(expr, train_samps, test_samps, cohort, cv_seed)

    def train_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.train_samps
        return self.train_mut.status(samps, mtype)

    def test_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.test_samps
        return self.test_mut.status(samps, mtype)

    def mutex_test(self, mtype1, mtype2):
        """Tests the mutual exclusivity of two mutation types.

        Args:
            mtype1, mtype2 (MuType)

        Returns:
            pval (float): The p-value given by a Fisher's one-sided exact test
                          using the training samples in the cohort.

        Examples:
            >>> self.mutex_test(MuType({('Gene', 'TP53'): None}),
            >>>                 MuType({('Gene', 'CDH1'): None}))
            >>> self.mutex_test(MuType({('Gene', 'PIK3CA'): None}),
            >>>                 MuType({('Gene', 'BRAF'): {
            >>>                             ('Location', '600'): None
            >>>                        }}))

        """
        samps1 = mtype1.get_samples(self.train_mut)
        samps2 = mtype2.get_samples(self.train_mut)

        if not samps1 or not samps2:
            pval = 1

        else:
            both_samps = samps1 & samps2

            _, pval = fisher_exact(
                np.array([[len(self.samples - (samps1 | samps2)),
                           len(samps1 - both_samps)],
                          [len(samps2 - both_samps),
                           len(both_samps)]]),
                alternative='less')

        return pval


class MutCohort(VariantCohort):
    """An expression dataset used to predict mutations, including CNAs.

    A MutCohort is constructed by first constructing a VariantCohort with the
    same attributes, and then adding copy number alteration (CNA) data on top
    of the variant mutation data.

    Note that CNAs are split according to the 'Form' mutation level, with each
    branch at this level corresponding to a type of CNA, eg. -2 for homozygous
    loss, 1 for heterozygous amplification, etc. If further mutation levels
    specified they will only be added to the branches of the mutation trees
    corresponding to variants.

    Examples:
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>> cdata = MutCohort(
        >>>     syn, cohort='TCGA-OV', mut_genes=['RB1', 'TTN'],
        >>>     mut_levels=['Gene', 'Form', 'Protein']
        >>>     )

    """

    def __init__(self,
                 syn, cohort, mut_genes, mut_levels=('Gene', 'Type'),
                 cv_seed=None, cv_prop=2.0 / 3):
        if mut_levels[0] != 'Gene' or mut_levels[1] != 'Form':
            raise ValueError("A cohort with CNA info must use 'Gene' as the"
                             "first mutation level and 'Form' as the second!")

        # initiates a cohort with expression and variant mutation data
        super(MutCohort, self).__init__(syn, cohort, mut_genes, mut_levels,
                                        cv_seed, cv_prop)

        # loads copy number data, gets list of samples with CNA info
        copy_data = get_copies_firehose(cohort.split('-')[-1], mut_genes)
        copy_samps = frozenset(
            reduce(lambda x, y: x & y,
                   set(tuple(copies.keys())
                       for gn, copies in copy_data.items()))
            )

        # removes samples that don't have CNA info
        self.samples = self.samples & copy_samps
        self.train_samps_ = self.train_samps_ & copy_samps
        self.test_samps_ = self.test_samps_ & copy_samps

        # removes expression data for samples with no CNA info
        self.train_expr_ = self.train_expr_.loc[self.train_samps_, :]
        self.test_expr_ = self.test_expr_.loc[self.test_samps_, :]

        # removes variant data for samples with no CNA info
        self.train_mut_ = self.train_mut_.subtree(self.train_samps_)
        self.test_mut_ = self.test_mut_.subtree(self.test_samps_)

        # adds copy number alteration data to the mutation trees
        for gn in mut_genes:
            copy_vals = list(np.unique(list(copy_data[gn].values())))
            copy_vals.remove(0)
            val_labels = ['CNA_{}'.format(val) for val in copy_vals]

            if gn not in self.train_mut_._child:
                self.train_mut_._child[gn] = MuTree(
                    muts=pd.DataFrame(
                        {'Form': val_labels,
                         'Sample': [None for _ in val_labels]}
                        ),
                    levels=['Form'])

            if gn not in self.test_mut_._child:
                self.test_mut_._child[gn] = MuTree(
                    muts=pd.DataFrame(
                        {'Form': val_labels,
                         'Sample': [None for _ in val_labels]}
                        ),
                    levels=['Form'])

            for val_lbl in val_labels:
                self.train_mut_[gn]._child[val_lbl] = set()
                self.test_mut_[gn]._child[val_lbl] = set()

            for samp, val in copy_data[gn].items():
                if val != 0:
                    lbl_indx = copy_vals.index(val)

                    if samp in self.train_samps_:
                        self.train_mut_[gn]._child[val_labels[lbl_indx]].\
                            update({samp})
                    else:
                        self.test_mut_[gn]._child[val_labels[lbl_indx]].\
                            update({samp})

            for val_lbl in val_labels:
                if self.train_mut_[gn]._child[val_lbl]:
                    self.train_mut_[gn]._child[val_lbl] = frozenset(
                        self.train_mut_[gn]._child[val_lbl])
                else:
                    del(self.train_mut_[gn]._child[val_lbl])

                if self.test_mut_[gn]._child[val_lbl]:
                    self.test_mut_[gn]._child[val_lbl] = frozenset(
                        self.test_mut_[gn]._child[val_lbl])
                else:
                    del(self.test_mut_[gn]._child[val_lbl])


class DrugCohort(ValueCohort):
    """An expression dataset used to predict clinical drug response.

        Args:
            drug_list (list of str): Which drugs to include
            cv_prop (float): Proportion of samples to use for cross-validation

        Attributes:
            train_samps_(frozenset of str)
            test_samps_ (frozenset of str)
            train_expr_ (pandas DataFrame of floats)
            test_expr_ (pandas DataFrame of floats)
            train_resp_ (pandas DataFrame of floats)
            test_resp_ (pandas DataFrame of floats)

        Examples:

        """

    def __init__(self, cohort, drug_names, cv_seed=None, cv_prop=2.0 / 3):
        if cv_prop <= 0 or cv_prop > 1:
            raise ValueError("Improper cross-validation ratio that is "
                             "not > 0 and <= 1.0")
        self.drug_names = drug_names
        self.cv_prop = cv_prop

        cell_expr = get_expr_ioria()

        # TODO: choose a non-AUC measure of drug response
        drug_resp = get_drug_ioria(drug_names)

        # drops cell lines (rows) w/ no expression data & genes (cols)
        # with any missing values
        cell_expr = cell_expr.dropna(axis=0, how='all')\
            .dropna(axis=1, how='any')

        # drops cell lines (rows) w/ no expression data
        drug_resp = drug_resp.dropna(axis=0, how='all')

        # gets set of cell lines ("samples") shared between drug_resp and
        # cell_expr datasets
        use_samples = set(cell_expr.index) & set(drug_resp.index)

        # discards data for cell lines which are not in samples set
        cell_expr = cell_expr.loc[use_samples, :]
        drug_resp = drug_resp.loc[use_samples, :]

        # TODO: query bmeg for annotation data on each drug (def in drugs.py),
        # set as attribute

        random.seed(a=cv_seed)
        if cv_prop < 1:

            # separate samples (cell line names) into train and test frozensets.
            train_samps = set(
                random.sample(population=use_samples,
                              k=int(round(len(use_samples) * cv_prop))))
            test_samps = use_samples - train_samps

            self.train_resp = drug_resp.loc[train_samps, :]
            self.test_resp = drug_resp.loc[test_samps, :]

        else:
            train_samps = use_samples
            test_samps = set()

        super().__init__(cell_expr, train_samps, test_samps, cohort, cv_seed)

    def train_pheno(self, drug, samps=None):
        if samps is None:
            samps = self.train_samps

        return self.train_resp.loc[samps, drug]

    def test_pheno(self, drug, samps=None):
        if samps is None:
            samps = self.test_samps

        return self.test_resp.loc[samps, drug]
