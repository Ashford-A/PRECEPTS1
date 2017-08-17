
"""Consolidating -omic datasets.

This module contains classes for grouping continuous -omic datasets such as
expression or proteomic measurements with -omic phenotypic features such as
variants, copy number alterations, or drug response data so that the former
can be used to predict the latter using machine learning pipelines.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>
         Hannah Manning <manningh@ohsu.edu>

"""

from .expression import get_expr_bmeg
from .variants import get_variants_mc3, MuTree
from .copies import get_copies_firehose
from .pathways import parse_sif
from .annot import get_gencode
from .drugs import get_expr_ioria, get_drug_ioria

import numpy as np
from scipy.stats import fisher_exact
import random

from functools import reduce
from abc import abstractmethod


class CohortError(Exception):
    pass


class OmicCohort(object):
    """Base class for -omic datasets paired with a phenotype to predict.

    This class consists of a matrix of -omic measurements for a set of samples
    on a set of genetic features that will be used to predict phenotypes
    defined by the classes listed below. These measurements are stored in the
    omic_mat attribute, which is partitioned into a training cohort of samples
    and a testing cohort.

    Note that abstract :func:`train_pheno` and :func:`test_pheno` methods are
    defined here as well, which correspond to retrieval of phenotypic data as
    defined in downstream classes.

    Attributes:
        omic_mat (pandas DataFrame), shape = [n_samples, n_genes]
        train_samps (:obj:`frozenset` of :obj:`str`)
            Samples to be used for machine learning training.
        test_samps (:obj:`frozenset` of :obj:`str`)
            Samples to be used for machine learning testing.
        genes (:obj:`frozenset` of :obj:`str`)
            Genetic features included in the -omic dataset.
        cohort (str): The source of the datasets.
        cv_seed (int): A random seed used for sampling from the datasets.

    """

    def __init__(self, omic_mat, train_samps, test_samps, cohort, cv_seed):

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to each other and the -omic dataset
        self.train_samps = frozenset(train_samps)
        if not train_samps:
            raise CohortError("There must be at least one training sample!")

        if not (set(train_samps) & set(omic_mat.index)):
            raise CohortError("At least one training sample must be in the"
                              "-omic dataset!")

        if (test_samps is not None
                and not set(test_samps) & set(omic_mat.index)):
            raise CohortError("At least one testing sample must be in the"
                              "-omic dataset!")

        if test_samps is not None and set(train_samps) & set(test_samps):
            raise CohortError("Training sample set and testing sample"
                              "set must be disjoint!")

        # when we don't have a testing cohort, use entire the entire
        # dataset as the training cohort
        if test_samps is None:
            self.samples = self.train_samps.copy()

        # when we have a training cohort and a testing cohort
        else:
            self.samples = frozenset(train_samps) | frozenset(test_samps)
            self.test_samps = frozenset(test_samps)

        # remove duplicate features from the dataset and get
        # list of genomic features
        self.omic_mat = omic_mat.loc[self.samples,
                                     ~omic_mat.columns.duplicated()]
        self.genes = frozenset(self.omic_mat.columns)

        self.cohort = cohort
        self.cv_seed = cv_seed

    def omic_dims(self,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None,
                  use_test=False):
        """Gets a subset of the dimensions in the cohort's -omic dataset.

        This is a utility function whereby a list of samples and/or genes to
        be included and/or excluded in a given analysis can be specified.
        These lists are then checked against the samples and genetic features
        actually available in the training or testing cohort, and the
        cohort dimensions that are both available and match the inclusion/
        exclusion criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        sample or gene is asked to be both included and excluded it will be
        excluded. Returned dimensions are sorted to ensure that subsetted
        datasets with the same features and/or samples will be identical.

        Arguments:
            include_samps (:obj:`iterable` of :obj: `str`, optional)
            exclude_samps (:obj:`iterable` of :obj: `str`, optional)
            include_genes (:obj:`iterable` of :obj: `str`, optional)
            exclude_genes (:obj:`iterable` of :obj: `str`, optional)
            use_test (bool, optional): Whether to use testing cohort of
                samples, default is to use the training cohort.

        Returns:
            samps (:obj:`set` of :obj:`str`): The samples to be used.
            genes (:obj:`set` of :obj:`str`): The genetic features to be used.

        """

        # get samples and genes available in the given cohort
        if use_test:
            samps = self.test_samps.copy()
        else:
            samps = self.train_samps.copy()
        genes = self.genes.copy()

        # decide which samples to use
        if include_samps is not None:
            samps &= set(include_samps)
        if exclude_samps is not None:
            samps -= set(exclude_samps)

        # decide which genetic features to use
        if include_genes is not None:
            genes &= set(include_genes)
        if exclude_genes is not None:
            genes -= set(exclude_genes)

        return sorted(samps), sorted(genes)

    def train_omics(self,
                    include_samps=None, exclude_samps=None,
                    include_genes=None, exclude_genes=None):
        """Retrieval of the training cohort from the -omic dataset."""

        samps, genes = self.omic_dims(include_samps, exclude_samps,
                                      include_genes, exclude_genes,
                                      use_test=False)

        return self.omic_mat.loc[samps, genes]

    def test_omics(self,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):
        """Retrieval of the testing cohort from the -omic dataset."""

        samps, genes = self.omic_dims(include_samps, exclude_samps,
                                      include_genes, exclude_genes,
                                      use_test=True)

        return self.omic_mat.loc[samps, genes]

    @abstractmethod
    def train_pheno(self, pheno):
        """Returns the values for a phenotype in the training sub-cohort."""

    @abstractmethod
    def test_pheno(self, pheno):
        """Returns the values for a phenotype in the testing sub-cohort."""


class PresenceCohort(OmicCohort):
    """An -omic dataset used to predict the presence of binary phenotypes."""

    @abstractmethod
    def train_pheno(self, pheno):
        """Returns the binary labels corresponding to the presence of
           a phenotype for each of the samples in the training sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`bool`)
        """

    @abstractmethod
    def test_pheno(self, pheno):
        """Returns the binary labels corresponding to the presence of
           a phenotype for each of the samples in the testing sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`bool`)
        """

    def mutex_test(self, pheno1, pheno2):
        """Tests the mutual exclusivity of two phenotypes.

        Returns:
            pval (float): The p-value given by a Fisher's one-sided exact test
                          on the pair of phenotypes in the training cohort.

        Examples:
            >>> from HetMan.features.variants import MuType
            >>>
            >>> self.mutex_test(MuType({('Gene', 'TP53'): None}),
            >>>                 MuType({('Gene', 'CDH1'): None}))
            >>>
            >>> self.mutex_test(MuType({('Gene', 'PIK3CA'): None}),
            >>>                 MuType({('Gene', 'BRAF'): {
            >>>                             ('Location', '600'): None
            >>>                        }}))

        """
        pheno1_vec = self.train_pheno(pheno1)
        pheno2_vec = self.train_pheno(pheno2)

        return fisher_exact(table=[np.bincount(pheno1_vec),
                                   np.bincount(pheno2_vec)],
                            alternative='less')[1]


class ValueCohort(OmicCohort):
    """An -omic dataset used to predict continuous phenotypes."""

    @abstractmethod
    def train_pheno(self, pheno):
        """Returns the values of a phenotype for each of the samples
           in the training sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`float`)
        """

    @abstractmethod
    def test_pheno(self, pheno):
        """Returns the values of a phenotype for each of the samples
           in the testing sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`float`)
        """


class VariantCohort(PresenceCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        syn (synapseclient.Synapse): A logged-into Synapse instance.
        mut_genes (:obj:`list` of :obj:`str`)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`)
            What variant annotation levels to consider.
        cv_prop (float): Proportion of samples to use for cross-validation.

    Attributes:
        path (dict): Pathway Commons neighbourhood for the mutation genes.
        train_mut (.variants.MuTree): Training cohort mutations.
        test_mut (.variants.MuTree): Testing cohort mutations.

    Examples:
        >>> import synapseclient
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
        use_samples = list(set(variants['Sample']) & set(expr.index))
        use_samples.sort()
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
            test_samps = set(use_samples) - train_samps

            self.test_mut = MuTree(
                muts=variants.loc[variants['Sample'].isin(test_samps), :],
                levels=mut_levels
                )

        else:
            train_samps = set(use_samples)
            test_samps = None

        self.train_mut = MuTree(
            muts=variants.loc[variants['Sample'].isin(train_samps), :],
            levels=mut_levels
            )

        super().__init__(expr, train_samps, test_samps, cohort, cv_seed)

    def train_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.train_samps

        return self.train_mut.status(samps, mtype)

    def test_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.test_samps

        return self.test_mut.status(samps, mtype)


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
                 syn, cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 cv_seed=None, cv_prop=2.0 / 3):
        if mut_levels[0] != 'Gene' or mut_levels[1] != 'Form':
            raise ValueError("A cohort with CNA info must use 'Gene' as the"
                             "first mutation level and 'Form' as the second!")

        # initiates a cohort with expression and variant mutation data
        super().__init__(syn, cohort, mut_genes, mut_levels, cv_seed, cv_prop)

        # loads copy number data, gets list of samples with CNA info
        copy_data = get_copies_firehose(cohort.split('-')[-1], mut_genes)
        copy_samps = frozenset(
            reduce(lambda x, y: x & y,
                   set(tuple(copies.keys())
                       for gn, copies in copy_data.items()))
            )

        # removes samples that don't have CNA info
        self.samples = self.samples & copy_samps
        self.train_samps = self.train_samps & copy_samps
        if cv_prop < 1.0:
            self.test_samps = self.test_samps & copy_samps

        # removes expression data for samples with no CNA info, removes
        # variant data for samples with no CNA info
        self.omic_mat = self.omic_mat.loc[self.samples, :]
        self.train_mut = self.train_mut.subtree(self.train_samps)
        if cv_prop < 1.0:
            self.test_mut = self.test_mut.subtree(self.test_samps)

        # adds copy number alteration data to the mutation trees
        for gn in mut_genes:
            copy_vals = list(np.unique(list(copy_data[gn].values())))
            copy_vals.remove(0)
            val_labels = ['CNA_{}'.format(val) for val in copy_vals]

            if gn not in self.train_mut._child:
                self.train_mut._child[gn] = MuTree(
                    muts=pd.DataFrame(
                        {'Form': val_labels,
                         'Sample': [None for _ in val_labels]}
                        ),
                    levels=['Form'])

            if cv_prop < 1.0 and gn not in self.test_mut._child:
                self.test_mut._child[gn] = MuTree(
                    muts=pd.DataFrame(
                        {'Form': val_labels,
                         'Sample': [None for _ in val_labels]}
                        ),
                    levels=['Form'])

            for val_lbl in val_labels:
                self.train_mut[gn]._child[val_lbl] = set()
                if cv_prop < 1.0:
                    self.test_mut[gn]._child[val_lbl] = set()

            for samp, val in copy_data[gn].items():
                if val != 0:
                    lbl_indx = copy_vals.index(val)

                    if samp in self.train_samps:
                        self.train_mut[gn]._child[val_labels[lbl_indx]].\
                            update({samp})
                    elif cv_prop < 1.0:
                        self.test_mut[gn]._child[val_labels[lbl_indx]].\
                            update({samp})

            for val_lbl in val_labels:
                if self.train_mut[gn]._child[val_lbl]:
                    self.train_mut[gn]._child[val_lbl] = frozenset(
                        self.train_mut[gn]._child[val_lbl])
                else:
                    del(self.train_mut[gn]._child[val_lbl])

                if cv_prop < 1.0:
                    if self.test_mut[gn]._child[val_lbl]:
                        self.test_mut[gn]._child[val_lbl] = frozenset(
                            self.test_mut[gn]._child[val_lbl])
                    else:
                        del(self.test_mut[gn]._child[val_lbl])


class DrugCohort(ValueCohort):
    """An expression dataset used to predict clinical drug response.

        Args:
            drug_names (:obj:`list` of :obj:`str`):
                Which drugs to include as phenotypes.

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
