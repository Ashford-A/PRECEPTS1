
"""Consolidating -omic datasets for prediction of phenotypes.

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
from .drugs import get_expr_ioria, get_drug_ioria
from .dream import get_dream_data

from .pathways import parse_sif
from .annot import get_gencode

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact
import random

from functools import reduce
from abc import abstractmethod


class CohortError(Exception):
    pass


class Cohort(object):
    """Base class for -omic datasets paired with phenotypes to predict.

    This class consists of a cohort of samples which are split into training
    and testing sub-cohorts used in the evaluation of machine learning
    models, along with a list of genes for which -omic features will be
    used in such models. The nature of these -omic features as well as the
    phenotypes they will be used to predict are defined by children classes.

    Attributes:
        train_samps (:obj:`frozenset` of :obj:`str`)
            Samples to be used for machine learning training.
        test_samps (:obj:`frozenset` of :obj:`str`)
            Samples to be used for machine learning testing.
        genes: The genetic features included in the -omic dataset.
        cohort_lbl (str): The source of the datasets.
        cv_seed (int): A random seed used for sampling from the datasets.

    """

    def __init__(self, train_samps, test_samps, genes, cohort_lbl, cv_seed):

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to each other
        self.train_samps = frozenset(train_samps)
        if not train_samps:
            raise CohortError("There must be at least one training sample!")

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

        # set the remaining attributes inherent to all cohorts
        self.genes = genes
        self.cohort_lbl = cohort_lbl
        self.cv_seed = cv_seed

    @staticmethod
    def split_samples(cv_seed, cv_prop, samps):
        """Splits a list of samples into training and testing sub-cohorts.

        Args:
            cv_seed (int): A random seed used for sampling.
            cv_prop (float): The proportion of samples to include in the
                             training sub-cohort.
            samps (:obj:`iterable` of :obj:`str`)
                The samples to split into sub-cohorts.

        Returns:
            train_samps (set): The samples in the training sub-cohort.
            test_samps (set): The samples in the testing sub-cohort.

        """
        random.seed(a=cv_seed)

        if cv_prop <= 0 or cv_prop > 1:
            raise ValueError("Improper cross-validation ratio that is "
                             "not > 0 and <= 1.0")

        # if not all samples are to be in the training sub-cohort, randomly
        # choose samples for the testing cohort...
        if cv_prop < 1:
            train_samps = set(
                random.sample(population=samps,
                              k=int(round(len(samps) * cv_prop)))
                )
            test_samps = set(samps) - train_samps

        # ...otherwise, copy the sample list to create the training cohort
        else:
            train_samps = set(samps)
            test_samps = set()

        return train_samps, test_samps

    def subset_samps(self,
                     include_samps=None, exclude_samps=None, use_test=False):
        """Gets a subset of the samples in the cohort's -omic dataset.

        This is a utility function whereby a list of samples to be included
        and/or excluded in a given analysis can be specified. This list is
        checked against the samples actually available in the training or
        testing cohort, and the samples that are both available and match the
        inclusion/exclusion criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        sample is asked to be both included and excluded it will be excluded.
        Returned samples are sorted to ensure that subsetted datasets with the
        same samples will be identical.

        Arguments:
            include_samps (:obj:`iterable` of :obj: `str`, optional)
            exclude_samps (:obj:`iterable` of :obj: `str`, optional)

        Returns:
            samps (:obj:`list` of :obj:`str`)

        See Also:
            `subset_genes`: similar function but for genetic features
                            of the dataset

        """
        if use_test:
            samps = self.test_samps.copy()
        else:
            samps = self.train_samps.copy()

        # decide which samples to use
        if include_samps is not None:
            samps &= set(include_samps)
        if exclude_samps is not None:
            samps -= set(exclude_samps)

        return sorted(samps)

    @abstractmethod
    def subset_genes(self, include_genes=None, exclude_genes=None):
        """Get a subset of genetic features available in this cohort."""

    def train_data(self,
                   pheno,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):
        """Retrieval of the training cohort from the -omic dataset."""

        samps = self.subset_samps(include_samps, exclude_samps,
                                  use_test=False)
        genes = self.subset_genes(include_genes, exclude_genes)

        pheno_vec = self.train_pheno(pheno, samps)
        nan_stat = ~np.isnan(pheno_vec)
        samps = np.array(samps)[nan_stat]

        return self.omic_loc(samps, genes), np.array(pheno_vec)[nan_stat]

    def test_data(self,
                  pheno,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):
        """Retrieval of the testing cohort from the -omic dataset."""

        samps = self.subset_samps(include_samps, exclude_samps,
                                  use_test=True)
        genes = self.subset_genes(include_genes, exclude_genes)

        pheno_vec = self.test_pheno(pheno, samps)
        nan_stat = ~np.isnan(pheno_vec)
        samps = np.array(samps)[nan_stat]

        return self.omic_loc(samps, genes), np.array(pheno_vec)[nan_stat]

    @abstractmethod
    def omic_loc(self, samps=None, genes=None):
        """Retrieval of a subset of the -omic dataset."""

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""


class UniCohort(Cohort):
    """A single -omic dataset paired with phenotypes to predict.

    Attributes:
        omic_mat (pandas DataFrame), shape = [n_samples, n_genes]

    """

    def __init__(self,
                 omic_mat, train_samps, test_samps, cohort_lbl, cv_seed):

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to the -omic dataset
        if not (set(train_samps) & set(omic_mat.index)):
            raise CohortError("At least one training sample must be in the"
                              "-omic dataset!")

        if (test_samps is not None
                and not set(test_samps) & set(omic_mat.index)):
            raise CohortError("At least one testing sample must be in the"
                              "-omic dataset!")

        Cohort.__init__(self,
                        train_samps, test_samps, self.omic_mat.columns,
                        cohort_lbl, cv_seed)

        # remove duplicate features from the dataset as well as samples
        # not listed in either the training or testing sub-cohorts
        self.omic_mat = omic_mat.loc[self.samples,
                                     ~omic_mat.columns.duplicated()]

    def subset_genes(self, include_genes=None, exclude_genes=None):
        """Gets a subset of the genes in the cohort's -omic dataset.

        This is a utility function whereby a list of genes to be included
        and/or excluded in a given analysis can be specified. This list is
        checked against the genes actually available in the dataset, and the
        genes that are both available and match the inclusion/exclusion
        criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        gene is asked to be both included and excluded it will be excluded.
        Returned genes are sorted to ensure that subsetted datasets with the
        same genes will be identical.

        Arguments:
            include_genes (:obj:`iterable` of :obj: `str`, optional)
            exclude_genes (:obj:`iterable` of :obj: `str`, optional)

        Returns:
            genes (:obj:`list` of :obj:`str`)

        See Also:
            `subset_samps`: similar function but for genetic features
                            of the dataset

        """
        genes = self.genes.copy()

        # decide which genetic features to use
        if include_genes is not None:
            genes &= set(include_genes)
        if exclude_genes is not None:
            genes -= set(exclude_genes)

        return sorted(genes)

    def omic_loc(self, samps=None, genes=None):
        """Retrieves a subset of the -omic dataset's
           samples and/or genetic features."""

        if samps is None:
            samps = self.samples.copy()
        if genes is None:
            genes = self.genes.copy()

        return self.omic_mat.loc[samps, genes]


class MultiCohort(Cohort):
    """Multiple -omic datasets paired with phenotypes to predict.

    Note that all datasets included in this class are to come from the same
    cohort of samples for which the phenotypes are defined. Thus each -omic
    dataset must have the same number of rows (samples), but can have
    varying numbers of columns (genes) depending on the nature of the -omic
    measurement in question.

    For example, a `MultiCohort` could consist of RNA sequencing and copy
    number alteration measurements taken from the same cohort of TCGA
    samples.

    Attributes:
        omic_mat_list (:obj:`list` of :obj:`pd.DataFrame`)

    """
    
    def __init__(self,
                 omic_mat_list, train_samps, test_samps,
                 cohort_lbl, cv_seed):

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to each of the -omic datasets
        for omic_mat in omic_mat_list:
            if not (set(train_samps) & set(omic_mat.index)):
                raise CohortError("At least one training sample must be "
                                  "in each -omic dataset!")

            if (test_samps is not None
                    and not set(test_samps) & set(omic_mat.index)):
                raise CohortError("At least one testing sample must be "
                                  "in each -omic dataset!")
    
        Cohort.__init__(
            self,
            train_samps, test_samps,
            [set(omic_mat.columns) for omic_mat in omic_mat_list],
            cohort_lbl, cv_seed
            )

        # removes duplicate genetic features from each of the datasets as
        # well as samples not in either the training or testing sub-cohorts
        self.omic_mat_list = [
            omic_mat.loc[self.samples, ~omic_mat.columns.duplicated()]
            for omic_mat in omic_mat_list
            ]

    def subset_genes(self, include_genes=None, exclude_genes=None):
        """Gets a subset of the genes in the cohort's -omic datasets.

        This is a utility function whereby a list of genes to be included
        and/or excluded in a given analysis can be specified. This list is
        checked against the genes actually available in the datasets, and the
        genes that are both available and match the inclusion/exclusion
        criteria are returned.

        Note that exclusion takes precedence over inclusion, that is, if a
        gene is asked to be both included and excluded it will be excluded.
        Returned genes are sorted to ensure that lists of subsetted datasets
        with the same genes will be identical.

        Also note that the genes to be included and excluded can be given as
        lists specific to each -omic dataset in the cohort, are a single list
        to be used to subset all the -omic datasets.

        Arguments:
            include_genes (:obj:`list` or :obj:`iterable` of :obj: `str`,
                           optional)
            exclude_genes (:obj:`list` or :obj:`iterable` of :obj: `str`,
                           optional)

        Returns:
            genes (:obj:`list` of :obj:`str`)

        See Also:
            `subset_samps`: similar function but for genetic features
                            of the dataset

        """
        genes = self.genes.copy()

        # adds genes to the list of genes to retrieve
        if include_genes is not None:
            if isinstance(list(include_genes)[0], str):
                genes = [gns & set(include_genes) for gns in self.genes]
            else:
                genes = [gns & set(in_gns)
                         for gns, in_gns in zip(self.genes, include_genes)]

        # removes genes from the list of genes to retrieve
        if exclude_genes is not None:
            if isinstance(list(exclude_genes)[0], str):
                genes = [gns - set(exclude_genes) for gns in genes]
            else:
                genes = [gns - set(ex_gns)
                         for gns, ex_gns in zip(genes, exclude_genes)]

        return [sorted(gns) for gns in genes]

    def omic_loc(self, samps=None, genes=None):
        """Retrieves a subset of each -omic dataset's
           samples and/or genetic features."""

        if samps is None:
            samps = self.samples.copy()
        if genes is None:
            genes = self.genes.copy()

        return [omic_mat.loc[samps, gns]
                for omic_mat, gns in zip(self.omic_mat_list, genes)]


class PresenceCohort(Cohort):
    """An -omic dataset used to predict the presence of binary phenotypes.
    
    This class is used to predict features such as the presence of a
    particular type of variant or copy number alteration, the presence of a
    binarized drug response, etc.
    """

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

    #TODO: extend this to MultiCohorts?
    def mutex_test(self, pheno1, pheno2):
        """Tests the mutual exclusivity of two phenotypes.

        Args:
            pheno1, pheno2: A pair of phenotypes stored in this cohort.

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
        conting_df = pd.DataFrame({'ph1': pheno1_vec, 'ph2': pheno2_vec})

        return fisher_exact(
            table=pd.crosstab(conting_df['ph1'], conting_df['ph2']),
            alternative='less'
            )[1]


class ValueCohort(Cohort):
    """An -omic dataset used to predict the value of continuous phenotypes.
   
    This class is used to predict features such as the area under the curve
    representing response to a particular drug, the abundance of a protein
    or phosphoprotein, CNA GISTIC score, etc.
    
    """

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


class VariantCohort(PresenceCohort, UniCohort):
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
        >>>
        >>> cdata = VariantCohort(
        >>>     syn, cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon']
        >>>     )

    """

    def __init__(self,
                 syn, cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 cv_seed=None, cv_prop=2.0/3):

        # loads gene expression and mutation data
        expr = get_expr_bmeg(cohort)
        variants = get_variants_mc3(syn)

        # loads the pathway neighbourhood of the variant genes, as well as
        # annotation data for all genes
        self.path = parse_sif(mut_genes)
        annot = get_gencode()

        # filters out genes that have both low levels of expression
        # and low variance of expression
        expr_mean = np.mean(expr)
        expr_var = np.var(expr)
        expr = expr.loc[:, ((expr_mean > np.percentile(expr_mean, 5))
                            | (expr_var > np.percentile(expr_var, 5)))]

        # filters out genes that do not have annotation available
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
        train_samps, test_samps = self.split_samples(
            cv_seed, cv_prop, use_samples)

        if test_samps:
            self.test_mut = MuTree(
                muts=variants.loc[variants['Sample'].isin(test_samps), :],
                levels=mut_levels
                )

        else:
            test_samps = None

        self.train_mut = MuTree(
            muts=variants.loc[variants['Sample'].isin(train_samps), :],
            levels=mut_levels
            )

        self.mut_genes = mut_genes
        self.cv_prop = cv_prop

        super().__init__(expr, train_samps, test_samps, cohort, cv_seed)

    def train_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.train_samps

        return self.train_mut.status(samps, mtype)

    def test_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.test_samps

        return self.test_mut.status(samps, mtype)


class MutCohort(VariantCohort, UniCohort):
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
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
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


class DrugCohort(ValueCohort, UniCohort):
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


class DreamCohort(ValueCohort, UniCohort):
    """A cohort for NCI-CPTAC DREAM Proteomics Sub-challenges 2 & 3.

    Args:
        omic_type (str): Which -omic datasets to use as prediction features.

    See Also:
        :module:`.dream`: The methods used to download the datasets used
                          in these sub-challenges.

    """

    def __init__(self,
                 syn, cohort, omic_type='rna', cv_seed=0, cv_prop=0.8):

        # gets the prediction features and the abundances to predict
        feat_mat = get_dream_data(syn, cohort, omic_type).fillna(0.0)
        prot_mat = get_dream_data(syn, cohort, 'prot')

        # filters out genes that have both low levels of expression
        # and low variance of expression
        feat_mean = np.mean(feat_mat)
        feat_var = np.var(feat_mat)
        feat_mat = feat_mat.loc[
            :, ((feat_mean > np.percentile(feat_mean, 10))
                | (feat_var > np.percentile(feat_var, 10)))
            ]

        # gets the samples that are common between the datasets, get the
        # training/testing cohort split
        use_samples = set(feat_mat.index) & set(prot_mat.index)
        train_samps, test_samps = self.split_samples(
            cv_seed, cv_prop, use_samples)

        # splits the protein abundances into training/testing sub-cohorts
        self.train_prot = prot_mat.loc[train_samps, :]
        if test_samps:
            self.test_prot = prot_mat.loc[test_samps, :]
        else:
            test_samps = None

        super().__init__(feat_mat, train_samps, test_samps, cohort, cv_seed)

    def train_pheno(self, prot_gene, samps=None):
        if samps is None:
            samps = self.train_samps

        return self.train_prot.loc[samps, prot_gene]

    def test_pheno(self, prot_gene, samps=None):
        if samps is None:
            samps = self.test_samps

        return self.test_prot.loc[samps, prot_gene]

    def train_data(self,
                   pheno,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):
        self.mut_genes = [pheno.split('__')[-1]]
        self.path = parse_sif(self.mut_genes)

        return super().train_data(pheno,
                                  include_samps, exclude_samps,
                                  include_genes, exclude_genes)

    def test_data(self,
                  pheno,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):
        self.mut_genes = [pheno.split('__')[-1]]
        self.path = parse_sif(self.mut_genes)

        return super().test_data(pheno,
                                 include_samps, exclude_samps,
                                 include_genes, exclude_genes)


class TransferDreamCohort(MultiCohort, DreamCohort):
    """A cohort for NCI-CPTAC DREAM Proteomics Sub-challenges 2 & 3.
    
    """

    def __init__(self, syn, cohort, cv_seed=0, cv_prop=0.8):

        # gets the prediction features and the abundances to predict
        rna_mat = get_dream_data(syn, cohort, 'rna').fillna(0.0)
        cna_mat = get_dream_data(syn, cohort, 'cna').fillna(0.0)
        prot_mat = get_dream_data(syn, cohort, 'prot')

        # filters out genes that have both low levels of expression
        # and low variance of expression
        rna_mean = np.mean(rna_mat)
        rna_var = np.var(rna_mat)
        rna_mat = rna_mat.loc[
            :, ((rna_mean > np.percentile(rna_mean, 10))
                | (rna_var > np.percentile(rna_var, 10)))
            ]

        # filters out genes that have both low levels of expression
        # and low variance of expression
        cna_mean = np.mean(cna_mat)
        cna_var = np.var(cna_mat)
        cna_mat = cna_mat.loc[
            :, ((cna_mean > np.percentile(cna_mean, 10))
                | (cna_var > np.percentile(cna_var, 10)))
            ]

        # gets the samples that are common between the datasets, get the
        # training/testing cohort split
        use_samples = (set(rna_mat.index) & set(cna_mat.index)
                       & set(prot_mat.index))
        train_samps, test_samps = self.split_samples(
            cv_seed, cv_prop, use_samples)

        # splits the protein abundances into training/testing sub-cohorts
        self.train_prot = prot_mat.loc[train_samps, :]
        if test_samps:
            self.test_prot = prot_mat.loc[test_samps, :]
        else:
            test_samps = None

        MultiCohort.__init__(self,
                             [rna_mat, cna_mat], train_samps, test_samps,
                             cohort, cv_seed)

