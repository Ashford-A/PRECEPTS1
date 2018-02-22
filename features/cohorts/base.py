
"""Consolidating -omic datasets for prediction of phenotypes.

This module contains classes for grouping continuous -omic datasets such as
expression or proteomic measurements with -omic phenotypic features such as
variants, copy number alterations, or drug response data so that the former
can be used to predict the latter using machine learning pipelines.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>
         Hannah Manning <manningh@ohsu.edu>

"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import random
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

    Args:
        genes (frozenset): The genetic features included in the -omic dataset.
        cv_seed (:obj: `int`, optional)
            A random seed used for sampling from the datasets.

    """

    def __init__(self, genes, cv_seed=None):
        self.genes = genes

        if cv_seed is None:
            self.cv_seed = 0
        else:
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
                random.sample(population=sorted(tuple(samps)),
                              k=int(round(len(samps) * cv_prop)))
                )
            test_samps = set(samps) - train_samps

        # ...otherwise, copy the sample list to create the training cohort
        else:
            train_samps = set(samps)
            test_samps = set()

        return train_samps, test_samps

    @abstractmethod
    def subset_samps(self, include_samps=None, exclude_samps=None):
        """Get a subset of the samples available in this cohort."""

    @abstractmethod
    def subset_genes(self, include_genes=None, exclude_genes=None):
        """Get a subset of the genetic features available in this cohort."""

    @abstractmethod
    def omic_loc(self, samps=None, genes=None):
        """Retrieval of a subset of the -omic dataset."""

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""
        raise CohortError("Cannot use base Cohort class!")

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""
        raise CohortError("Cannot use base Cohort class!")


class UniCohort(Cohort):
    """A single -omic dataset paired with phenotypes to predict.

    Attributes:
        omic_mat (pandas DataFrame), shape = [n_samples, n_genes]

    """

    def __init__(self,
                 omic_mat, train_samps, test_samps, cv_seed):

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to the -omic dataset
        if not (set(train_samps) & set(omic_mat.index)):
            raise CohortError("At least one training sample must be in the "
                              "-omic dataset!")

        if (test_samps is not None
                and not set(test_samps) & set(omic_mat.index)):
            raise CohortError("At least one testing sample must be in the"
                              "-omic dataset!")

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

        super().__init__(frozenset(omic_mat.columns), cv_seed)

        # remove duplicate features from the dataset as well as samples
        # not listed in either the training or testing sub-cohorts
        self.omic_mat = omic_mat.loc[self.samples,
                                     ~omic_mat.columns.duplicated()]

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
            use_test (bool, optional): Whether to use the testing cohort.

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

        # decide what samples to use based on exclusion and inclusion criteria
        if include_samps is not None:
            samps &= set(include_samps)
        if exclude_samps is not None:
            samps -= set(exclude_samps)

        return sorted(samps)

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

    def train_data(self,
                   pheno,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):
        """Retrieval of the training cohort from the -omic dataset."""

        samps = self.subset_samps(include_samps, exclude_samps,
                                  use_test=False)
        genes = self.subset_genes(include_genes, exclude_genes)
        pheno_mat = np.array(self.train_pheno(pheno, samps))

        if pheno_mat.ndim == 1:
            pheno_mat = pheno_mat.reshape(-1, 1)

        elif pheno_mat.shape[1] == len(samps):
            pheno_mat = np.transpose(pheno_mat)

        elif pheno_mat.shape[0] != len(samps):
            raise ValueError(
                "Given phenotype(s) do not return a valid matrix of values "
                "to predict from the training data!"
                )

        nan_stat = np.any(~np.isnan(pheno_mat), axis=1)
        samps = np.array(samps)[nan_stat]

        if pheno_mat.shape[1] == 1:
            pheno_mat = pheno_mat.ravel()

        return self.omic_loc(samps, genes), pheno_mat[nan_stat]

    def test_data(self,
                  pheno,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):
        """Retrieval of the testing cohort from the -omic dataset."""

        samps = self.subset_samps(include_samps, exclude_samps,
                                  use_test=True)
        genes = self.subset_genes(include_genes, exclude_genes)
        pheno_mat = np.array(self.test_pheno(pheno, samps))

        if pheno_mat.ndim == 1:
            pheno_mat = pheno_mat.reshape(-1, 1)

        elif pheno_mat.shape[1] == len(samps):
            pheno_mat = np.transpose(pheno_mat)

        elif pheno_mat.shape[0] != len(samps):
            raise ValueError(
                "Given phenotype(s) do not return a valid matrix of values "
                "to predict from the training data!"
                )

        nan_stat = np.any(~np.isnan(pheno_mat), axis=1)
        samps = np.array(samps)[nan_stat]

        if pheno_mat.shape[1] == 1:
            pheno_mat = pheno_mat.ravel()

        return self.omic_loc(samps, genes), pheno_mat[nan_stat]

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""


class TransferCohort(Cohort):
    """Multiple -omic datasets paired with phenotypes to predict.

    Note that all datasets included in this class are to come from the same
    cohort of samples for which the phenotypes are defined. Thus each -omic
    dataset must have the same number of rows (samples), but can have
    varying numbers of columns (genes) depending on the nature of the -omic
    measurement in question.

    For example, a `TransferCohort` could consist of RNA sequencing and copy
    number alteration measurements taken from the same cohort of TCGA
    samples.

    Args:
        omic_mats (:obj:`list` or :obj:`dict` of :obj:`pd.DataFrame`)

    Attributes:
        omic_mats (:obj:`dict` of :obj:`pd.DataFrame`)

    """
    
    def __init__(self,
                 omic_mats, train_samps, test_samps,
                 cohort_lbl, cv_seed):

        if not isinstance(omic_mats, dict):
            raise TypeError("`omic_mats` must be a dictionary, found {} "
                            "instead!".format(type(omic_mats)))

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to the -omic datasets
        for coh, omic_mat in omic_mats.items():
            if not (set(train_samps[coh]) & set(omic_mat.index)):
                raise CohortError(
                    "At least one training sample must be in the -omic "
                    "dataset for cohort {}!".format(coh)
                    )

            if (test_samps[coh] is not None
                    and not set(test_samps[coh]) & set(omic_mat.index)):
                raise CohortError(
                    "At least one testing sample must be in the -omic "
                    "dataset for cohort {}!".format(coh)
                    )

        # check that the samples listed in the training and testing
        # sub-cohorts are valid relative to each other
        self.samples = dict()
        self.train_samps = dict()
        self.test_samps = dict()

        for coh in omic_mats:
            self.train_samps[coh] = frozenset(train_samps[coh])

            if not train_samps[coh]:
                raise CohortError("There must be at least one training "
                                  "sample in cohort {}!".format(coh))

            if (test_samps[coh] is not None
                    and set(train_samps[coh]) & set(test_samps[coh])):
                raise CohortError(
                    "Training sample set and testing sample set must be "
                    "disjoint for cohort {}!".format(coh)
                    )

            # when we don't have a testing cohort, use entire the entire
            # dataset as the training cohort
            if test_samps[coh] is None:
                self.samples[coh] = self.train_samps[coh].copy()

            # when we have a training cohort and a testing cohort
            else:
                self.samples[coh] = (frozenset(train_samps[coh])
                                     | frozenset(test_samps[coh]))
                self.test_samps[coh] = frozenset(test_samps[coh])

        self.omic_mats = {coh: omic_mat.loc[self.samples[coh],
                                            ~omic_mat.columns.duplicated()]
                          for coh, omic_mat in omic_mats.items()}

        super().__init__({coh: set(omic_mat.columns)
                          for coh, omic_mat in omic_mats.items()},
                         cohort_lbl, cv_seed)

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
                genes = {lbl: gns & set(include_genes)
                         for lbl, gns in self.genes.items()}
            else:
                genes = {lbl: gns & set(in_gns) for (lbl, gns), in_gns in
                         zip(self.genes.items(), include_genes)}

        # removes genes from the list of genes to retrieve
        if exclude_genes is not None:
            if isinstance(list(exclude_genes)[0], str):
                genes = {lbl: gns - set(exclude_genes)
                         for lbl, gns in self.genes.items()}
            else:
                genes = {lbl: gns - set(ex_gns) for (lbl, gns), ex_gns in
                         zip(self.genes.items(), exclude_genes)}

        return {lbl: sorted(gns) for lbl, gns in genes.items()}

    def omic_loc(self, samps=None, genes=None):
        """Retrieves a subset of each -omic dataset's
           samples and/or genetic features."""

        if samps is None:
            samps = self.samples.copy()
        if genes is None:
            genes = self.genes.copy()

        return {lbl: self.omic_mats[lbl].loc[samps, genes[lbl]]
                for lbl in self.omic_mats}

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""


class PresenceCohort(Cohort):
    """An -omic dataset used to predict the presence of binary phenotypes.
    
    This class is used to predict features such as the presence of a
    particular type of variant or copy number alteration, the presence of a
    binarized drug response, etc.
    """

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the binary labels corresponding to the presence of
           a phenotype for each of the samples in the training sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`bool`)
        """

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the binary labels corresponding to the presence of
           a phenotype for each of the samples in the testing sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`bool`)
        """

    # TODO: extend this to TransferCohorts?
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
    def train_pheno(self, pheno, samps=None):
        """Returns the values of a phenotype for each of the samples
           in the training sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`float`)
        """

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values of a phenotype for each of the samples
           in the testing sub-cohort.

        Returns:
            pheno_vec (:obj:`list` of :obj:`float`)
        """

