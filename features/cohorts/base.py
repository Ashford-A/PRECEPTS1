
"""Consolidating -omic datasets for prediction of phenotypes.

This module contains abstract classes for grouping continuous -omic datasets
such as expression or proteomic levels with -omic phenotypic features such as
variants, copy number alterations, and drug response measurements so that the
former can be used to predict the latter using machine learning pipelines.

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
    """Base abstract class for -omic datasets used in machine learning.

    This class consists of a dataset of -omic measurements collected for a
    collection of samples over a set of genetic features. The samples are
    divided into training and testing sub-cohorts for use in the evaluation of
    machine learning models. The nature of these -omic measurements and the
    phenotypes the models will be used to predict are defined by
    children classes.

    Args:
        omic_data : An -omic dataset or collection thereof.
        train_samps, test_samps : Subsets of samples in each -omic dataset.
        genes : The genetic features included in each -omic dataset.
        cv_seed (int, optional)
            A random seed used for sampling from the datasets.

    """

    def __init__(self,
                 omic_data, train_samps, test_samps, genes, cv_seed=None):

        self.omic_data = omic_data
        self.train_samps = train_samps
        self.test_samps = test_samps
        self.genes = genes

        if cv_seed is None:
            self.cv_seed = 0
        else:
            self.cv_seed = cv_seed

    def train_data(self,
                   pheno,
                   include_samps=None, exclude_samps=None,
                   include_genes=None, exclude_genes=None):
        """Retrieval of the training cohort from the -omic dataset."""

        samps = self.subset_samps(include_samps, exclude_samps,
                                  use_test=False)
        genes = self.subset_genes(include_genes, exclude_genes)

        pheno, samps = self.parse_pheno(
            self.train_pheno(pheno, samps), samps)

        return self.omic_loc(samps, genes), pheno

    def test_data(self,
                  pheno,
                  include_samps=None, exclude_samps=None,
                  include_genes=None, exclude_genes=None):
        """Retrieval of the testing cohort from the -omic dataset."""

        samps = self.subset_samps(include_samps, exclude_samps,
                                  use_test=True)
        genes = self.subset_genes(include_genes, exclude_genes)

        pheno, samps = self.parse_pheno(
            self.test_pheno(pheno, samps), samps)

        return self.omic_loc(samps, genes), pheno

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
    def subset_samps(self,
                     include_samps=None, exclude_samps=None, use_test=False):
        """Get a subset of the samples available in this cohort."""

        if use_test:
            return self.test_samps
        else:
            return self.train_samps

    @abstractmethod
    def subset_genes(self, include_genes=None, exclude_genes=None):
        """Get a subset of the genetic features available in this cohort."""

        return self.genes

    @abstractmethod
    def omic_loc(self, samps=None, genes=None):
        """Retrieval of a subset of the -omic dataset."""

        return self.omic_data

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""

        return np.array([])

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""

        return np.array([])

    def parse_pheno(self, pheno, samps):
        pheno = np.array(pheno)

        if pheno.ndim == 1:
            pheno_mat = pheno.reshape(-1, 1)

        elif pheno.shape[1] == len(samps):
            pheno_mat = np.transpose(pheno)

        elif pheno.shape[0] != len(samps):
            raise ValueError("Given phenotype(s) do not return a valid "
                             "matrix of values to predict!")

        else:
            pheno_mat = pheno.copy()

        nan_stat = np.any(~np.isnan(pheno_mat), axis=1)
        samps_use = np.array(samps)[nan_stat]

        if pheno_mat.shape[1] == 1:
            pheno_mat = pheno_mat.ravel()

        return pheno_mat[nan_stat], samps_use


class UniCohort(Cohort):
    """An -omic dataset from one source for use in predicting phenotypes.

    This class consists of a dataset of -omic measurements collected for a
    collection of samples coming from a single context, such as TCGA-BRCA
    or ICGC PACA-AU.

    Args:
        omic_mat (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])
        train_samps, test_samps
            Subsets of samples in the index of the -omic data frame.
        cv_seed (int, optional)
            A random seed used for sampling from the dataset.

    """

    def __init__(self, omic_mat, train_samps, test_samps, cv_seed=None):

        if not isinstance(omic_mat, pd.DataFrame):
            raise TypeError("`omic_mat` must be a pandas DataFrame, found "
                            "{} instead!".format(type(omic_mat)))

        genes = tuple(omic_mat.columns)
        if isinstance(genes[0], str):
            genes = frozenset(genes)

        elif isinstance(genes[0], tuple):
            genes = frozenset(x[0] for x in genes)

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
        if not train_samps:
            raise CohortError("There must be at least one training sample!")

        if test_samps is not None and set(train_samps) & set(test_samps):
            raise CohortError("Training sample set and testing sample"
                              "set must be disjoint!")

        # when we don't have a testing cohort, use entire the entire
        # dataset as the training cohort
        train_samps = frozenset(train_samps)
        if test_samps is None:
            self.samples = train_samps.copy()

        # when we have a training cohort and a testing cohort
        else:
            self.samples = frozenset(train_samps) | frozenset(test_samps)
            test_samps = frozenset(test_samps)

        # remove duplicate features from the dataset as well as samples
        # not listed in either the training or testing sub-cohorts
        omic_mat = omic_mat.loc[self.samples, ~omic_mat.columns.duplicated()]
        super().__init__(omic_mat, train_samps, test_samps, genes, cv_seed)

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

        return self.omic_data.loc[samps, list(genes)]

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""

        raise CohortError("Cannot use UniCohort class!")

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""

        raise CohortError("Cannot use UniCohort class!")


class TransferCohort(Cohort):
    """Multiple -omic datasets for use in transfer learning of phenotypes.

    This class consists of multiple datasets of -omic measurements, each for a
    different set of samples coming from a unique context, which will
    nevertheless be used to predict phenotypes common between the contexts.

    Args:
        omic_mats (:obj:`dict` of :obj:`pd.DataFrame`)
        train_samps, test_samps (dict)
            Subsets of samples in the index of each -omic data frame.
        cv_seed (int, optional)
            A random seed used for sampling from the dataset.

    """
    
    def __init__(self, omic_mats, train_samps, test_samps, cv_seed=None):

        if not isinstance(omic_mats, dict):
            raise TypeError("`omic_mats` must be a dictionary, found {} "
                            "instead!".format(type(omic_mats)))

        omic_dict = dict()
        genes = dict()
        self.samples = dict()

        for coh, omic_mat in omic_mats.items():
            if not isinstance(omic_mat, pd.DataFrame):
                raise TypeError(
                    "`omic_mats` must have pandas DataFrames as values, "
                    "found {} instead for cohort {}!".format(
                        type(omic_mat), coh)
                    )

            coh_genes = tuple(omic_mat.columns)
            if isinstance(coh_genes[0], str):
                genes[coh] = frozenset(coh_genes)

            elif isinstance(coh_genes[0], tuple):
                genes[coh] = frozenset(x[0] for x in coh_genes)

            # check that the samples listed in the training and testing
            # sub-cohorts are valid relative to the -omic datasets
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
            train_samps[coh] = frozenset(train_samps[coh])
            if test_samps[coh] is None:
                self.samples[coh] = train_samps[coh].copy()

            # when we have a training cohort and a testing cohort
            else:
                self.samples[coh] = (frozenset(train_samps[coh])
                                     | frozenset(test_samps[coh]))
                test_samps[coh] = frozenset(test_samps[coh])

            omic_dict[coh] = omic_mats[coh].loc[
                self.samples[coh], ~omic_mats[coh].columns.duplicated()]

        super().__init__(omic_dict, train_samps, test_samps, genes, cv_seed)

    def subset_samps(self,
                     include_samps=None, exclude_samps=None, use_test=False):

        if use_test:
            coh_samps = self.test_samps.copy()
        else:
            coh_samps = self.train_samps.copy()

        # decide what samples to use based on exclusion and inclusion criteria
        if include_samps is not None:
            coh_samps = {coh: samps & set(include_samps[coh])
                         for coh, samps in coh_samps.items()}

        if exclude_samps is not None:
            coh_samps = {coh: samps - set(exclude_samps[coh])
                         for coh, samps in coh_samps.items()}

        return {coh: sorted(samps) for coh, samps in coh_samps.items()}

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

        return {lbl: self.omic_data[lbl].loc[samps[lbl], genes[lbl]]
                for lbl in self.omic_data}

    @abstractmethod
    def train_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the training sub-cohort."""

    @abstractmethod
    def test_pheno(self, pheno, samps=None):
        """Returns the values for a phenotype in the testing sub-cohort."""

    def parse_pheno(self, pheno, samps):

        parse_dict = {
            coh: super(TransferCohort, self).parse_pheno(
                pheno[coh], samps[coh])
            for coh in pheno
            }

        return ({coh: phn for (coh, (phn, _)) in parse_dict.items()},
                {coh: smps for (coh, (_, smps)) in parse_dict.items()})


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
