
"""Consolidating -omic datasets to predict sample mutation frequencies.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .base import *
import numpy as np


class BaseMutFreqCohort(ValueCohort, UniCohort):
    """A single dataset used to predict sample mutation frequencies.

    Args:
        expr (:obj:`pd.DataFrame`, shape = [n_samps, n_features])
        variants (:obj:`pd.DataFrame`, shape = [n_variants, ])
        feat_annot (dict): Annotation of the features in the expression data.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    """

    def __init__(self,
                 expr, variants, matched_samps, feat_annot,
                 cv_prop=2.0/3, cv_seed=None):

        # gets the set of samples shared across the expression and mutation
        # data that are also primary tumour samples
        use_samples = [samp for samp, _ in matched_samps]
        expr_samps = [samp for (_, (samp, _)) in matched_samps]
        var_samps = [samp for (_, (_, samp)) in matched_samps]

        # gets the subset of expression data corresponding to the shared
        # samples and annotated genes, renames expression samples to the
        # shared sample names
        expr = expr.loc[expr_samps, list(feat_annot)]
        expr.index = [use_samples[expr_samps.index(samp)]
                      for samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(var_samps), :]
        new_samps = [use_samples[var_samps.index(samp)]
                     for samp in variants['Sample']]

        variants = variants.drop(labels=['Sample'], axis="columns",
                                 inplace=False)
        variants['Sample'] = new_samps

        # filters out genes that have both low levels of expression and low
        # variance of expression
        expr_mean = np.mean(expr)
        expr_var = np.var(expr)
        expr = expr.loc[:, ((expr_mean > np.percentile(expr_mean, 5))
                            | (expr_var > np.percentile(expr_var, 5)))]

        # gets subset of samples to use for training, and split the expression
        # and variant datasets accordingly into training/testing cohorts
        train_samps, test_samps = self.split_samples(
            cv_seed, cv_prop, use_samples)

        # calculates the number of mutated genes each sample has
        self.mut_freqs = variants.loc[
            ~variants['Form'].isin(
                ['HomDel', 'HetDel', 'HetGain', 'HomGain']),
                :].groupby(by='Sample').Gene.nunique()

        self.cv_prop = cv_prop
        super().__init__(expr, train_samps, test_samps, cv_seed)

    def train_pheno(self, samps=None):
        """Gets the mutation frequency of samples in the training cohort.

        Returns:
            freq_list (:obj:`list` of :obj:`int`)

        """

        # uses all the training samples if no list of samples provided
        if samps is None:
            samps = self.train_samps

        # filters out the provided samples not in the training cohort
        else:
            samps = set(samps) & self.train_samps

        return self.mut_freqs.loc[sorted(samps)].tolist()

    def test_pheno(self, samps=None):
        """Gets the mutation frequency of samples in the testing cohort.

        Returns:
            freq_list (:obj:`list` of :obj:`int`)

        """

        # uses all the training samples if no list of samples provided
        if samps is None:
            samps = self.test_samps

        # filters out the provided samples not in the training cohort
        else:
            samps = set(samps) & self.test_samps

        return self.mut_freqs.loc[sorted(samps)].tolist()

