
"""Consolidating -omic datasets for the prediction of mutations and CNAs.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .base import *
from HetMan.features.data.pathways import *
from HetMan.features.mutations.branches import MuType
from HetMan.features.mutations.trees import MuTree

import numpy as np

from functools import reduce
from operator import and_


class BaseMutationCohort(PresenceCohort, UniCohort):
    """A single dataset used to predict mutations in genes.

    Args:
        expr (:obj:`pd.DataFrame`, shape = [n_samps, n_features])
        variants (:obj:`pd.DataFrame`, shape = [n_variants, ])
        feat_annot (dict): Annotation of the features in the expression data.
        mut_genes (:obj:`list` of :obj:`str`, optional)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`, optional)
            What variant annotation levels to consider.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Attributes:
        path (dict): Pathway Commons neighbourhood for the mutation genes.
        train_mut (.variants.MuTree): Training cohort mutations.
        test_mut (.variants.MuTree): Testing cohort mutations.

    """

    def __init__(self,
                 expr, variants, matched_samps, feat_annot,
                 mut_genes=None, mut_levels=('Gene', 'Form'), top_genes=100,
                 samp_cutoff=None, cv_prop=2.0/3, cv_seed=None):

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

        if mut_genes is None:
            self.path = None

            variants = variants.loc[variants['Sample'].isin(var_samps), :]
            var_df = variants.loc[
                ~variants['Form'].isin(
                    ['HomDel', 'HetDel', 'HetGain', 'HomGain']),
                :]

            gn_counts = var_df.groupby(by='Gene').Sample.nunique()
            gn_counts = gn_counts.loc[feat_annot.keys()]

            if samp_cutoff is None:
                gn_counts = gn_counts.sort_values(ascending=False)
                cutoff_mask = ([True] * min(top_genes, len(gn_counts))
                               + [False] * max(len(gn_counts) - top_genes, 0))

            elif isinstance(samp_cutoff, int):
                cutoff_mask = gn_counts >= samp_cutoff

            elif isinstance(samp_cutoff, float):
                cutoff_mask = gn_counts >= samp_cutoff * len(use_samples)

            elif hasattr(samp_cutoff, '__getitem__'):
                if isinstance(samp_cutoff[0], int):
                    cutoff_mask = ((samp_cutoff[0] <= gn_counts)
                                   & (samp_cutoff[1] >= gn_counts))

                elif isinstance(samp_cutoff[0], float):
                    cutoff_mask = (
                            (samp_cutoff[0] * len(use_samples) <= gn_counts)
                            & (samp_cutoff[1] * len(use_samples) >= gn_counts)
                        )

            else:
                raise ValueError("Unrecognized `samp_cutoff` argument!")

            gn_counts = gn_counts[cutoff_mask]
            variants = variants.loc[variants['Gene'].isin(gn_counts.index), :]

        else:
            self.path = get_gene_neighbourhood(mut_genes)
            variants = variants.loc[variants['Gene'].isin(mut_genes)
                                    & variants['Sample'].isin(var_samps), :]

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

        # if the cohort is to have a testing cohort, build the tree with info
        # on which testing samples have which types of mutations
        if test_samps:
            self.test_mut = MuTree(
                muts=variants.loc[variants['Sample'].isin(test_samps), :],
                levels=mut_levels
                )

        else:
            test_samps = None

        # likewise, build a representation of mutation types across
        # training cohort samples
        self.train_mut = MuTree(
            muts=variants.loc[variants['Sample'].isin(train_samps), :],
            levels=mut_levels
            )

        self.mut_genes = mut_genes
        self.cv_prop = cv_prop

        super().__init__(expr, train_samps, test_samps, cv_seed)

    def train_pheno(self, mtype, samps=None):
        """Gets the mutation status of samples in the training cohort.

        Args:
            mtype (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the training cohort
                will be ignored. Defaults to using all the training samples.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)

        """

        # uses all the training samples if no list of samples provided
        if samps is None:
            samps = self.train_samps

        # filters out the provided samples not in the training cohort
        else:
            samps = set(samps) & self.train_samps

        if isinstance(mtype, MuType):
            stat_list = self.train_mut.status(samps, mtype)

        elif isinstance(tuple(mtype)[0], MuType):
            stat_list = [self.train_mut.status(samps, mtp)
                         for mtp in sorted(mtype)]

        else:
            raise TypeError("A VariantCohort accepts only MuTypes or lists "
                            "of MuTypes as training phenotypes!")

        return stat_list

    def test_pheno(self, mtype, samps=None):
        """Gets the mutation status of samples in the testing cohort.

        Args:
            mtype (:obj:`MuType` or :obj:`list` of :obj:`MuType`)
                A particular type of mutation or list of types.
            samps (:obj:`list` of :obj:`str`, optional)
                A list of samples, of which those not in the testing cohort
                will be ignored. Defaults to using all the testing samples.

        Returns:
            stat_list (:obj:`list` of :obj:`bool`)

        """

        # uses all the testing samples if no list of samples provided
        if samps is None:
            samps = self.test_samps

        # filters out the provided samples not in the testing cohort
        else:
            samps = set(samps) & self.test_samps

        if isinstance(mtype, MuType):
            stat_list = self.test_mut.status(samps, mtype)

        elif isinstance(tuple(mtype)[0], MuType):
            stat_list = [self.test_mut.status(samps, mtp)
                         for mtp in sorted(mtype)]

        else:
            raise TypeError("A VariantCohort accepts only MuTypes or lists "
                            "of MuTypes as testing phenotypes!")

        return stat_list


class BaseTransferMutationCohort(PresenceCohort, TransferCohort):
    """Mutiple datasets used to predict mutations using transfer learning.

    Args:
        expr_dict (:obj:`dict` of :obj:`pd.DataFrame`)
        variant_dict (:obj:`dict` of :obj:`pd.DataFrame`)

    """

    def __init__(self,
                 expr_dict, variant_dict, matched_samps, feat_annot,
                 mut_genes=None, mut_levels=('Gene', 'Form'), top_genes=250,
                 samp_cutoff=None, cv_prop=2.0/3, cv_seed=None):

        use_samples = {coh: [samp for samp, _ in mtchs]
                       for coh, mtchs in matched_samps.items()}
        expr_samps = {coh: [samp for (_, (samp, _)) in mtchs]
                      for coh, mtchs in matched_samps.items()}
        var_samps = {coh: [samp for (_, (_, samp)) in mtchs]
                     for coh, mtchs in matched_samps.items()}

        expr = {
            coh: expr_dict[coh].loc[
                expr_samps[coh], list(feat_annot[coh])]
            for coh in expr_dict
            }

        for coh in expr_dict:
            expr[coh].index = [use_samples[coh][expr_samps[coh].index(samp)]
                               for samp in expr[coh].index]

        if mut_genes is None:
            self.path = None

            variants = {coh: var.loc[var['Sample'].isin(var_samps[coh]), :]
                        for coh, var in variant_dict.items()}

            var_df = {
                coh: var.loc[~var['Form'].isin(['HomDel', 'HetDel',
                                                'HetGain', 'HomGain']), :]
                for coh, var in variants.items()
                }

            gn_counts = {
                coh: var.groupby(by='Gene').Sample.nunique().loc[
                    feat_annot[coh].keys()]
                for coh, var in var_df.items()
                }

            if samp_cutoff is None:
                use_counts = {
                    coh: gn_cnts.sort_values(ascending=False)[:top_genes]
                    for coh, gn_cnts in gn_counts.items()
                    }

            elif isinstance(samp_cutoff, int):
                use_counts = {coh: gn_cnts[gn_cnts >= samp_cutoff]
                              for coh, gn_cnts in gn_counts.items()}

            elif isinstance(samp_cutoff, float):
                use_counts = {
                    coh: gn_cnts[
                        gn_cnts >= samp_cutoff * len(use_samples[coh])]
                    for coh, gn_cnts in gn_counts.items()
                    }

            elif hasattr(samp_cutoff, '__getitem__'):
                if isinstance(samp_cutoff[0], int):
                    use_counts = {
                        coh: gn_cnts[(samp_cutoff[0] <= gn_cnts)
                                     & (samp_cutoff[1] >= gn_cnts)]
                        for coh, gn_cnts in gn_counts.items()
                        }

                elif isinstance(samp_cutoff[0], float):
                    use_counts = {
                        coh: gn_cnts[(samp_cutoff[0]
                                      * len(use_samples[coh]) <= gn_cnts)
                                     & (samp_cutoff[1]
                                        * len(use_samples[coh]) >= gn_cnts)]
                        for coh, gn_cnts in gn_counts.items()
                        }

            else:
                raise ValueError("Unrecognized `samp_cutoff` argument!")

            use_gns = reduce(and_,
                             [cnts.index for cnts in use_counts.values()])
            variants = {coh: var.loc[var['Gene'].isin(use_gns), :]
                        for coh, var in variants.items()}

        else:
            self.path = get_gene_neighbourhood(mut_genes)
            variants = {coh: var.loc[var['Gene'].isin(mut_genes)
                                     & var['Sample'].isin(var_samps[coh]), :]
                        for coh, var in variant_dict.items()}

        for coh in variants:
            new_samps = [use_samples[coh][var_samps[coh].index(samp)]
                         for samp in variants[coh]['Sample']]

            variants[coh] = variants[coh].drop(
                labels=['Sample'], axis="columns", inplace=False)
            variants[coh]['Sample'] = new_samps

        # filters out genes that have both low levels of expression and low
        # variance of expression
        expr_mean = {coh: np.mean(expr[coh]) for coh in expr_dict}
        expr_var = {coh: np.var(expr[coh]) for coh in expr_dict}
        expr = {coh: expr[coh].loc[
                         :,
                         ((expr_mean[coh] > np.percentile(expr_mean[coh], 5))
                          | (expr_var[coh] > np.percentile(expr_var[coh],
                                                           5)))]
                    for coh in expr_dict}

        split_samps = {coh: self.split_samples(cv_seed, cv_prop,
                                               use_samples[coh])
                       for coh in expr_dict}

        train_samps = {coh: samps[0] for coh, samps in split_samps.items()}
        test_samps = {coh: samps[1] for coh, samps in split_samps.items()}

        self.train_mut = dict()
        self.test_mut = dict()
        for coh in expr_dict:

            if test_samps[coh]:
                self.test_mut[coh] = MuTree(
                    muts=variants[coh].loc[
                         variants[coh]['Sample'].isin(test_samps[coh]), :],
                    levels=mut_levels
                    )

            else:
                test_samps[coh] = None

            self.train_mut[coh] = MuTree(
                muts=variants[coh].loc[
                     variants[coh]['Sample'].isin(train_samps[coh]), :],
                levels=mut_levels
                )

        self.mut_genes = mut_genes
        self.cv_prop = cv_prop

        super().__init__(expr, train_samps, test_samps, cv_seed)

    @classmethod
    def combine_cohorts(cls, *cohorts, **named_cohorts):
        new_cohort = TransferCohort.combine_cohorts(*cohorts, **named_cohorts)
        new_cohort.__class__ = cls

        new_cohort.train_mut = dict()
        new_cohort.test_mut = dict()

        for cohort in cohorts:
            new_cohort.train_mut[cohort.cohort] = cohort.train_mut

            if hasattr(cohort, "test_mut"):
                new_cohort.test_mut[cohort.cohort] = cohort.test_mut

        for lbl, cohort in named_cohorts.items():
            new_cohort.train_mut[lbl] = cohort.train_mut

            if hasattr(cohort, "test_mut"):
                new_cohort.test_mut[lbl] = cohort.test_mut

        return new_cohort

    def train_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.train_samps

        return {coh: self.train_mut[coh].status(samps[coh], mtype)
                for coh in samps}

    def test_pheno(self, mtype, samps=None):
        if samps is None:
            samps = self.test_samps

        return {coh: self.test_mut[coh].status(samps[coh], mtype)
                for coh in samps}
