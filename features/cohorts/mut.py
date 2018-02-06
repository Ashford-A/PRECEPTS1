
"""Consolidating -omic datasets for prediction of phenotypes.

This module contains classes for grouping continuous -omic datasets such as
expression or proteomic measurements with -omic phenotypic features such as
variants, copy number alterations, or drug response data so that the former
can be used to predict the latter using machine learning pipelines.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>
         Hannah Manning <manningh@ohsu.edu>

"""

from .base import *

from ..expression import *
from ..variants import *
from ..copies import get_copies_firehose

from ..pathways import *
from ..annot import get_gencode
from ..utils import match_tcga_samples

import numpy as np
import pandas as pd

from functools import reduce
from itertools import cycle


class VariantCohort(PresenceCohort, UniCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        mut_genes (:obj:`list` of :obj:`str`)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`)
            What variant annotation levels to consider.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant data from.
        cv_seed (int): The random seed to use for cross-validation sampling.
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
        >>>     cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )
        >>>
        >>> cdata2 = VariantCohort(
        >>>     cohort='TCGA-PAAD', mut_genes=['KRAS'],
        >>>     mut_levels=['Form_base', 'Exon', 'Location'],
        >>>     expr_source='Firehose', data_dir='../input-data/firehose',
        >>>     cv_seed=98, cv_prop=0.8, syn=syn
        >>>     )

    """

    def __init__(self,
                 cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 expr_source='BMEG', var_source='mc3',
                 cv_seed=None, cv_prop=2.0/3, **coh_args):

        # loads gene expression data from the given source, log-normalizes it
        # and removes missing values as necessary
        if expr_source == 'BMEG':
            expr_mat = get_expr_bmeg(cohort)
            expr = log_norm_expr(expr_mat.fillna(expr_mat.min().min()))

        elif expr_source == 'Firehose':
            expr_mat = get_expr_firehose(cohort, coh_args['data_dir'])
            expr = expr_mat.fillna(expr_mat.min().min())

        else:
            raise ValueError("Unrecognized source of expression data!")

        # loads gene variant data from the given source
        if var_source is None:
            var_source = expr_source

        if var_source == 'mc3':
            variants = get_variants_mc3(coh_args['syn'])

        elif var_source == 'Firehose':
            variants = get_variants_firehose(cohort, coh_args['data_dir'])

        else:
            raise ValueError("Unrecognized source of variant data!")

        # loads the pathway neighbourhood of the variant genes, as well as
        # annotation data for all genes
        self.path = get_gene_neighbourhood(mut_genes)
        annot = get_gencode()

        # gets the genes for which we have both expression and annotation data
        annot = {g: a for g, a in annot.items()
                 if a['gene_name'] in expr.columns}
        annot_genes = [a['gene_name'] for g, a in annot.items()]

        # gets the set of samples shared across the expression and mutation
        # data that are also primary tumour samples
        matched_samps = match_tcga_samples(expr.index, variants['Sample'])
        use_samples = [samp for samp, _ in matched_samps]
        expr_samps = [samp for (_, (samp, _)) in matched_samps]
        var_samps = [samp for (_, (_, samp)) in matched_samps]

        # gets the subset of expression data corresponding to the shared
        # samples and annotated genes
        expr = expr.loc[expr.index.isin(expr_samps), annot_genes]
        expr.index = [use_samples[expr_samps.index(samp)]
                      for samp in expr.index]

        # gets the subset of variant data for the shared samples with the
        # genes whose mutations we want to consider
        variants = variants.loc[variants['Gene'].isin(mut_genes)
                                & variants['Sample'].isin(var_samps), :]
        variants['Sample'] = [use_samples[var_samps.index(samp)]
                              for samp in variants['Sample']]

        # filters out genes that have both low levels of expression and low
        # variance of expression
        expr_mean = np.mean(expr)
        expr_var = np.var(expr)
        expr = expr.loc[:, ((expr_mean > np.percentile(expr_mean, 5))
                            | (expr_var > np.percentile(expr_var, 5)))]

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
        if samps is None:
            samps = self.test_samps
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


class TransferVariantCohort(PresenceCohort, TransferCohort):
    """Sharing information across multiple cohorts to predict variants."""

    def __init__(self,
                 cohorts, mut_genes, mut_levels=('Gene', 'Form'),
                 expr_sources='BMEG', var_sources='mc3',
                 cv_seed=None, cv_props=2.0/3, **coh_args):

        if isinstance(expr_sources, str):
            expr_sources = [expr_sources]

        if var_sources is None:
            var_sources = expr_sources
        elif isinstance(var_sources, str):
            var_sources = [var_sources]

        if not isinstance(cv_props, list):
            cv_props = [cv_props]

        # loads gene expression data from the given source, log-normalizes it
        # and removes missing values as necessary
        expr = dict()
        for coh, expr_source in zip(cohorts, cycle(expr_sources)):
            if expr_source == 'BMEG':
                expr_mat = get_expr_bmeg(coh)
                expr[coh] = log_norm_expr(
                    expr_mat.fillna(expr_mat.min().min()))

            elif expr_source == 'Firehose':
                expr_mat = get_expr_firehose(coh, coh_args['data_dir'])
                expr[coh] = expr_mat.fillna(expr_mat.min().min())

            else:
                raise ValueError("Unrecognized source of expression data!")

        # loads gene variant data from the given source
        if 'mc3' in var_sources:
            mc3_vars = get_variants_mc3(coh_args['syn'])

        variants = dict()
        for coh, var_source in zip(cohorts, cycle(var_sources)):

            if var_source == 'mc3':
                variants[coh] = mc3_vars.copy()

            elif var_source == 'Firehose':
                variants[coh] = get_variants_firehose(
                    coh, coh_args['data_dir'])

            else:
                raise ValueError("Unrecognized source of variant data!")

        # loads the pathway neighbourhood of the variant genes, as well as
        # annotation data for all genes
        self.path = get_gene_neighbourhood(mut_genes)
        annot = get_gencode()

        # gets the genes for which we have both expression and annotation data
        annot = {coh: {g: a for g, a in annot.items()
                       if a['gene_name'] in expr[coh].columns}
                 for coh in cohorts}
        annot_genes = {coh: [a['gene_name'] for g, a in annot[coh].items()]
                       for coh in cohorts}

        # gets the set of samples shared across the expression and mutation
        # data that are also primary tumour samples
        matched_samps = {
            coh: match_tcga_samples(expr[coh].index, variants[coh]['Sample'])
            for coh in cohorts
            }

        use_samples = {coh: [samp for samp, _ in matched_samps[coh]]
                       for coh in cohorts}
        expr_samps = {coh: [samp for (_, (samp, _)) in matched_samps[coh]]
                      for coh in cohorts}
        var_samps = {coh: [samp for (_, (_, samp)) in matched_samps[coh]]
                     for coh in cohorts}

        # gets the subset of expression data corresponding to the shared
        # samples and annotated genes
        expr = {coh: expr[coh].loc[expr[coh].index.isin(expr_samps[coh]),
                                   annot_genes[coh]]
                for coh in cohorts}

        for coh in cohorts:
            expr[coh].index = [use_samples[coh][expr_samps[coh].index(samp)]
                               for samp in expr[coh].index]

        # gets the subset of variant data for the shared samples with the
        # genes whose mutations we want to consider
        variants = {coh: variants[coh].loc[
                         variants[coh]['Gene'].isin(mut_genes)
                         & variants[coh]['Sample'].isin(var_samps[coh]), :]
                    for coh in cohorts}

        for coh in cohorts:
            variants[coh]['Sample'] = [
                use_samples[coh][var_samps[coh].index(samp)]
                for samp in variants[coh]['Sample']
                ]

        # filters out genes that have both low levels of expression and low
        # variance of expression
        expr_mean = {coh: np.mean(expr[coh]) for coh in cohorts}
        expr_var = {coh: np.var(expr[coh]) for coh in cohorts}
        expr = {coh: expr[coh].loc[
                     :, ((expr_mean[coh] > np.percentile(expr_mean[coh], 5))
                         | (expr_var[coh] > np.percentile(expr_var[coh], 5)))]
                for coh in cohorts}

        # gets annotation data for the genes whose mutations
        # are under consideration
        annot_data = {
            coh: {a['gene_name']: {'ID': g, 'Chr': a['chr'],
                                   'Start': a['Start'], 'End': a['End']}
                  for g, a in annot[coh].items()
                  if a['gene_name'] in mut_genes}
            for coh in cohorts
            }

        self.annot = annot
        self.mut_annot = annot_data

        # gets subset of samples to use for training, and split the expression
        # and variant datasets accordingly into training/testing cohorts
        split_samps = {
            coh: self.split_samples(cv_seed, cv_prop, use_samples[coh])
            for coh, cv_prop in zip(cohorts, cycle(cv_props))
            }

        train_samps = {coh: samps[0] for coh, samps in split_samps.items()}
        test_samps = {coh: samps[1] for coh, samps in split_samps.items()}

        self.train_mut = dict()
        self.test_mut = dict()
        for coh in cohorts:

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
        self.cv_props = cv_props

        super().__init__(expr, train_samps, test_samps,
                         '_'.join(sorted(cohorts)), cv_seed)

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

