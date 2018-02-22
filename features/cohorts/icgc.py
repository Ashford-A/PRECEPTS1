
from .base import UniCohort
from .mut import VariantCohort as BaseVarCohort
from ..expression import get_expr_icgc, log_norm_expr
from ..variants import *

from ..pathways import *
from ..annot import get_gencode
from ..utils import match_icgc_samples

import numpy as np


class VariantCohort(BaseVarCohort):

    def __init__(self,
                 data_dir, cohort, mut_genes=None, top_genes=100,
                 samp_cutoff=None, mut_levels=('Gene', 'Form'),
                 cv_seed=None, cv_prop=2.0/3):

        # load expression and mutation data for the given cohort from the
        # directory where the data was downloaded
        expr = get_expr_icgc(cohort, data_dir)
        variants = get_variants_icgc(cohort, data_dir)

        # if a list of mutated genes was given, load pathway information
        # about the neighbourhood of those genes
        if mut_genes is None:
            self.path = None
        else:
            self.path = get_gene_neighbourhood(mut_genes)

        # load annotation data for the genes in the expression dataset
        annot = get_gencode()
        annot = {g: a for g, a in annot.items()
                 if g in expr.columns}

        # get gene names from the annotation data for genes whose
        # mutations we want to consider
        if mut_genes is None:
            annot_genes = {a['gene_name']: g for g, a in annot.items()}

        else:
            annot_genes = {a['gene_name']: g for g, a in annot.items()
                           if a['gene_name'] in mut_genes}

        # save gene annotation data as attributes of the cohort
        self.annot = annot
        self.mut_annot = annot_genes

        # match the samples in the expression dataset to the samples
        # in the mutation dataset
        matched_samps = match_icgc_samples(expr.index, variants['Sample'],
                                           cohort, data_dir)

        # get the names of the samples we will use and the corresponding
        # sample names in the expression and mutation datasets
        use_samples = [samp for samp, _ in matched_samps]
        expr_samps = [samp for (_, (samp, _)) in matched_samps]
        var_samps = [samp for (_, (_, samp)) in matched_samps]

        expr = expr.loc[expr_samps, annot.keys()]
        expr.index = [use_samples[expr_samps.index(samp)]
                      for samp in expr.index]
        expr.columns = [annot[gn]['gene_name'] for gn in expr.columns]

        if mut_genes is None:
            var_df = variants.loc[~variants['Gene'].isnull(), :]
            var_df = var_df.loc[var_df['Sample'].isin(var_samps), :]
            var_df = var_df.loc[~var_df['Form'].isin(
                ['intron_variant', 'upstream_gene_variant',
                 'downstream_gene_variant', 'intragenic_variant']), :]

            gn_counts = var_df.groupby(by='Gene').Sample.nunique()
            gn_counts = gn_counts.loc[annot_genes.values()]

            if samp_cutoff is None:
                gn_counts = gn_counts.sort_values(ascending=False)
                cutoff_mask = ([True] * top_genes
                               + [False] * (len(gn_counts) - top_genes))

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

            gn_counts = gn_counts[cutoff_mask]
            variants = var_df.loc[var_df['Gene'].isin(gn_counts.index), :]

        else:
            use_indx = variants['Gene'].isin(annot_genes.values())
            use_indx &= variants['Sample'].isin(var_samps)
            variants = variants.loc[use_indx, :]

        new_samps = [use_samples[var_samps.index(samp)]
                     for samp in variants['Sample']]
        new_gns = [annot[gn]['gene_name'] for gn in variants['Gene']] 

        variants = variants.drop(labels=['Sample', 'Gene'],
                                 axis="columns", inplace=False)
        variants['Sample'] = new_samps
        variants['Gene'] = new_gns

        expr = log_norm_expr(expr.fillna(expr.min().min()))
        expr_mean = np.mean(expr)
        expr_var = np.var(expr)
        expr = expr.loc[:, ((expr_mean > np.percentile(expr_mean, 5))
                            | (expr_var > np.percentile(expr_var, 5)))]

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

        UniCohort.__init__(self,
                           expr, train_samps, test_samps, cohort, cv_seed)

