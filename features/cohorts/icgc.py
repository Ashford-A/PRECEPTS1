
from .mut import BaseMutationCohort
from ..expression import get_expr_icgc, log_norm_expr
from ..variants import get_variants_icgc

from ..annot import get_gencode
from ..utils import match_icgc_samples


class MutationCohort(BaseMutationCohort):

    def __init__(self,
                 cohort, data_dir, mut_genes, mut_levels=('Gene', 'Form'),
                 top_genes=100, samp_cutoff=None,
                 cv_prop=2.0/3, cv_seed=None):

        # load expression and mutation data for the given cohort from the
        # directory where the data was downloaded
        expr = get_expr_icgc(cohort, data_dir)
        variants = get_variants_icgc(cohort, data_dir)

        annot = get_gencode()
        use_genes = annot.keys() & expr.columns

        expr = expr.loc[:, use_genes]
        expr = log_norm_expr(expr.fillna(expr.min().min()))

        # load annotation data for the genes in the expression dataset
        expr.columns = [annot[ens]['gene_name'] for ens in expr.columns]
        gene_annot = {annot[ens]['gene_name']: {**{'Ens': ens}, **annot[ens]}
                      for ens in use_genes}

        # match the samples in the expression dataset to the samples
        # in the mutation dataset
        matched_samps = match_icgc_samples(expr.index, variants['Sample'],
                                           cohort, data_dir)

        var_df = variants.loc[~variants['Gene'].isnull(), :]
        var_df = var_df.loc[var_df['Gene'].isin(annot.keys()), :]

        variants = var_df.loc[~var_df['Form'].isin(
            ['intron_variant', 'upstream_gene_variant',
             'downstream_gene_variant', 'intragenic_variant']
            ), :]

        new_gns = [annot[gn]['gene_name'] for gn in variants['Gene']] 
        variants = variants.drop(labels=['Gene'], axis="columns",
                                 inplace=False)
        variants['Gene'] = new_gns

        super().__init__(expr, variants, matched_samps, gene_annot, mut_genes,
                         mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed)
