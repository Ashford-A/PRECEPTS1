
"""Consolidating datasets from the International Cancer Genome Consortium.

Authors: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from .mut import BaseMutationCohort
from ..expression import get_expr_icgc
from ..variants import get_variants_icgc

from ..annot import get_gencode
from ..utils import log_norm, match_icgc_samples


class MutationCohort(BaseMutationCohort):

    def __init__(self,
                 cohort, data_dir, mut_genes,
                 mut_levels=('Gene', 'Form_base'), top_genes=100,
                 samp_cutoff=None, cv_prop=2.0/3, cv_seed=None):

        # load ICGC expression and mutation data for the given cohort from
        # the given local directory
        expr = get_expr_icgc(cohort, data_dir)
        variants = get_variants_icgc(cohort, data_dir)

        # load gene annotation data, find the genes for which we have both
        # expression and annotation data
        annot = get_gencode()
        use_genes = annot.keys() & expr.columns

        # remove genes from the expression data for which we do not have
        # annotation data, log-normalize the expression values
        expr = expr.loc[:, use_genes]
        expr = log_norm(expr.fillna(0.0))

        # replace the gene Ensembl IDs with gene names in the expression
        # data, reformat annotation data to use gene names as keys
        expr.columns = [annot[ens]['gene_name'] for ens in expr.columns]
        gene_annot = {annot[ens]['gene_name']: {**{'Ens': ens}, **annot[ens]}
                      for ens in use_genes}

        # match the samples in the expression dataset to the samples
        # in the mutation dataset
        matched_samps = match_icgc_samples(expr.index, variants['Sample'],
                                           cohort, data_dir)

        # remove entries in the mutation data that are not associated with a
        # gene for which we have annotation data
        var_df = variants.loc[~variants['Gene'].isnull(), :]
        var_df = var_df.loc[var_df['Gene'].isin(annot.keys()), :]

        # map mutation types to those used by MC3
        var_df['Form'] = var_df['Form'].map(
            {'missense_variant': 'Missense_Mutation', 
             '3_prime_UTR_variant': "3'UTR", '5_prime_UTR_variant': "5'UTR",
             'frameshift_variant': 'Frame_Shift',
             'stop_gained': 'Nonsense_Mutation',
             'disruptive_inframe_deletion': 'In_Frame',
             'inframe_deletion': 'In_Frame', 'inframe_insertion': 'In_Frame',
             'splice_donor_variant': 'Splice_Site',
             'splice_acceptor_variant': 'Splice_Site',
             'splice_region_variant': 'Splice_Site',
             'synonymous_variant': 'Silent', 'exon_variant': 'Silent',
             '5_prime_UTR_premature_start_codon_gain_variant': "5'Flank",
             'start_lost': "5'UTR", 'stop_lost': "3'UTR",
             'stop_retained_variant': "3'UTR",
             'disruptive_inframe_insertion': 'In_Frame',
             'initiator_codon_variant': "5'UTR"}
            )

        # remove mutations whose types couldn't be mapped 
        variants = var_df.loc[~var_df['Form'].isnull(), :]

        # replace gene Ensembl IDs with gene names in the mutation data
        new_gns = [annot[gn]['gene_name'] for gn in variants['Gene']] 
        variants = variants.drop(labels=['Gene'], axis="columns",
                                 inplace=False)
        variants['Gene'] = new_gns

        super().__init__(expr, variants, matched_samps, gene_annot, mut_genes,
                         mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed)

