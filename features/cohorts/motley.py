
from .tcga import get_expr_data as get_tcga_expr

from dryadic.features.cohorts.mut import BaseMutationCohort
from dryadic.features.cohorts.utils import (
    match_tcga_samples, get_gencode, log_norm, drop_duplicate_genes)

import pandas as pd
from functools import reduce
from operator import and_


class MutationConcatCohort(BaseMutationCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        mut_genes (:obj:`list` of :obj:`str`)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`)
            What variant annotation levels to consider.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant call data from.
        copy_source (:obj:`str`, optional)
            Where to load continuous copy number alteration data from. The
            default is to not use continuous CNA scores.
        top_genes (:obj:`int`, optional)
            How many of the genes in the cohort ordered by descending
            mutation frequency to load mutation data for.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = MutationCohort(
        >>>     cohort='BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )
        >>>
        >>> cdata2 = MutationCohort(
        >>>     cohort='PAAD', mut_genes=['KRAS'],
        >>>     mut_levels=['Form_base', 'Exon', 'Location'],
        >>>     expr_source='Firehose', expr_dir='../input-data/firehose',
        >>>     cv_seed=98, cv_prop=0.8, syn=syn
        >>>     )
        >>>
        >>> cdata3 = MutationCohort(
        >>>     cohort='LUSC', mut_genes=None, mut_levels=['Gene', 'Form'],
        >>>     expr_source='Firehose', var_source='mc3',
        >>>     copy_source='Firehose', expr_dir='../input-data/firehose',
        >>>     copy_dir='../input-data/firehose', samp_cutoff = 0.2,
        >>>     syn=syn, cv_prop=0.75, cv_seed=5601
        >>>     )
        >>>
    """

    def __init__(self,
                 cohorts, mut_genes, mut_levels,
                 expr_source, var_source, copy_source,
                 annot_file, domain_dir=None, type_file=None,
                 top_genes=100, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):
        self.cohorts = cohorts

        expr_dict = {cohort: get_expr_data(cohort, expr_source, **coh_args)
                     for cohort in cohorts}

        if 'annot_fields' in coh_args:
            annot_data = get_gencode(annot_file, coh_args['annot_fields'])
        else:
            annot_data = get_gencode(annot_file)

        # restructure annotation data around expression gene labels
        gene_annot = {cohort: {
            at['gene_name']: {**{'Ens': ens}, **at}
            for ens, at in annot_data.items()
            if at['gene_name'] in set(expr.columns.get_level_values('Gene'))
            }
            for cohort, expr in expr_dict.items()}

        data_dict = {cohort: add_variant_data(cohort, var_source, copy_source,
                                              expr, gene_annot[cohort],
                                              **coh_args)
                     for cohort, expr in expr_dict.items()}

        use_genes = reduce(and_,
                           [expr.columns for expr in expr_dict.values()])
        expr = pd.concat([data_list[0] for data_list in data_dict.values()])

        self.gene_annot = {gn: ant
                           for gn, ant in gene_annot[cohorts[0]].items()
                           if gn in use_genes}

        if (type_file is not None and 'use_types' in coh_args
                and coh_args['use_types'] is not None):
            type_data = pd.read_csv(type_file,
                                    sep='\t', index_col=0, comment='#')

            use_samps = set()
            for cohort in cohorts:
                if coh_args['use_types'][cohort] is not None:
                    use_samps |= set(
                        type_data.index[(type_data.DISEASE == cohort)
                                        & (type_data.SUBTYPE.isin(
                                            coh_args['use_types'][cohort]))]
                        )

                else:
                    use_samps |= set(data_dict[cohort][0].index)

            use_samps &= set(expr.index)

        else:
            use_samps = expr.index

        expr = expr.loc[use_samps]
        self.cohort_samps = {coh: set(data_dict[coh][0].index) & use_samps
                             for coh in cohorts}

        variants = pd.concat([data_list[1]
                              for data_list in data_dict.values()])
        copy_df = pd.concat([data_list[2]
                             for data_list in data_dict.values()])

        if mut_genes:
            self.alleles = variants.loc[
                variants.Gene.isin(mut_genes),
                ['Sample', 'Protein', 'ref_count', 'alt_count']
                ]

        # add a mutation level indicating if a mutation is a CNA or not by
        # first figuring out where to situate it relative to the other
        # levels...
        if 'Gene' in mut_levels:
            scale_lvl = mut_levels.index('Gene') + 1
        else:
            scale_lvl = 0

        # ...and then inserting the new level, and adding its corresponding
        # values to the mutation and copy number alteration datasets
        mut_levels.insert(scale_lvl, 'Scale')
        mut_levels.insert(scale_lvl + 1, 'Copy')
        variants['Scale'] = 'Point'
        copy_df['Scale'] = 'Copy'

        super().__init__(expr, pd.concat([variants, copy_df], sort=True),
                         mut_genes, mut_levels, domain_dir,
                         top_genes, samp_cutoff, cv_prop, cv_seed)

