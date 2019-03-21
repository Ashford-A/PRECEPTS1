
from ..data.expression import get_expr_firehose, get_expr_bmeg, get_expr_toil
from ..data.variants import get_variants_mc3, get_variants_firehose
from ..data.copies import get_copies_firehose
from .mut_freq import BaseMutFreqCohort

from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.mut import BaseTransferMutationCohort
from dryadic.features.cohorts.utils import (
    match_tcga_samples, get_gencode, log_norm)

import pandas as pd
from functools import reduce
from operator import and_
from itertools import cycle, combinations

from pathlib import Path
from sklearn.preprocessing import scale


def get_expr_data(cohort, expr_source, **expr_args):
    """Loads RNA-seq expression data for a given TCGA cohort and data source.

    Args:
        cohort (str): A cohort in TCGA with expression data.
        expr_source (str): A repository which contains TCGA datasets.

    """

    if expr_source == 'BMEG':
        expr_mat = get_expr_bmeg(cohort)
        expr = log_norm(expr_mat.fillna(0.0))

    elif expr_source == 'Firehose':
        expr_mat = get_expr_firehose(cohort, expr_args['expr_dir'])
        expr = log_norm(expr_mat.fillna(0.0))

    elif expr_source == 'toil':
        expr = get_expr_toil(cohort, expr_args['expr_dir'],
                             expr_args['collapse_txs'])

    else:
        raise ValueError("Unrecognized source of expression data!")

    return expr


def add_variant_data(cohort, var_source, copy_source, expr, gene_annot,
                     **var_args):
    """Adds variant calls to loaded TCGA expression data.

    Args:
        cohort (str): A cohort in TCGA with variant call data.
        var_source (str): A repository which contains TCGA datasets.
        expr (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])
        gene_annot (dict): Annotation for each of the expressed genes.
        add_cna (bool, optional): Whether to add discrete copy number calls.
                                  Default is to only use point mutations.

    """

    # load mutation data from the given source
    if 'mc3' in var_args:
        variants = var_args['mc3']

    elif var_source == 'mc3':
        variants = get_variants_mc3(var_args['syn'])
    
    elif var_source == 'Firehose':
        variants = get_variants_firehose(cohort, var_args['var_dir'])
    
    else:
        raise ValueError("Unrecognized source of variant data!")

    # load copy number alteration data from the given source
    if copy_source == 'Firehose':
        if 'copy_dir' not in var_args:
            copy_dir = var_args['expr_dir']
        else:
            copy_dir = var_args['copy_dir']

        copy_data = get_copies_firehose(cohort, copy_dir, discrete=True)

    else:
        raise ValueError("Unrecognized source of copy number data!")

    copy_df = pd.DataFrame(copy_data.stack())
    copy_df = copy_df.reset_index(level=copy_df.index.names)
    copy_df.columns = ['Sample', 'Gene', 'Copy']

    expr_match, var_match, copy_match = match_tcga_samples(
        expr.index, variants.loc[:, 'Sample'], copy_df.loc[:, 'Sample'])

    new_expr = expr.loc[
        expr.index.isin(expr_match),
        expr.columns.get_level_values('Gene').isin(gene_annot)
        ]
    new_expr.index = [expr_match[old_samp] for old_samp in new_expr.index]

    new_vars = variants[variants.isin({
        'Sample': var_match.keys(), 'Gene': gene_annot.keys()}).loc[
            :, ['Gene', 'Sample']].all(axis=1)].copy()
    new_vars.Sample = new_vars.Sample.apply(lambda samp: var_match[samp])
 
    new_copy = copy_df.loc[(copy_df.Copy != 0)
                           & copy_df.Sample.isin(copy_match)
                           & copy_df.Gene.isin(gene_annot)
                           , :].copy()

    new_copy.Sample = new_copy.Sample.apply(lambda samp: copy_match[samp])
    new_copy.Copy = new_copy.Copy.map({-2: 'DeepDel', -1: 'ShalDel',
                                       1: 'ShalGain', 2: 'DeepGain'})

    return new_expr, new_vars, new_copy


def list_cohorts(data_source, **data_args):
    """Finds all the TCGA cohorts available in a given data repository.

    Args:
        data_source (str): A repository which contains TCGA datasets.

    """

    if data_source == 'BMEG':
        pass

    elif data_source == 'Firehose':
        cohorts = {pth.parent.parent.name
                   for pth in Path(data_args['expr_dir']).glob(
                       "stddata__*/**/*__RSEM_genes_normalized__*")}

        cohorts &= {pth.parent.parent.name
                    for pth in Path(data_args['copy_dir']).glob(
                        "analyses__*/**/*CopyNumber_Gistic2.Level_4*")}

    elif data_source == 'toil':
        pass

    else:
        raise ValueError("Unrecognized source of expression data!")

    return cohorts


class MutationCohort(BaseMutationCohort):
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
                 cohort, mut_genes, mut_levels,
                 expr_source, var_source, copy_source,
                 annot_file, domain_dir=None, type_file=None,
                 top_genes=100, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):
        self.cohort = cohort

        # load expression and gene annotation datasets
        expr = get_expr_data(cohort, expr_source, **coh_args)
        if 'annot_fields' in coh_args:
            annot_data = get_gencode(annot_file, coh_args['annot_fields'])
        else:
            annot_data = get_gencode(annot_file)

        # restructure annotation data around expression gene labels
        self.gene_annot = {
            at['gene_name']: {**{'Ens': ens}, **at}
            for ens, at in annot_data.items()
            if at['gene_name'] in set(expr.columns.get_level_values('Gene'))
            }

        expr, variants, copy_df = add_variant_data(
            cohort, var_source, copy_source, expr,
            self.gene_annot, **coh_args
            )

        if (type_file is not None and 'use_types' in coh_args
                and coh_args['use_types'] is not None):
            type_data = pd.read_csv(type_file, sep='\t', index_col=0)
            type_data = type_data[type_data.DISEASE == cohort]

            use_samps = set(type_data.index[type_data.SUBTYPE.isin(
                coh_args['use_types'])])
            use_samps &= set(expr.index)

        else:
            use_samps = expr.index

        expr = expr.loc[use_samps]
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


class MutFreqCohort(BaseMutFreqCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant data from.
        copy_source (:obj:`str`, optional)
            Where to load the copy number alteration data from. The default
            is to not use any CNA data.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = MutationCohort(
        >>>     cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )

    """

    def __init__(self,
                 cohort, expr_source, var_source, copy_source, annot_file,
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        self.cohort = cohort

        expr = get_expr_data(cohort, expr_source, **coh_args)
        annot_data = get_gencode(annot_file)

        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        expr, variants = add_variant_data(cohort, var_source, copy_source,
                                          expr, self.gene_annot, **coh_args)

        super().__init__(expr, variants, cv_prop, cv_seed)


class PanCancerMutCohort(BaseMutationCohort):

    def __init__(self, 
                 mut_genes, mut_levels, expr_source, var_source, copy_source,
                 annot_file, top_genes=100, samp_cutoff=None,
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        expr_cohorts = list_cohorts(expr_source, **expr_args)

        expr_dict = dict()
        for cohort in expr_cohorts:
            try:
                expr_data = get_expr_data(cohort, expr_source, **coh_args)

                expr_dict[cohort] = pd.DataFrame(scale(expr_data),
                                                 index=expr_data.index,
                                                 columns=expr_data.columns)

            except:
                print('no expression found for {}'.format(cohort))

        # removes samples that appear in more than one cohort
        for coh1, coh2 in combinations(expr_dict, 2):
            if len(expr_dict[coh1].index & expr_dict[coh2].index):
                ovlp1 = expr_dict[coh1].index.isin(expr_dict[coh2].index)
                ovlp2 = expr_dict[coh2].index.isin(expr_dict[coh1].index)

                if np.all(ovlp1):
                    expr_dict[coh2] = expr_dict[coh2].loc[~ovlp2, :]
                elif np.all(ovlp2) or np.sum(~ovlp1) >= np.sum(~ovlp2):
                    expr_dict[coh1] = expr_dict[coh1].loc[~ovlp1, :]
                else:
                    expr_dict[coh2] = expr_dict[coh2].loc[~ovlp2, :]

        expr = pd.concat(list(expr_dict.values()))
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        if var_source == 'mc3':
            variants, matched_samps = add_variant_data(
                cohort=None, var_source='mc3', expr=expr,
                gene_annot=self.gene_annot, **coh_args
                )

        else:
            if var_source is None:
                var_source = expr_source
                var_cohorts = expr_cohorts

            else:
                var_args = dict()
                if 'var_dir' in coh_args:
                    var_args['data_dir'] = coh_args['var_dir']
                if 'syn' in coh_args:
                    var_args['syn'] = coh_args['syn']

                var_cohorts = list_cohorts(var_source, **var_args)
                var_cohorts &= expr_dict

            var_dict = dict()
            for cohort in var_cohorts:
                try:
                    var_dict[cohort] = add_variant_data(cohort, expr_source,
                                                        **coh_args)

                    if copy_source is not None:
                        var_dict[cohort] = pd.concat([
                            var_dict[cohort],
                            get_copy_data(cohort, copy_source, **coh_args)
                            ])

                except:
                    print('no variants found for {}'.format(cohort))
            
            variants = pd.concat(list(var_dict.values()))

        expr = expr.loc[expr.index.isin(matched_samps[0]),
                        expr.columns.isin(list(self.gene_annot))]
        expr.index = [matched_samps[0][old_samp] for old_samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(matched_samps[1]), :]
        variants['Sample'] = [matched_samps[1][old_samp]
                              for old_samp in variants['Sample']]

        # for each expression cohort, find the samples that were matched to
        # a sample in the mutation call data
        cohort_samps = {
            cohort: set(matched_samps[0][samp]
                        for samp in expr_df.index & matched_samps[0])
            for cohort, expr_df in expr_dict.items()
            }

        # save a list of matched samples for each cohort as an attribute
        self.cohort_samps = {cohort: samps
                             for cohort, samps in cohort_samps.items()
                             if samps}
        copy_data = None
        super().__init__(expr, variants, copy_data, mut_genes, mut_levels,
                         top_genes, samp_cutoff, cv_prop, cv_seed)


class TransferMutationCohort(BaseTransferMutationCohort):

    def __init__(self,
                 cohorts, mut_genes, mut_levels, expr_sources, var_sources,
                 copy_sources, annot_file, top_genes=100, samp_cutoff=None,
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        self.cohorts = cohorts

        if isinstance(expr_sources, str):
            expr_sources = [expr_sources]

        if var_sources is None:
            var_sources = expr_sources

        elif 'mc3' in var_sources:
            coh_args = {**coh_args,
                        **{'mc3': get_variants_mc3(coh_args['syn'])}}

        elif isinstance(var_sources, str):
            var_sources = [var_sources]

        if isinstance(copy_sources, str):
            copy_sources = [copy_sources]

        # load expression data for each cohort, get gene annotation
        expr_raw = {coh: get_expr_data(coh, expr_src, **coh_args)
                    for coh, expr_src in zip(cohorts, cycle(expr_sources))}
        annot_data = get_gencode(annot_file)

        self.gene_annot = {
            at['gene_name']: {**{'Ens': ens}, **at}
            for ens, at in annot_data.items()
            if at['gene_name'] in reduce(
                and_, [expr.columns for expr in expr_raw.values()])
            }

        expr_dict = {cohort: None for cohort in cohorts}
        var_dict = {cohort: None for cohort in cohorts}
        for cohort, var_source, copy_source in zip(
                cohorts, cycle(var_sources), cycle(copy_sources)):

            expr_dict[cohort], var_dict[cohort], copy_df = add_variant_data(
                cohort, var_source, copy_source, expr_raw[cohort],
                self.gene_annot, **coh_args
                )
 
            var_dict[cohort].loc[:, 'Scale'] = 'Point'
            copy_df.loc[:, 'Scale'] = 'Copy'
            var_dict[cohort] = pd.concat([var_dict[cohort], copy_df],
                                         sort=True)
 
        if 'Gene' in mut_levels:
            scale_lvl = mut_levels.index('Gene') + 1
        else:
            scale_lvl = 0
 
        mut_levels.insert(scale_lvl, 'Scale')
        mut_levels.insert(scale_lvl + 1, 'Copy')

        super().__init__(expr_dict, var_dict,
                         mut_genes, mut_levels, top_genes, samp_cutoff,
                         cv_prop, cv_seed)

