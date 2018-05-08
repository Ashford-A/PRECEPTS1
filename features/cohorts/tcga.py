
from .mut import *
from .mut_freq import *

from ..data.expression import get_expr_firehose, get_expr_bmeg, get_expr_toil
from ..data.variants import get_variants_mc3, get_variants_firehose
from ..data.copies import get_copies_firehose

from dryadic.features.cohorts import BaseMutationCohort
from dryadic.features.cohorts.utils import (
    match_tcga_samples, get_gencode, log_norm)

from functools import reduce
from operator import and_
from itertools import cycle

import os
from glob import glob
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


def get_variant_data(cohort, var_source, **var_args):
    """Loads variant calls for a given TCGA cohort and data source.

    Args:
        cohort (str): A cohort in TCGA with variant call data.
        var_source (str): A repository which contains TCGA datasets.

    """

    if var_source == 'mc3':
        variants = get_variants_mc3(var_args['syn'])
    
    elif var_source == 'Firehose':
        variants = get_variants_firehose(cohort, var_args['var_dir'])
    
    else:
        raise ValueError("Unrecognized source of variant data!")

    return variants


def get_copy_data(cohort, copy_source, **copy_args):
    if copy_source == 'Firehose':
        copy_data = get_copies_firehose(cohort, copy_args['copy_dir'])
        
        # reshapes the matrix of CNA values into the same long format
        # mutation data is represented in
        copy_df = pd.DataFrame(copy_data.stack())
        copy_df = copy_df.reset_index(level=copy_df.index.names)
        copy_df.columns = ['Sample', 'Gene', 'Form']
        
        # removes CNA values corresponding to an absence of a variant
        copy_df = copy_df.loc[copy_df['Form'] != 0, :]
        
        # maps CNA integer values to their descriptions, appends
        # CNA data to the mutation data
        copy_df['Form'] = copy_df['Form'].map({-2: 'HomDel', -1: 'HetDel',
                                               1: 'HetGain', 2: 'HomGain'})
    
    else:
        raise ValueError("Unrecognized source of CNA data!")

    return copy_df


def list_cohorts(data_source, **data_args):
    """Finds all the TCGA cohorts available in a given data repository.

    Args:
        data_source (str): A repository which contains TCGA datasets.

    """

    if data_source == 'BMEG':
        pass

    elif data_source == 'Firehose':
        cohorts = {pth.split('/')[-1]
                   for pth in glob(os.path.join(data_args['data_dir'],
                                                'stddata__*/*'))}

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
        var_source (str): Where to load the variant data from.
        copy_source (:obj:`str`, optional)
            Where to load the copy number alteration data from. The default
            is to not use any CNA data.
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
        >>>     cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )
        >>>
        >>> cdata2 = MutationCohort(
        >>>     cohort='TCGA-PAAD', mut_genes=['KRAS'],
        >>>     mut_levels=['Form_base', 'Exon', 'Location'],
        >>>     expr_source='Firehose', data_dir='../input-data/firehose',
        >>>     cv_seed=98, cv_prop=0.8, syn=syn
        >>>     )
        >>>
        >>> # load expression data with variant calls for the fifty most
        >>> # frequently mutated genes in the TCGA ovarian cohort
        >>> cdata3 = MutationCohort(cohort='TCGA-OV',
        >>>                         mut_genes=None, top_genes=50)
        >>>
        >>> # load expression data with variant calls for genes mutated in at
        >>> # least forty of the samples in the TCGA colon cohort
        >>> cdata3 = MutationCohort(cohort='TCGA-COAD',
        >>>                         mut_genes=None, samp_cutoff=40)

    """

    def __init__(self,
                 cohort, mut_genes, mut_levels, expr_source, var_source,
                 copy_source, annot_file, top_genes=100, samp_cutoff=None,
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        self.cohort = cohort

        if var_source is None:
            var_source = expr_source

        expr = get_expr_data(cohort, expr_source, **coh_args)
        variants = get_variant_data(cohort, var_source, **coh_args)

        # gets annotation data for each gene in the expression data
        annot_data = get_gencode(annot_file)
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in annot_data.items()
                           if at['gene_name'] in expr.columns}

        if copy_source == 'Firehose':
            if 'copy_dir' not in coh_args and expr_source == 'Firehose':
                copy_dir = coh_args['expr_dir']
            else:
                copy_dir = coh_args['copy_dir']

            copy_data = get_copies_firehose(cohort, copy_dir, discrete=True)

        else:
            raise ValueError("Unrecognized source of copy number data!")

        copy_df = pd.DataFrame(copy_data.stack())
        copy_df = copy_df.reset_index(level=copy_df.index.names)
        copy_df.columns = ['Sample', 'Gene', 'Copy']
        matched_samps = match_tcga_samples(
            expr.index, variants['Sample'], copy_df['Sample'])

        expr = expr.loc[expr.index.isin(matched_samps[0]),
                        expr.columns.isin(self.gene_annot)]
        expr.index = [matched_samps[0][old_samp] for old_samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(matched_samps[1])
                                & variants['Gene'].isin(self.gene_annot)
                                , :]
        variants['Sample'] = [matched_samps[1][old_samp]
                              for old_samp in variants['Sample']]

        copy_df = copy_df.loc[(copy_df['Copy'] != 0)
                              & copy_df['Sample'].isin(matched_samps[2])
                              & copy_df['Gene'].isin(self.gene_annot)
                              , :]

        copy_df['Sample'] = [matched_samps[2][old_samp]
                             for old_samp in copy_df['Sample']]
        copy_df['Copy'] = copy_df['Copy'].map({-2: 'HomDel', -1: 'HetDel',
                                               1: 'HetGain', 2: 'HomGain'})

        if 'Gene' in mut_levels:
            scale_lvl = mut_levels.index('Gene') + 1
        else:
            scale_lvl = 0

        mut_levels.insert(scale_lvl, 'Scale')
        mut_levels.insert(scale_lvl + 1, 'Copy')
        variants['Scale'] = 'Point'
        copy_df['Scale'] = 'Copy'
        variants = pd.concat([variants, copy_df], sort=True)

        super().__init__(expr, variants, mut_genes, mut_levels,
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
                 cohort, expr_source='BMEG', var_source='mc3',
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        self.cohort = cohort

        if var_source is None:
            var_source = expr_source

        expr = get_expr_data(cohort, expr_source, **coh_args)
        variants = get_variant_data(cohort, var_source, **coh_args)
        matched_samps = match_tcga_samples(expr.index, variants['Sample'])

        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        super().__init__(expr, variants, matched_samps, cv_prop, cv_seed)


class PanCancerMutCohort(BaseMutationCohort):

    def __init__(self, 
                 mut_genes, mut_levels=('Gene', 'Form'),
                 expr_source='BMEG', var_source='mc3', copy_source=None,
                 top_genes=100, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):
        expr_args = dict()
        if 'expr_dir' in coh_args:
            expr_args['data_dir'] = coh_args['expr_dir']
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

        expr = pd.concat(list(expr_dict.values()))

        if var_source == 'mc3':
            variants = get_variant_data(cohort=None, var_source='mc3',
                                        **coh_args)

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
                    var_dict[cohort] = get_variant_data(cohort, expr_source,
                                                        **coh_args)

                    if copy_source is not None:
                        var_dict[cohort] = pd.concat([
                            var_dict[cohort],
                            get_copy_data(cohort, copy_source, **coh_args)
                            ])

                except:
                    print('no variants found for {}'.format(cohort))
            
            variants = pd.concat(list(var_dict.values()))

        matched_samps = match_tcga_samples(expr.index, variants['Sample'])

        # gets annotation data for each gene in the expression data, saves the
        # label of the cohort used as an attribute
        gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                      for ens, at in get_gencode().items()
                      if at['gene_name'] in expr.columns}
        self.cohort = cohort

        super().__init__(expr, variants, matched_samps, gene_annot, mut_genes,
                         mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed)


class TransferMutationCohort(BaseTransferMutationCohort):

    def __init__(self,
                 cohorts, mut_genes, mut_levels=('Gene', 'Form'),
                 expr_sources='BMEG', var_sources='mc3', copy_sources=None,
                 top_genes=250, samp_cutoff=None, cv_prop=2.0/3, cv_seed=None,
                 **coh_args):

        if isinstance(expr_sources, str):
            expr_sources = [expr_sources]

        if var_sources is None:
            var_sources = expr_sources

        if isinstance(var_sources, str):
            var_sources = [var_sources]

        if 'mc3' in var_sources:
            mc3_vars = get_variants_mc3(coh_args['syn'])
        else:
            mc3_vars = None

        if isinstance(copy_sources, str):
            copy_sources = [copy_sources]

        expr_dict = dict()
        for coh, expr_source in zip(cohorts, cycle(expr_sources)):
            if expr_source == 'BMEG':
                expr_mat = get_expr_bmeg(coh)
                expr_dict[coh] = log_norm(expr_mat.fillna(0.0))

            elif expr_source == 'Firehose':
                expr_mat = get_expr_firehose(coh, coh_args['expr_dir'])
                expr_dict[coh] = log_norm(expr_mat.fillna(0.0))

            elif expr_source == 'toil':
                expr_dict[coh] = get_expr_toil(coh, coh_args['expr_dir'],
                                               coh_args['collapse_txs'])

            else:
                raise ValueError("Unrecognized source of expression data!")

        var_dict = dict()
        for coh, var_source in zip(cohorts, cycle(var_sources)):

            if var_source == 'mc3':
                var_dict[coh] = mc3_vars.copy()

            elif var_source == 'Firehose':
                var_dict[coh] = get_variants_firehose(
                    coh, coh_args['var_dir'])

            else:
                raise ValueError("Unrecognized source of variant data!")

        matched_samps = {coh: match_tcga_samples(expr_dict[coh].index,
                                                 var_dict[coh]['Sample'])
                         for coh in cohorts}

        if copy_sources is not None:
            for coh, copy_source in zip(cohorts, cycle(copy_sources)):
                if copy_source == 'Firehose':
                    copy_data = get_copies_firehose(coh, coh_args['copy_dir'])

                    # reshapes the matrix of CNA values into the same long
                    # format mutation data is represented in
                    copy_df = pd.DataFrame(copy_data.stack())
                    copy_df = copy_df.reset_index(level=copy_df.index.names)
                    copy_df.columns = ['Sample', 'Gene', 'Form']
                    copy_df = copy_df.loc[copy_df['Form'] != 0, :]

                    # maps CNA integer values to their descriptions, appends
                    # CNA data to the mutation data
                    copy_df['Form'] = copy_df['Form'].map(
                        {-2: 'HomDel', -1: 'HetDel', 1: 'HetGain',
                         2: 'HomGain'})
                    var_dict[coh] = pd.concat([var_dict[coh], copy_df])

        annot_data = get_gencode()
        gene_annot = {coh: {at['gene_name']: {**{'Ens': ens}, **at}
                            for ens, at in annot_data.items()
                            if at['gene_name'] in expr_dict[coh].columns}
                      for coh in cohorts}
        self.cohorts = cohorts

        super().__init__(
            expr_dict, var_dict, matched_samps, gene_annot, mut_genes,
            mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed
            )

