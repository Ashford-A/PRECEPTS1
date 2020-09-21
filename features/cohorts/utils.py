
from ...experiments.utilities.data_dirs import (
    firehose_dir, syn_root, metabric_dir, baml_dir, gencode_dir,
    subtype_file, expr_sources
    )

from ...features.cohorts.beatAML import (
    process_input_datasets as process_baml_datasets)
from ...features.cohorts.metabric import (
    process_input_datasets as process_metabric_datasets)
from ...features.cohorts.tcga import (
    process_input_datasets as process_tcga_datasets)

from dryadic.features.data.vep import process_variants
from dryadic.features.cohorts.mut import BaseMutationCohort

import os
import synapseclient
import pandas as pd
import dill as pickle


def get_input_datasets(cohort, expr_source, mut_fields=None):
    data_dict = {data_k: None
                 for data_k in ('expr', 'vars', 'copy', 'annot', 'assembly')}

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    if cohort == 'beatAML':
        if expr_source != 'toil__gns':
            raise ValueError("Only gene-level Kallisto calls are available "
                             "for the beatAML cohort!")

        data_dict['assembly'] = 'GRCh37'

        data_dict.update({
            data_k: baml_data
            for data_k, baml_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_baml_datasets(baml_dir, gencode_dir, syn,
                                      annot_fields=['transcript'],
                                      mut_fields=mut_fields)
                )
            })

    elif cohort.split('_')[0] == 'METABRIC':
        if expr_source != 'microarray':
            raise ValueError("Only Illumina microarray mRNA calls are "
                             "available for the METABRIC cohort!")

        data_dict['assembly'] = 'GRCh37'

        if '_' in cohort:
            use_types = cohort.split('_')[1]
        else:
            use_types = None

        data_dict.update({
            data_k: mtbc_data
            for data_k, mtbc_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_metabric_datasets(
                    metabric_dir, gencode_dir, use_types,
                    annot_fields=['transcript'], mut_fields=mut_fields
                    )
                )
            })

    elif cohort.split('_')[0] == 'CCLE':
        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        if source_base == 'microarray':
            expr_dir = ccle_dir
        else:
            expr_dir = expr_sources[source_base]

        cdata = CellLineCohort(
            mut_levels=mut_lvls, mut_genes=use_genes, expr_source=source_base,
            ccle_dir=ccle_dir, annot_file=annot_file, domain_dir=domain_dir,
            expr_dir=expr_dir, collapse_txs=collapse_txs,
            cv_seed=8713, test_prop=0
            )

    else:
        data_dict['assembly'] = 'GRCh37'

        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        data_dict.update({
            data_k: tcga_data
            for data_k, tcga_data in zip(
                ('expr', 'vars', 'copy', 'annot'),
                process_tcga_datasets(
                    cohort, expr_source=source_base,
                    var_source='mc3', copy_source='Firehose',
                    expr_dir=expr_sources[source_base], annot_dir=gencode_dir,
                    type_file=subtype_file, collapse_txs=collapse_txs,
                    annot_fields=['transcript'], syn=syn,
                    mut_fields=mut_fields
                    )
                )
            })

    return data_dict


def get_cohort_data(cohort, expr_source, mut_lvls, vep_cache_dir, out_path,
                    use_genes=None):
    data_dict = get_input_datasets(
        cohort, expr_source,
        mut_fields=['Sample', 'Gene', 'Chr', 'Start', 'End',
                    'RefAllele', 'TumorAllele']
        )

    var_df = pd.DataFrame({'Chr': data_dict['vars'].Chr.astype('int'),
                           'Start': data_dict['vars'].Start.astype('int'),
                           'End': data_dict['vars'].End.astype('int'),
                           'RefAllele': data_dict['vars'].RefAllele,
                           'VarAllele': data_dict['vars'].TumorAllele,
                           'Sample': data_dict['vars'].Sample})

    if isinstance(mut_lvls[0], str):
        var_fields = ['Gene', 'Canonical', 'Location', 'VarAllele']
        cdata_lvls = [mut_lvls]

        for lvl in mut_lvls[3:]:
            if '-domain' in lvl and 'Domains' not in var_fields:
                var_fields += ['Domains']
            else:
                var_fields += [lvl]

    elif isinstance(mut_lvls[0], tuple):
        var_fields = {'Gene', 'Canonical', 'Location', 'VarAllele'}
        cdata_lvls = list(mut_lvls)

        for lvl_list in mut_lvls:
            for lvl in lvl_list[2:]:
                if '-domain' in lvl and 'Domains' not in var_fields:
                    var_fields |= {'Domains'}
                else:
                    var_fields |= {lvl}

    else:
        raise TypeError(
            "Unrecognized <mut_lvls> argument: `{}`!".format(mut_lvls))

    # run the VEP command line wrapper to obtain a standardized
    # set of point mutation calls
    variants = process_variants(
        var_df, out_fields=var_fields, cache_dir=vep_cache_dir,
        temp_dir=out_path, assembly=data_dict['assembly'],
        distance=0, consequence_choose='pick', forks=4, update_cache=False
        )

    # remove mutation calls not assigned to a canonical transcript by VEP as
    # well as those not associated with genes linked to cancer processes
    variants = variants.loc[variants.CANONICAL == 'YES']
    if use_genes:
        variants = variants.loc[variants.Gene.isin(use_genes)]
        copies = data_dict['copy'].loc[data_dict['copy'].Gene.isin(use_genes)]
    else:
        copies = data_dict['copy']

    assert not variants.duplicated().any(), (
        "Variant data contains {} duplicate entries!".format(
            variants.duplicated().sum())
        )

    cdata = BaseMutationCohort(data_dict['expr'], variants, cdata_lvls,
                               copies, data_dict['annot'], leaf_annot=None)

    return cdata


def load_cohort(cohort, expr_source, mut_lvls,
                vep_cache_dir, use_path=None, temp_path=None, use_genes=None):
    if use_path is not None and os.path.exists(use_path):
        try:
            with open(use_path, 'rb') as f:
                cdata = pickle.load(f)

        except:
            cdata = get_cohort_data(cohort, expr_source, mut_lvls,
                                    vep_cache_dir, temp_path, use_genes)

    else:
        cdata = get_cohort_data(cohort, expr_source, mut_lvls,
                                vep_cache_dir, temp_path, use_genes)

    if mut_lvls not in cdata.mtrees:
        cdata.merge(get_cohort_data(cohort, expr_source, mut_lvls,
                                    vep_cache_dir, temp_path, use_genes))

    return cdata

