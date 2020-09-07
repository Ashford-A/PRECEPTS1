
from ...experiments.utilities.data_dirs import (
    firehose_dir, syn_root, metabric_dir, baml_dir, gencode_dir,
    subtype_file, expr_sources
    )

import synapseclient
from ...features.cohorts.beatAML import (
    process_input_datasets as process_baml_datasets)
from ...features.cohorts.metabric import (
    process_input_datasets as process_metabric_datasets)
from ...features.cohorts.tcga import (
    process_input_datasets as process_tcga_datasets)


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

