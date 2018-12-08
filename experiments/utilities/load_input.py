
from .data_dirs import (
    expr_dir, copy_dir, syn_root, domain_dir, annot_file, type_file)
from ...features.cohorts.tcga import MutationCohort
import synapseclient

cohort_subtypes = {
    'nonbasal': ['LumA', 'LumB', 'Her2', 'Normal'],
    'luminal': ['LumA', 'LumB'],
    }


def parse_subtypes(cohort):
    use_subtypes = None
    coh_info = cohort.split('_')

    if len(coh_info) > 1:
        if coh_info[1] in cohort_subtypes:
            use_subtypes = cohort_subtypes[coh_info[1]]
        else:
            use_subtypes = [coh_info[1]]

    return use_subtypes


def load_firehose_cohort(cohort, genes, mut_levels=None,
                         cv_prop=1.0, cv_seed=None):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    return MutationCohort(cohort=cohort.split('_')[0], mut_genes=list(genes),
                          mut_levels=mut_levels, domain_dir=domain_dir,
                          expr_source='Firehose', var_source='mc3',
                          copy_source='Firehose', annot_file=annot_file,
                          type_file=type_file, expr_dir=expr_dir,
                          copy_dir=copy_dir, cv_prop=cv_prop, syn=syn,
                          cv_seed=cv_seed, use_types=parse_subtypes(cohort))

