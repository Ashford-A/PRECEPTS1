
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
                         cv_seed=None, test_prop=0.):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    return MutationCohort(cohort=cohort.split('_')[0], mut_levels=mut_levels,
                          mut_genes=list(genes), domain_dir=domain_dir,
                          expr_source='Firehose', var_source='mc3',
                          copy_source='Firehose', annot_file=annot_file,
                          type_file=type_file, expr_dir=expr_dir,
                          copy_dir=copy_dir, syn=syn, cv_seed=cv_seed,
                          test_prop=test_prop, annot_fields=['transcript'],
                          use_types=parse_subtypes(cohort))

