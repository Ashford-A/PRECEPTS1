
from dryadic.features.mutations import MuType
from HetMan.experiments.subvariant_transfer.merge_transfer import (
    merge_cohort_data)
from .data_dirs import (expr_sources, copy_dir, syn_root, domain_dir,
                        annot_file, gene_list, type_file)

ex_mtypes = {'Nothing': MuType([]),
             'Shallow': MuType({('Scale', 'Copy'): {(
                 'Copy', ('ShalGain', 'ShalDel')): None}}),
             'ShalGain': MuType({('Scale', 'Copy'): {(
                 'Copy', 'ShalGain'): None}}),
             'ShalDel': MuType({('Scale', 'Copy'): {(
                 'Copy', 'ShalDel'): None}})}

__all__ = ['ex_mtypes', 'merge_cohort_data', 'expr_sources', 'copy_dir',
           'syn_root', 'domain_dir', 'annot_file', 'gene_list', 'type_file']

