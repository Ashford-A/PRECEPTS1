
from .data_dirs import expr_dir, copy_dir, syn_root, annot_file
from dryadic.features.mutations import MuType

copy_mtypes = {MuType({('Copy', 'HomDel'): None}),
               MuType({('Copy', ('HomDel', 'HetDel')): None}),
               MuType({('Copy', 'HomGain'): None}),
               MuType({('Copy', ('HomGain', 'HetGain')): None})}

__all__ = ['copy_mtypes', 'expr_dir', 'copy_dir', 'syn_root', 'annot_file']

