
from .data_dirs import (expr_sources, copy_dir, syn_root, domain_dir,
                        annot_file, gene_list, type_file, beatAML_files)
from dryadic.features.mutations import MuType


cis_lbls = ('None', 'Self', 'Chrm')
pnt_mtype = MuType({('Scale', 'Point'): None})


# TODO: remove stuff from __all__ that isn't directly
# related to input datasets?
__all__ = ['expr_sources', 'copy_dir', 'syn_root', 'domain_dir', 'annot_file',
           'gene_list', 'type_file', 'beatAML_files', 'pnt_mtype', 'cis_lbls',
           'pnt_mtype']

