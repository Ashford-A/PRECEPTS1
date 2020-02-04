
from .data_dirs import (expr_sources, expr_dir, copy_dir, syn_root,
                        domain_dir, annot_file, gene_list, type_file,
                        beatAML_files, metabric_dir, ccle_dir)
from dryadic.features.mutations import MuType


pnt_mtype = MuType({('Scale', 'Point'): None})
copy_mtype = MuType({('Scale', 'Copy'): None})
gain_mtype = MuType({('Scale', 'Copy'): {('Copy', 'DeepGain'): None}})
loss_mtype = MuType({('Scale', 'Copy'): {('Copy', 'DeepDel'): None}})

train_cohorts = {'BLCA', 'BRCA_LumA', 'CESC_SquamousCarcinoma', 'HNSC_HPV-',
                 'KIRC', 'LGG_IDHmut-non-codel', 'LIHC', 'LUAD', 'LUSC',
                 'METABRIC_LumA', 'OV', 'PRAD', 'SKCM', 'STAD', 'THCA'}

__all__ = ['expr_sources', 'expr_dir', 'copy_dir', 'syn_root',
           'domain_dir', 'annot_file', 'gene_list', 'type_file',
           'beatAML_files', 'metabric_dir', 'ccle_dir']

