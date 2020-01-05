
from .data_dirs import (expr_sources, copy_dir, syn_root, domain_dir,
                        annot_file, gene_list, type_file, beatAML_files,
                        metabric_dir, ccle_dir)

from dryadic.features.mutations import MuType
from HetMan.experiments.subvariant_test import pnt_mtype


cis_lbls = ('None', 'Self', 'Chrm')
train_cohorts = {'BLCA', 'BRCA_nonbasal', 'CESC_SquamousCarcinoma',
                 'HNSC_HPV-', 'KIRC', 'LGG_IDHmut-non-codel', 'LIHC', 'LUAD',
                 'LUSC', 'OV', 'PRAD', 'SKCM', 'STAD', 'THCA'}

__all__ = ['expr_sources', 'copy_dir', 'syn_root', 'domain_dir',
           'annot_file', 'gene_list', 'type_file', 'beatAML_files',
           'metabric_dir', 'ccle_dir']

