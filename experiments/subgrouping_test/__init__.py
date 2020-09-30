
import os
base_dir = os.path.join(os.environ['DATADIR'],
                        'dryads-research', 'subgrouping_test')

train_cohorts = {'BLCA', 'BRCA_LumA', 'CESC_SquamousCarcinoma', 'HNSC_HPV-',
                 'KIRC', 'LGG_IDHmut-non-codel', 'LIHC', 'LUAD', 'LUSC',
                 'METABRIC_LumA', 'OV', 'PRAD', 'SKCM', 'STAD', 'THCA'}

