
import os
base_dir = os.path.join(os.environ['DATADIR'],
                        'dryads-research', 'subgrouping_test')

# "canonical" cohorts selected using sample size criteria and homogeneity
# with respect to molecular subtypes as described in Grzadkowski et al. (2021)
# all of these use the "Firehose" `expr_source` except for METABRIC, which uses
# "microarray", and beatAML, which uses "toil__gns"
train_cohorts = {
    'BLCA', 'BRCA_LumA', 'CESC_SquamousCarcinoma', 'HNSC_HPV-', 'KIRC',
    'LGG_IDHmut-non-codel', 'LIHC', 'LUAD', 'LUSC', 'METABRIC_LumA', 'OV',
    'PRAD', 'SKCM', 'STAD', 'THCA', 'beatAML'
    }

