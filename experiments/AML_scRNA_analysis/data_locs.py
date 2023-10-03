#expr_dir = "/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_ALL_merged"
#expr_dir = "/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_healthy_BM_samples"
expr_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_D0_AML_samples_and_4_healthy_BM_samples'
#expr_dir = '/home/groups/precepts/ashforda/beatAML_files_for_precepts/waves_1_to_4/RNAseq'

firehose_dir = "/home/groups/precepts/grzadkow/input-data/firehose"
syn_root = "/home/groups/precepts/precepts1-lite/dependencies/local_Synapse_cache"

metabric_dir = "/home/groups/precepts/input-data/brca_metabric"
#baml_dir = "/home/groups/precepts/ashforda/beatAML_files_for_precepts/CTD2"
baml_dir = '/home/groups/precepts/ashforda/beatAML_files_for_precepts/waves_1_to_4'
vep_cache_dir = "/home/groups/precepts/precepts1-lite/dependencies/VEP_cache"

ccle_dir = "/home/groups/precepts/input-data/cellline_ccle_broad"
#gencode_dir = "/home/groups/precepts/precepts1-lite/dependencies/biological_features_annotation_files"
gencode_dir = '/home/groups/precepts/input-data/GENCODE'
oncogene_list = "/home/groups/precepts/input-data/OncoKB_cancerGeneList.txt"
subtype_file = "/home/groups/precepts/input-data/tcga_subtypes.txt"
domain_dir = "/home/users/grzadkow/input-data/ensembl/GRCh37"

expr_sources = {
    'Firehose': "/home/groups/precepts/grzadkow/input-data/firehose",
    'toil': "/home/groups/precepts/input-data/tatlow-tcga",
    }

import os
iorio_dir = "/home/groups/precepts/input-data/iorio-landscape"
resp_files = {'AUC': os.path.join(iorio_dir, "TableS4B.xlsx")}
drug_assocs = ("/home/groups/precepts/input-data"
               "/oncokb_biomarker_drug_associations.tsv")

