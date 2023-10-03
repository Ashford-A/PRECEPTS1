#!/bin/bash
#SBATCH --job-name=AML-sc_plots
#SBATCH --verbose


source activate research

python -m dryads-research.experiments.AML_scRNA_analysis.plot_cluster Ridge\
	--feats_file /home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_ALL_merged/Temp_Files/dryads-research/AML_scRNA_analysis/default__default/Ridge/setup/feat-list.p \
	--comp_files /home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_ALL_merged/umap_embedding.tsv

