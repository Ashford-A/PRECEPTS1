#!/bin/bash
#SBATCH --verbose

# this script is designed to produce the figures included in the manuscript
#	"Systematic interrogation of mutation groupings reveals divergent
#	downstream expression programs within key cancer genes"
#	(MR Grzadkowski, HD Holly, J Somers, and E Demir)
# currently under review at BMC Bioinformatics


# activate conda environment, locate experiment output root
source activate research
OUTDIR=$DATADIR/dryads-research/subgrouping_test

# find all of the completed experiments
declare -A out_list

for out_file in $( find $OUTDIR/*__*__samps-*/out-conf__*__*.p.gz)
do
	file_nm=${out_file##*subgrouping_test/}
	file_nm=${file_nm%%.p.gz}

	src=${file_nm%%__samps-*}
	coh=${src##*__}
	src=${src%__*}
	clf=${file_nm##*__}

	lvls=${file_nm##*out-conf__}
	lvls=${lvls%__*}
	out_list[${src},${coh},${clf}]=${out_list[${src},${coh},${clf}]},${lvls}
done

# produce plots for cohorts with all four mutation annotation hierarchies tested
for K in "${!out_list[@]}"
do
	if [ $( awk -F"," '{print NF-1}' <<< "${out_list[$K]}" ) -eq 4 ]
	then
		src=${K%%,*}
		coh=${K%,*}
		coh=${coh#*,*}
		clf=${K##*,}

		# Figure 1, Figures S6A-B, S8, S10A-E, S11
		python -m dryads-research.experiments.subgrouping_test.plot_aucs \
			$src $coh $clf --legends

		# Figures S2, S3
		python -m dryads-research.experiments.subgrouping_test.plot_models \
			$src $coh $clf

		# Figures S4, S5
		python -m dryads-research.experiments.subgrouping_test.plot_random \
			$src $coh $clf

		# Figures S16
		python -m dryads-research.experiments.subgrouping_test.plot_accuracy \
			$src $coh $clf
	fi
done

# Tables S1
python -m dryads-research.experiments.subgrouping_test.generate_summaries Ridge

# Figure 3, Figure S6C
python -m dryads-research.experiments.subgrouping_test.plot_classif Ridge

# Figure S7A
python -m dryads-research.experiments.subgrouping_tour.plot_aucs \
	microarray METABRIC_LumA default default Ridge

# Figure S7B
python -m dryads-research.experiments.subgrouping_tour.plot_search \
	microarray METABRIC_LumA default default Ridge

# Figure 2, Figure S9
python -m dryads-research.experiments.subgrouping_test.plot_cohorts \
	METABRIC_LumA BRCA_LumA Ridge

# Figures S12, S13, S14
python -m dryads-research.experiments.subgrouping_test.plot_genes Ridge

# Figures S15
python -m dryads-research.experiments.subgrouping_threshold.plot_sig \
	BRCA_LumA Ridge

# Figure 4, Figures S17A-C
python -m dryads-research.experiments.subgrouping_test.plot_projection \
	microarray METABRIC_LumA GATA3 Ridge -t Consequence

# Figure 5, Figures S18
python -m dryads-research.experiments.subgrouping_test.plot_drug \
	Firehose LUSC NFE2L2 Ridge

