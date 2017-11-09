#!/bin/bash

#SBATCH --job-name=subv-detect_plots
#SBATCH --partition=exacloud
#SBATCH --mem=16000
#SBATCH --time=150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-det-plots_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-det-plots_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate precepts

if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

if [ -z ${classif+x} ]
then
	echo "no classifier defined, defaulting to Lasso"
	export classif="Lasso"
fi

python HetMan/experiments/subvariant_detection/plot_search.py $cohort $classif
python HetMan/experiments/subvariant_detection/plot_portray.py $cohort $classif

