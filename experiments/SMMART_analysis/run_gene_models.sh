#!/bin/bash

#SBATCH --job-name=SMMART-models_gene
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2150


# disable threading on each CPU, pause to ease I/O burden, load required
# scripts and packages
export OMP_NUM_THREADS=1
sleep $(($SLURM_ARRAY_TASK_ID * 7));
source activate HetMan

while getopts t:g:c: var
do
	case "$var" in
		t)	export cohort=$OPTARG;;
		g)	export gene=$OPTARG;;
		c)	export classif=$OPTARG;;
		[?])	echo "Usage: $0 [-t] a TCGA cohort" \
			     "[g] a mutated gene [-c] a mutation classifier";
			exit 1;;
	esac
done

srun -p=exacloud \
	python HetMan/experiments/SMMART_analysis/fit_gene_models.py \
	$cohort $gene $classif $toil_dir $syn_root $patient_dir \
	--tune_splits=8 --test_count=30 --infer_splits=12 --parallel_jobs=12 \
	--cv_id=$SLURM_ARRAY_TASK_ID

