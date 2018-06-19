#!/bin/bash

#SBATCH --job-name=gene-baseline_fit
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5000


export OMP_NUM_THREADS=1
echo $OUTDIR
sleep $(($SLURM_ARRAY_TASK_ID * 7));

cv_id=$(($SLURM_ARRAY_TASK_ID % 25));
task_id=$(($SLURM_ARRAY_TASK_ID / 25));

srun --output=$OUTDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${cv_id}_${task_id}.err \
	python $BASEDIR/fit_tests.py -v \
	$expr_source $cohort $syn_root $samp_cutoff $classif --cv_id=$cv_id \
	--task_count=$(( $array_size / 25 + 1 )) --task_id=$task_id \

