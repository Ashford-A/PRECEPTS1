#!/bin/bash

#SBATCH --job-name=cna-isolate_fit
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000


# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $OUTDIR
echo $array_size

# pause between starting array jobs to ease I/O load
sleep $(($SLURM_ARRAY_TASK_ID * 7));

srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.txt \
	--error=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.err \
	python HetMan/experiments/cna_isolate/fit_isolate.py -v \
	$cohort $gene $classif \
	--task_count=$(( $array_size + 1 )) --task_id=$SLURM_ARRAY_TASK_ID \
	--tune_splits=8 --test_count=32 --infer_splits=40 --parallel_jobs=8

