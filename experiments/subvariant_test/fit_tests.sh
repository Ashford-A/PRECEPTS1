#!/bin/bash

#SBATCH --job-name=subv-test_fit
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5000


# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $OUTDIR
echo $array_size

# pause between starting array jobs to ease I/O load
sleep $(($SLURM_ARRAY_TASK_ID * 7));

cv_id=$(($SLURM_ARRAY_TASK_ID % 10));
task_id=$(($SLURM_ARRAY_TASK_ID / 10));
SETUP_DIR=$BASEDIR/setup/${cohort}/${gene}

# test the subtypes corresponding to the given sub-task ID
srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${task_id}_${cv_id}.txt \
	--error=$OUTDIR/slurm/fit-${task_id}_${cv_id}.err \
	python $CODEDIR/HetMan/experiments/utilities/isolate_mutype_test.py \
	$SETUP_DIR/mtypes_list__samps_${samp_cutoff}__levels_${mut_levels}.p \
	$OUTDIR $cohort $classif $cv_id --use_genes $gene -v \
	--task_count=$(( $array_size / 10 + 1 )) --task_id=$task_id \
	--tune_splits=4 --test_count=48 --parallel_jobs=12

