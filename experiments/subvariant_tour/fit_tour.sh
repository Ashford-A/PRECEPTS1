#!/bin/bash

#SBATCH --job-name=subv-tour_fit
#SBATCH --partition=exacloud

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-tour_fit_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-tour_fit_err-%A.txt
#SBATCH --verbose


# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $array_size
echo $BASEDIR
echo $OUTDIR

# pause between starting array jobs to ease I/O load
sleep $(($SLURM_ARRAY_TASK_ID * 13));

# gets the cross-validation ID and sub-task ID defined by this job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID % 5));
task_id=$(($SLURM_ARRAY_TASK_ID / 5));

srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${cv_id}_${task_id}.err \
	python HetMan/experiments/utilities/test_cohort_mutypes.py \
	$BASEDIR/setup/${cohort}/mtype_list__freq_${freq_cutoff}__levels_${mut_levels}.p \
	$OUTDIR $cohort $classif $cv_id $task_id --mut_levels=$mut_levels \
	--task_count=$(( $array_size / 5 + 1 )) --test_count=12 --parallel_jobs=12 -v

