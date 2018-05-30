#!/bin/bash

#SBATCH --job-name=dyad-isolate_fit
#SBATCH --partition=exacloud

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dyad-isolate_fit_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dyad-isolate_fit_err-%A.txt
#SBATCH --verbose


# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $array_size
echo $BASEDIR
echo $OUTDIR

# pause between starting array jobs to ease I/O load
sleep $(($SLURM_ARRAY_TASK_ID * 7));

# gets the sub-task ID defined by this job's SLURM array ID and the directory
# where the subtypes to be tested were saved
task_id=$SLURM_ARRAY_TASK_ID;
SETUP_DIR=$BASEDIR/setup/${cohort}

# test the subtypes corresponding to the given sub-task ID
srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${task_id}.err \
	python HetMan/experiments/utilities/pair_mutype_infer.py -v \
	$SETUP_DIR/pairs_list__samps_${samp_cutoff}.p $OUTDIR $cohort $classif \
	--task_count=$(( $array_size + 1 )) --task_id=$task_id \
	--tune_splits=4 --test_count=32 --infer_splits=40 --parallel_jobs=8

