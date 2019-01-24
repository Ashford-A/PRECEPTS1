#!/bin/bash

#SBATCH --job-name=subv-inf_fit
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
sleep $(($SLURM_ARRAY_TASK_ID * 13));

# gets the sub-task ID defined by this job's SLURM array ID and the directory
# where the subtypes to be tested were saved
task_id=$SLURM_ARRAY_TASK_ID;
SETUP_DIR=$BASEDIR/setup/$cohort/$gene

# test the subtypes corresponding to the given sub-task ID
srun --output=$OUTDIR/slurm/fit-${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${task_id}.err \
	python $RUNDIR/fit_infer.py $cohort $classif $gene \
	$mut_levels --samp_cutoff=$samp_cutoff \
	--task_count=$(( $array_size + 1 )) --task_id=$task_id \

