#!/bin/bash

#SBATCH --job-name=stan-genes-models
#SBATCH --partition=exacloud

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-genes-models_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-genes-models_err-%A.txt
#SBATCH --verbose

#SBATCH --array=0-49
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6000

if [ $solve_type == 'sampling' ];
then
	echo $solve_type
	#SBATCH --cpus-per-task=8

else
	echo $solve_type
	#SBATCH --cpus-per-task=1
fi

# pause between starting array jobs to reduce disk stress when loading
# packages, -omic datasets, etc., then move to working directory
sleep $(($SLURM_ARRAY_TASK_ID * 13));
cd ~/compbio/bergamot

# determine where the job's output will go, load software packages
OUTDIR=HetMan/experiments/stan_test/genes/models/$out_tag/$cohort/$solve_type
source activate precepts

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $out_tag
echo $cohort
echo $baseline

# get the cross-validation ID and sub-variant sub-task ID defined by this
# job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID % 5));
task_id=$(($SLURM_ARRAY_TASK_ID / 5));

srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${cv_id}_${task_id}.err \
	python HetMan/experiments/stan_test/genes/fit_models.py \
	$out_tag $cohort $solve_type $cv_id $task_id -v

