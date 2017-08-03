#!/bin/bash

#SBATCH --job-name=genevar-fit
#SBATCH --partition=exacloud

#SBATCH --array=0-39
#SBATCH --time=600
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/genevar-fit_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/genevar-fit_err-%A.txt
#SBATCH --verbose


# move to the working directory, find where to place output
cd ~/compbio/bergamot
TEMPDIR=HetMan/experiments/gene_subvariants/output/$cohort/$gene

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $cohort
echo $gene

# pause between starting array jobs to allow BMEG to take a nap-nap and recover
sleep $(($SLURM_ARRAY_TASK_ID * 23));

# get the cross-validation ID and sub-variant sub-task ID defined by this
# job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID / 8));
task_id=$(($SLURM_ARRAY_TASK_ID % 8));

# find the expression effects for the training/testing cohort split defined
# by the cross-validation ID and the sub-variant subset defined by the sub-task ID
srun -p=exacloud \
	--output=$TEMPDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$TEMPDIR/slurm/fit-${cv_id}_${task_id}.err \
	python HetMan/experiments/gene_subvariants/fit.py $cohort $gene $cv_id $task_id

