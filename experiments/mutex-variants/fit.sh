#!/bin/bash

#SBATCH --job-name=vm-fit
#SBATCH --partition=exacloud

#SBATCH --array=1-50
#SBATCH --time=600
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/ex-fit_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/ex-fit_err-%A.txt
#SBATCH --verbose

cd ~/compbio/bergamot
TEMPDIR=HetMan/experiments/mutex-variants/output_new/$cohort
export OMP_NUM_THREADS=1
echo $cohort

srun -p=exacloud --output=$TEMPDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.txt --error=$TEMPDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.err \
	python HetMan/experiments/mutex-variants/fit.py $cohort $SLURM_ARRAY_TASK_ID

