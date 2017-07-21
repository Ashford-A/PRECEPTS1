#!/bin/bash

#SBATCH --job-name=genevar-fit
#SBATCH --partition=exacloud

#SBATCH --array=1-10
#SBATCH --time=600
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/genevar-fit_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/genevar-fit_err-%A.txt
#SBATCH --verbose

cd ~/compbio/bergamot
TEMPDIR=HetMan/experiments/gene-variants/output/$cohort/$gene
export OMP_NUM_THREADS=1
echo $cohort
echo $gene

sleep $(($SLURM_ARRAY_TASK_ID * 17));

srun -p=exacloud --output=$TEMPDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.txt --error=$TEMPDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.err \
	python HetMan/experiments/gene-variants/fit.py $cohort $gene $SLURM_ARRAY_TASK_ID

