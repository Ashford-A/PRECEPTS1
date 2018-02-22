#!/bin/bash

#SBATCH --job-name=icgc-trans-base
#SBATCH --partition=exacloud

#SBATCH --array=0-9
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/icgc-trans-base_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/icgc-trans-base_err-%A.txt
#SBATCH --verbose


# move to the working directory, find where to place output
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $OUTDIR
echo $classif

# pause between starting array jobs to ease load when downloading -omic datasets
sleep $(($SLURM_ARRAY_TASK_ID * 11));

# find the expression effects for the training/testing cohort split defined
# by the cross-validation ID and the sub-variant subset defined by the sub-task ID
srun -p=exacloud \
	--output=$BASEDIR/setup/slurm_${classif}_${SLURM_ARRAY_TASK_ID}.txt \
	--error=$BASEDIR/setup/slurm_${classif}_${SLURM_ARRAY_TASK_ID}.err \
	python $BASEDIR/setup_baseline.py $classif $SLURM_ARRAY_TASK_ID -v

