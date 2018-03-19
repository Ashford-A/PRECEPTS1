#!/bin/bash

#SBATCH --job-name=icgc-transfer-fit
#SBATCH --partition=exacloud

#SBATCH --array=0-59
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/icgc-transfer-fit_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/icgc-transfer-fit_err-%A.txt
#SBATCH --verbose


# pause between starting array jobs to ease load when downloading -omic datasets
sleep $(($SLURM_ARRAY_TASK_ID * 13));

cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
export OMP_NUM_THREADS=1
echo $OUTDIR
echo $classif
echo $mtypes

# get the cross-validation ID and sub-variant sub-task ID defined by this
# job's SLURM array ID
cv_id=$(($SLURM_ARRAY_TASK_ID % 10));
task_id=$(($SLURM_ARRAY_TASK_ID / 10));

srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${cv_id}_${task_id}.txt \
	--error=$OUTDIR/slurm/fit-${cv_id}_${task_id}.err \
	python HetMan/experiments/icgc_transfer_test/fit_models.py \
	$classif $mtypes $cv_id $task_id -v

