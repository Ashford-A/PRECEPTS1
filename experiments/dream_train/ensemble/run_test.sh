#!/bin/bash

#SBATCH --job-name=dream-ensemble
#SBATCH --partition=exacloud

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-ens_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-ens_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate visions

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/dream_train/ensemble/output
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/regressors

# submits the script that enumerates the gene sub-variants to be considered
srun --output=$TEMPDIR/slurm/run.txt --error=$TEMPDIR/slurm/run.err \
	python HetMan/experiments/dream_train/ensemble/fit.py

