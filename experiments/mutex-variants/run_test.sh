#!/bin/bash

#SBATCH --job-name=var-mutex
#SBATCH --partition=exacloud
#SBATCH --mem=16000
#SBATCH --time=300

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/mutex_out_%j.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/mutex_err_%j.txt
#SBATCH --verbose

cd ~/compbio/bergamot
source activate HetMan

if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

TEMPDIR=HetMan/experiments/mutex-variants/output_new/$cohort
echo $TEMPDIR
rm -rf $TEMPDIR
mkdir -p $TEMPDIR/slurm

srun -p=exacloud --ntasks=1 --cpus-per-task=1 --output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/mutex-variants/setup.py $cohort

sbatch HetMan/experiments/mutex-variants/fit.sh

