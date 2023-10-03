#!/bin/bash

#SBATCH --job-name=drug-pred
#SBATCH --partition=exacloud
#SBATCH --time=500

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/drugs_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/drugs_%j.err
#SBATCH --verbose

cd ~/compbio/bergamot
source activate HetMan

if [ -z ${patient+x} ]
then
	echo "no patient defined, defaulting to SMRT_02299"
	export patient="SMRT_02299"
fi

if [ -z ${clf+x} ]
then
	echo "no classifier defined, defaulting to ElasticNet"
	export clf="ElasticNet"
fi

TEMPDIR=HetMan/experiments/drug_predictions/output/$patient/$clf
echo $TEMPDIR
rm -rf $TEMPDIR
mkdir -p $TEMPDIR/slurm

export OMP_NUM_THREADS=1

python HetMan/experiments/drug_predictions/drug_predict.py $patient $clf 55

