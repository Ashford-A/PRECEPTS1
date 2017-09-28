#!/bin/bash

#SBATCH --job-name=dream-train
#SBATCH --partition=exacloud
#SBATCH --mem=2000
#SBATCH --time=20

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-train_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-train_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate HetMan

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the name of the -omics dataset(s) to use as prediction input
if [ -z ${input+x} ]
then
	echo "no input defined, defaulting to rna"
	export input="rna"
fi

# finds the name of the regressor to use for prediction
if [ -z ${classif+x} ]
then
	echo "no classifier, defaulting to ElasticNet"
	export classif="ElasticNet"
fi

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/dream_train/baseline/output/$cohort/$input/$classif
echo $TEMPDIR
rm -rf $TEMPDIR

mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

srun -p=exacloud \
	--output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/dream_train/baseline/setup.py \
	$cohort $input $classif

sbatch HetMan/experiments/dream_train/baseline/fit.sh

