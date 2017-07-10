#!/bin/bash

#SBATCH --job-name=drug-pred
#SBATCH --partition=exacloud
#SBATCH --mem=16000
#SBATCH --time=400

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/dp_out-%j.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/dp_err-%j.err
#SBATCH --verbose

cd ~/compbio/bergamot
source activate HetMan

if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to TCGA-BRCA"
	cohort="TCGA-BRCA"
fi

TEMPDIR=HetMan/experiments/drug_predictions/output/$cohort
echo $TEMPDIR
rm -rf $TEMPDIR
mkdir -p $TEMPDIR/slurm

python HetMan/experiments/drug_predictions/drug_predict.py $cohort 55

