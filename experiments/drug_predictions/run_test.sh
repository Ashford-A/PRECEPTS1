#!/bin/bash

#SBATCH --job-name=drug-pred
#SBATCH --partition=exacloud
#SBATCH --time=500

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/drugs_%j_out.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/drugs_%j_err.txt
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

python HetMan/experiments/drug_predictions/drug_predict.py $cohort ElasticNet 55

