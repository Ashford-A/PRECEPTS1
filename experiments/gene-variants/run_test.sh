#!/bin/bash

#SBATCH --job-name=gn-vars
#SBATCH --partition=exacloud
#SBATCH --mem=16000
#SBATCH --time=300

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/gene-vars_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/gene-vars_%j.err
#SBATCH --verbose

cd ~/compbio/bergamot
source activate HetMan

if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

if [ -z ${gene+x} ]
then
	echo "no gene defined, defaulting to TP53"
	export gene="TP53"
fi

TEMPDIR=HetMan/experiments/gene-variants/output/$cohort/$gene
echo $TEMPDIR
rm -rf $TEMPDIR

mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

srun -p=exacloud --ntasks=1 --cpus-per-task=1 --output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/gene-variants/setup.py $cohort $gene

sbatch HetMan/experiments/gene-variants/fit.sh

