#!/bin/bash

#SBATCH --job-name=dream-stan
#SBATCH --partition=exacloud
#SBATCH --mem=32000
#SBATCH --time=1440
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-stan_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-stan_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate HetMan

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the pathway interaction type to use
if [ -z ${intx+x} ]
then
	echo "no pathway interaction type defined, defaulting to controls-expression-of"
	export intx="controls-expression-of"
fi

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/dream_train/stan/output/$cohort/$intx
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# submits the script that enumerates the gene sub-variants to be considered
srun --output=$TEMPDIR/slurm/run.txt --error=$TEMPDIR/slurm/run.err \
	python HetMan/experiments/dream_train/stan/run.py $cohort $intx

