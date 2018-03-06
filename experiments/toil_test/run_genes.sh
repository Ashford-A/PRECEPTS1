#!/bin/bash

#SBATCH --job-name=toil-genes
#SBATCH --partition=exacloud
#SBATCH --mem=4000
#SBATCH --time=100

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/toil-genes_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/toil-genes_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate precepts

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the name of mutation prediction pipeline to use
if [ -z ${classif+x} ]
then
	echo "no classifier, defaulting to Lasso"
	export classif="Lasso"
fi

# gets the output directory where results will be saved, removing it if it already exists
TEMPDIR=HetMan/experiments/toil_test/output/$cohort/$classif/genes
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# submits the script that enumerates the gene sub-variants to be considered
srun -p=exacloud \
	--output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/toil_test/setup_genes.py \
	$cohort $classif -v

sbatch HetMan/experiments/toil_test/fit_genes.sh

