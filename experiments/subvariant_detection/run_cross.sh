#!/bin/bash

#SBATCH --job-name=subv-cross
#SBATCH --partition=exacloud
#SBATCH --mem=4000
#SBATCH --time=200

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-cross_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-cross_%j.err
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
	echo "no classifier defined, defaulting to Lasso"
	export classif="Lasso"
fi

if [ -z ${gene+x} ]
then
	echo "no gene defined, defaulting to TP53"
	export gene="TP53"
fi

# gets the output directory where results will be saved, removing it if it already exists
TEMPDIR=HetMan/experiments/subvariant_detection/output/$cohort/$classif/cross/$gene
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# submits the script that enumerates the gene sub-variants to be considered
srun -p=exacloud \
	--output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/subvariant_detection/setup_cross.py \
	$cohort $classif $gene -v

sbatch HetMan/experiments/subvariant_detection/fit_cross.sh

