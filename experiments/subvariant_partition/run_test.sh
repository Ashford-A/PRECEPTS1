#!/bin/bash

#SBATCH --job-name=gn-part
#SBATCH --partition=exacloud
#SBATCH --mem-per-cpu=4000
#SBATCH --time=600
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/gn-part_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/gn-part_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate HetMan

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the name of the gene whose sub-variants are to be considered
if [ -z ${gene+x} ]
then
	echo "no gene defined, defaulting to TP53"
	export gene="TP53"
fi

# finds the name of classifier used to identify expression signatures
if [ -z ${classif+x} ]
then
	echo "no classifier defined, defaulting to HetMan.predict.classifers.Lasso"
	export classif="Lasso"
fi

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/subvariant_partition/output/$cohort/$gene/$classif
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# submits the array script that finds these sub-variants' expression effects
srun python HetMan/experiments/subvariant_partition/fit.py $cohort $gene $classif 77

