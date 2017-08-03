#!/bin/bash

#SBATCH --job-name=gn-vars
#SBATCH --partition=exacloud
#SBATCH --mem=24000
#SBATCH --time=300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/tmp/gene-vars_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/tmp/gene-vars_%j.err
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
TEMPDIR=HetMan/experiments/gene_subvariants/output/$cohort/$gene/$classif
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

# submits the script that enumerates the gene sub-variants to be considered
srun -p=exacloud --ntasks=1 --cpus-per-task=1 \
	--output=$TEMPDIR/slurm/setup.txt --error=$TEMPDIR/slurm/setup.err \
	python HetMan/experiments/gene_subvariants/setup.py $cohort $gene $classif

# submits the array script that finds these sub-variants' expression effects
sbatch HetMan/experiments/gene_subvariants/fit.sh

srun -p=exacloud \
	--output=$TEMPDIR/slurm/fit_cna.txt --error=$TEMPDIR/slurm/fit_cna.err \
	python HetMan/experiments/gene_subvariants/fit_cna.py $cohort $gene $classif

