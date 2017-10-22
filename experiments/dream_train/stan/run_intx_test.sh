#!/bin/bash

#SBATCH --job-name=dream-stan-intx
#SBATCH --partition=exacloud
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-stan-intx_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/dream-stan-intx_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate visions

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to OV"
	export cohort="OV"
fi

# gets the output directory where results will be saved,
# removing it if it already exists
TEMPDIR=HetMan/experiments/dream_train/stan/output/intx/$cohort
echo $TEMPDIR
rm -rf $TEMPDIR

# creates the output directory and sub-directories
mkdir -p $TEMPDIR/slurm
mkdir -p $TEMPDIR/tmp
mkdir -p $TEMPDIR/results

intx_list=('controls-expression-of' 'controls-phosphorylation-of' 'controls-state-change-of')
for intx_use in "${intx_list[@]}";
	do export intx=$intx_use;
	sbatch HetMan/experiments/dream_train/stan/run_cv.sh;
done;

