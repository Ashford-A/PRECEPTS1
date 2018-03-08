#!/bin/bash

#SBATCH --job-name=icgc-transfer
#SBATCH --partition=exacloud
#SBATCH --mem=4000
#SBATCH --time=1000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/icgc-transfer_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/icgc-transfer_%j.err
#SBATCH --verbose


# move to working directory, load required packages and modules
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
source activate precepts

if [ -z ${classif+x} ]
then
	echo "no mutation classifier defined, defaulting to Lasso"
	export classif="Lasso"
fi

export BASEDIR=HetMan/experiments/icgc_transfer_test
export OUTDIR=$BASEDIR/output/$classif

echo $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

if [ ! -e ${BASEDIR}/setup/cohort_genes.p ]
then
	echo "find genes to transfer..."

	srun -p=exacloud \
		--output=$BASEDIR/setup/slurm_genes.txt \
		--error=$BASEDIR/setup/slurm_genes.err \
		python $BASEDIR/setup_genes.py
fi

if [ $(find $BASEDIR/setup/${classif}__cv_*.p | wc -l) -lt 10 ]
then
	base_job=$(sbatch -p exacloud \
		--output=$BASEDIR/setup/slurm_${classif}.txt \
		--error=$BASEDIR/setup/slurm_${classif}.err \
		$BASEDIR/setup_baseline.sh)
	
	job_id=${base_job#"Submitted batch job "}
	sbatch --depend=afterok:${job_id} $BASEDIR/fit_models.sh

else
	sbatch $BASEDIR/fit_models.sh
fi

