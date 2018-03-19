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

if [ -z ${mtypes+x} ]
then
	echo "no mutation types defined, defaulting to genes"
	export mtypes="genes"
fi

export BASEDIR=HetMan/experiments/icgc_transfer_test
export OUTDIR=$BASEDIR/output/$classif/$mtypes

echo $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

if [ ! -e ${BASEDIR}/setup/cohort_${mtypes}.p ]
then
	echo "find ${mtypes} to transfer..."

	srun -p=exacloud \
		--output=$BASEDIR/setup/slurm_${mtypes}.txt \
		--error=$BASEDIR/setup/slurm_${mtypes}.err \
		python $BASEDIR/setup_${mtypes}.py
fi

if [ $(find $BASEDIR/setup/${classif}_${mtypes}__cv_*.p | wc -l) -lt 10 ]
then
	base_job=$(sbatch -p exacloud \
		--output=$BASEDIR/setup/slurm_${classif}_${mtypes}.txt \
		--error=$BASEDIR/setup/slurm_${classif}_${mtypes}.err \
		$BASEDIR/setup_baseline.sh)
	
	job_id=${base_job#"Submitted batch job "}
	sbatch --depend=afterok:${job_id} $BASEDIR/fit_models.sh

else
	sbatch $BASEDIR/fit_models.sh
fi

