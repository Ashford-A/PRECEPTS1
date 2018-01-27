#!/bin/bash

#SBATCH --job-name=stan-genes
#SBATCH --partition=exacloud
#SBATCH --mem=4000
#SBATCH --time=100

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-genes_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-genes_%j.err
#SBATCH --verbose


cd ~/compbio/bergamot
source activate precepts

if [ -z ${out_tag+x} ]
then
	echo "no output tag defined, defaulting to 'base'"
	export out_tag="base"
fi

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

if [ -z ${solve_type+x} ]
then
	echo "no Bayesian solving type defined, defaulting to optimization"
	export solve_type="optim"
fi

if [ -z ${baseline+x} ]
then
	echo "no baseline algorithm defined, defaulting to Lasso"
	export baseline="Lasso"
fi

echo $OUTDIR
rm -rf $OUTDIR

# creates the output directory and sub-directories
mkdir -p $OUTDIR/slurm
mkdir -p $OUTDIR/tmp
mkdir -p $OUTDIR/results

if [ ! -e HetMan/experiments/stan_test/genes/setup/${cohort}__mtype_list.p ]
then
	srun -p=exacloud \
		--output=$OUTDIR/slurm/setup_cohort.txt \
		--error=$OUTDIR/slurm/setup_cohort.err \
		python HetMan/experiments/stan_test/genes/setup_cohort.py $cohort -v
fi

if [ $(find HetMan/experiments/stan_test/genes/setup/baseline_perf/${cohort}_${baseline}__cv*__task*.p | wc -l) -lt 20 ]
then

	base_job=$(sbatch -p exacloud \
		--output=$OUTDIR/slurm/setup_baseline.txt \
		--error=$OUTDIR/slurm/setup_baseline.err \
		HetMan/experiments/stan_test/genes/setup_baseline.sh)

	job_id=${base_job#"Submitted batch job "}
	sbatch --depend=afterok:${job_id} \
		HetMan/experiments/stan_test/genes/fit_models.sh

else
	sbatch HetMan/experiments/stan_test/genes/fit_models.sh
fi

