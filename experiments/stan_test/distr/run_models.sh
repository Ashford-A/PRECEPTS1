#!/bin/bash

#SBATCH --job-name=stan-distr
#SBATCH --partition=exacloud

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-distr_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-distr_%j.err
#SBATCH --verbose

#SBATCH --array=0-9
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=8


# pause between starting array jobs to reduce disk stress when loading
# packages, -omic datasets, etc., then move to working directory
sleep $(($SLURM_ARRAY_TASK_ID * 13));
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
source activate precepts

# finds the label of the Stan model to use
if [ -z ${model_name+x} ]
then
	echo "no model name defined, defaulting to 'base'"
	export model_name="base"
fi

# finds which method to use to find the Stan parameter values
if [ -z ${solver+x} ]
then
	echo "no parameter search method defined, defaulting to optimization"
	export solver="optim"
fi

# finds the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort defined, defaulting to BRCA"
	export cohort="BRCA"
fi

# finds the gene whose mutations we'll use for prediction
if [ -z ${gene+x} ]
then
	echo "no mutated gene defined, defaulting to TP53"
	export gene="TP53"
fi

export OMP_NUM_THREADS=1
export BASEDIR=HetMan/experiments/stan_test/distr
export OUTDIR=$BASEDIR/output/$model_name/$solver/$cohort/$gene

echo $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.txt \
	--error=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.err \
	python HetMan/experiments/stan_test/distr/fit_models.py \
	$model_name $solver $cohort $gene $SLURM_ARRAY_TASK_ID -v

