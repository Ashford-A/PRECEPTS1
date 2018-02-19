#!/bin/bash

#SBATCH --job-name=stan-distr-models
#SBATCH --partition=exacloud

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-distr-models_out-%A.txt
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/stan-distr-models_err-%A.txt
#SBATCH --verbose

#SBATCH --array=0-9
#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --cpus-per-task=4


# pause between starting array jobs to reduce disk stress when loading
# packages, -omic datasets, etc., then move to working directory
sleep $(($SLURM_ARRAY_TASK_ID * 13));
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
source activate precepts

# disable threading on each CPU, print info about this array job
export OMP_NUM_THREADS=1
echo $model_name
echo $cohort
echo $gene

srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.txt \
	--error=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.err \
	python HetMan/experiments/stan_test/distr/fit_models.py \
	$model_name $cohort $gene $SLURM_ARRAY_TASK_ID -v

