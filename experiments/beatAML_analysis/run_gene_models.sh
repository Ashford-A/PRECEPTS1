#!/bin/bash

#SBATCH --job-name=beatAML-models_gene
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4000
#SBATCH --time=2150


export OMP_NUM_THREADS=1
sleep $(($SLURM_ARRAY_TASK_ID * 7));

export BASEDIR=HetMan/experiments/beatAML_analysis
source activate HetMan
srun -p=exacloud python $BASEDIR/fit_gene_models.py \
	--cv_id=$SLURM_ARRAY_TASK_ID

