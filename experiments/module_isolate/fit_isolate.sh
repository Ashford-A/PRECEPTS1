#!/bin/bash

#SBATCH --job-name=module-isolate_fit
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --time=2150
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000


export OMP_NUM_THREADS=1
echo $OUTDIR
echo $array_size

sleep $(($SLURM_ARRAY_TASK_ID * 7));
SETUP_DIR=$BASEDIR/setup/${cohort}/${gene_lbl}

# test the subtypes corresponding to the given sub-task ID
srun -p=exacloud \
	--output=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.txt \
	--error=$OUTDIR/slurm/fit-${SLURM_ARRAY_TASK_ID}.err \
	python HetMan/experiments/utilities/isolate_mutype_infer.py -v \
	$SETUP_DIR/mtypes_list__samps_${samp_cutoff}__levels_${mut_levels}.p \
	$OUTDIR $cohort $classif \
	--task_count=$(( $array_size + 1 )) --task_id=${SLURM_ARRAY_TASK_ID} \
	--tune_splits=4 --test_count=32 --infer_splits=40 --parallel_jobs=8

