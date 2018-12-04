#!/bin/bash

#SBATCH --job-name=var-baseline
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=8000
#SBATCH --time=500


source activate HetMan
export RUNDIR=$CODEDIR/HetMan/experiments/variant_baseline
export BASEDIR=$DATADIR/HetMan/variant_baseline
mkdir -p $BASEDIR/setup

while getopts e:t:s:c:m: var
do
	case "$var" in
		e)	export expr_source=$OPTARG;;
		t)	export cohort=$OPTARG;;
		s)	export samp_cutoff=$OPTARG;;
		c)	export classif=$OPTARG;;
		m)	export test_max=$OPTARG;;
		[?])	echo "Usage: $0 [-e] expression source directory" \
			     "[-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation classifier [-m] maximum tests per node";
			exit 1;;
	esac
done

export OUTDIR=$BASEDIR/output/$expr_source/${cohort}__samps-${samp_cutoff}/$classif
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

if [ ! -e $BASEDIR/setup/vars-list_${expr_source}__${cohort}__samps-${samp_cutoff}.p ]
then
	srun python $RUNDIR/setup_tests.py $expr_source $cohort $samp_cutoff
fi

vars_count=$(cat $BASEDIR/setup/vars-count_${expr_source}__${cohort}__samps-${samp_cutoff}.txt)
export array_size=$(( ($vars_count / $test_max + 1) * 25 - 1 ))

if [ $array_size -gt 299 ]
then
	export array_size=299
fi

sbatch --output=${slurm_dir}/var-baseline-fit.out \
	--error=${slurm_dir}/var-baseline-fit.err \
	--exclude=$ex_nodes --no-requeue \
	--array=0-$((array_size)) $RUNDIR/fit_tests.sh
