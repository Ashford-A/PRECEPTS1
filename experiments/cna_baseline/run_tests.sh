#!/bin/bash

#SBATCH --job-name=cna-baseline
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=5000
#SBATCH --time=500


export BASEDIR=HetMan/experiments/cna_baseline
source activate precepts

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

if [ ! -e $BASEDIR/setup/mtypes-list_${expr_source}__${cohort}__samps-${samp_cutoff}.p ]
then
	srun python $BASEDIR/setup_tests.py $expr_source $cohort $samp_cutoff
fi

mtypes_count=$(cat $BASEDIR/setup/mtypes-count_${expr_source}__${cohort}__samps-${samp_cutoff}.txt)
export array_size=$(( ($mtypes_count / $test_max + 1) * 10 - 1 ))

if [ $array_size -gt 499 ]
then
	export array_size=499
fi

sbatch --output=${slurm_dir}/cna-baseline-fit.out \
	--error=${slurm_dir}/cna-baseline-fit.err \
	--array=0-$((array_size)) $BASEDIR/fit_tests.sh

