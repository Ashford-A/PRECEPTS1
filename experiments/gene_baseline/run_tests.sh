#!/bin/bash

#SBATCH --job-name=gene-baseline
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=5000
#SBATCH --time=500


export BASEDIR=HetMan/experiments/gene_baseline
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

if [ ! -e $BASEDIR/setup/genes-list_${expr_source}__${cohort}__samps-${samp_cutoff}.p ]
then
	srun python $BASEDIR/setup_tests.py $expr_source $cohort $syn_root $samp_cutoff
fi

genes_count=$(cat $BASEDIR/setup/genes-count_${expr_source}__${cohort}__samps-${samp_cutoff}.txt)
export array_size=$(( ($genes_count / $test_max + 1) * 25 - 1 ))

if [ $array_size -gt 299 ]
then
	export array_size=299
fi

sbatch --output=${slurm_dir}/gene-baseline-fit.out \
	--error=${slurm_dir}/gene-baseline-fit.err \
	--array=0-$((array_size)) $BASEDIR/fit_tests.sh

