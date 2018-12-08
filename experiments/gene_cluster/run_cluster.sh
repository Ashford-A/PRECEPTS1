#!/bin/bash

#SBATCH --job-name=gene-cluster
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=8000
#SBATCH --time=500


source activate HetMan
export RUNDIR=$CODEDIR/HetMan/experiments/gene_cluster
export BASEDIR=$DATADIR/HetMan/gene_cluster
mkdir -p $BASEDIR/setup

while getopts e:t:s:m: var
do
	case "$var" in
		e)      export expr_source=$OPTARG;;
		t)	export cohort=$OPTARG;;
		s)	export samp_cutoff=$OPTARG;;
		m)	export test_max=$OPTARG;;
		[?])    echo "Usage: $0 [-e] expression source directory" \
			     "[-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-m] maximum tests per node";
			exit 1;;
	esac
done

export OUTDIR=$BASEDIR/output/$expr_source/${cohort}__samps-${samp_cutoff}
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

if [ ! -e $BASEDIR/setup/genes-list_${expr_source}__${cohort}__samps-${samp_cutoff}.p ]
then
	srun --output=$BASEDIR/setup/slurm_${cohort}.txt \
		--error=$BASEDIR/setup/slurm_${cohort}.err \
		python $RUNDIR/setup_cluster.py -v \
		$expr_source $cohort --samp_cutoff=$samp_cutoff
fi

genes_count=$(cat $BASEDIR/setup/genes-count_${expr_source}__${cohort}__samps-${samp_cutoff}.txt)
export array_size=$(( $genes_count / $test_max ))

if [ $array_size -gt 99 ]
then
	export array_size=99
fi

# run the subtype tests in parallel
sbatch --output=$slurm_dir/gene-clust-fit.out \
	--error=$slurm_dir/gene-clust-fit.err \
	--exclude=$ex_nodes --no-requeue \
	--array=0-$(( $array_size )) $RUNDIR/fit_cluster.sh

