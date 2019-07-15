#!/bin/bash

#SBATCH --job-name=var-mutex
#SBATCH --partition=exacloud
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts t:s:c:m:r var
do
	case "$var" in
		t)	export cohort=$OPTARG;;
		s)	export samp_cutoff=$OPTARG;;
		c)	export classif=$OPTARG;;
		m)	export test_max=$OPTARG;;
		r)	rewrite=true;;
		[?])	echo "Usage: $0 [-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation classifier [-m] maximum tests per node";
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/HetMan/variant_mutex/${cohort}__samps-${samp_cutoff}/$classif
FINALDIR=$DATADIR/HetMan/variant_mutex/${cohort}__samps-${samp_cutoff}
export RUNDIR=$CODEDIR/HetMan/experiments/variant_mutex
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create intermediate and final output directories, move to working directory
mkdir -p $FINALDIR
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $gene_file -d $subtype_file \
	-d $RUNDIR/setup_mutex.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/pairs-list.p -m setup/pairs-count.txt \
	-f setup.dvc --overwrite-dvcfile python $RUNDIR/setup_mutex.py $cohort \
	$samp_cutoff --setup_dir=$OUTDIR

pairs_count=$(cat setup/pairs-count.txt)
task_count=$(( $(( $pairs_count - 1 )) / $test_max + 1 ))

if [ -d .snakemake ]
then
	snakemake --unlock
fi

dvc run -d setup/cohort-data.p -d setup/pairs-list.p -d $RUNDIR/fit_mutex.py \
	-o $FINALDIR/out-data__${classif}.p -f output.dvc --overwrite-dvcfile \
	--remove-outs --no-commit 'snakemake -s $RUNDIR/Snakefile -j 100 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster "sbatch -p {cluster.partition} \
	-J {cluster.job-name} -t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} --mem-per-cpu {cluster.mem-per-cpu} \
	--exclude=$ex_nodes --no-requeue" --config cohort='"$cohort"' \
	samp_cutoff='"$samp_cutoff"' classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${classif}.dvc

