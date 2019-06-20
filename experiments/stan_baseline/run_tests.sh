#!/bin/bash

#SBATCH --job-name=stan-baseline
#SBATCH --partition=exacloud
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts t:g:m:r var
do
	case "$var" in
		t)	cohort=$OPTARG;;
		g)	gene=$OPTARG;;
		m)	model=$OPTARG;;
		r)	rewrite=true;;
		[?])	echo "Usage: $0 [-e] expression source directory" \
			     "[-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation classifier [-m] maximum tests per node" \
			     "[-r] whether existing results should be rewritten"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
out_tag=${cohort}__${gene}
OUTDIR=$TEMPDIR/HetMan/stan_baseline/${cohort}__${gene}/$model
FINALDIR=$DATADIR/HetMan/stan_baseline/$out_tag

export RUNDIR=$CODEDIR/HetMan/experiments/stan_baseline
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove intermediate output directory
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

dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $subtype_file \
	-d $RUNDIR/setup_tests.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/vars-list.p \
	-m setup/vars-count.txt -f setup.dvc --overwrite-dvcfile \
	python $RUNDIR/setup_tests.py $cohort $gene --setup_dir $OUTDIR

if [ -d .snakemake ]
then
	snakemake --unlock
fi

dvc run -d setup/cohort-data.p -d setup/vars-list.p -d $RUNDIR/fit_tests.py \
	-d $RUNDIR/models/${model%%'__'*}.py -o $FINALDIR/out-data__${model}.p \
	-f output.dvc --overwrite-dvcfile --remove-outs --no-commit \
	'snakemake -s $RUNDIR/Snakefile -j 100 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster \
	"sbatch -p {cluster.partition} -J {cluster.job-name} -t {cluster.time} \
	-o {cluster.output} -e {cluster.error} -n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config cohort='"$cohort"' use_gene='"$gene"' model='"$model"' \
	vars_count='"$( cat setup/vars-count.txt )"

cp output.dvc $FINALDIR/output__${model}.dvc

