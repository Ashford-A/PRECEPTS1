#!/bin/bash
#SBATCH --job-name=copy-baseline
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts e:t:s:c:m:r var
do
	case "$var" in
		e)	expr_source=$OPTARG;;
		t)	cohort=$OPTARG;;
		s)	samp_cutoff=$OPTARG;;
		c)	regress=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)	rewrite=true;;
		[?])	echo "Usage: $0 [-e] expression source directory" \
			     "[-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation regressor [-m] maximum tests per node" \
			     "[-r] whether existing results should be rewritten"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
out_tag=${cohort}__samps-${samp_cutoff}
OUTDIR=$TEMPDIR/HetMan/copy_baseline/$expr_source/${out_tag}/$regress
FINALDIR=$DATADIR/HetMan/copy_baseline/${expr_source}__$out_tag

export RUNDIR=$CODEDIR/HetMan/experiments/copy_baseline
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create intermediate and final output directories, move to working directory
mkdir -p $FINALDIR $TEMPDIR/HetMan/copy_baseline/$expr_source/setup
mkdir -p $TEMPDIR/HetMan/copy_baseline/$expr_source/$out_tag/setup
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

dvc run -d $firehose_dir -d $gencode_file -d $gene_file -d $subtype_file \
	-d $RUNDIR/setup_tests.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/gene-list.p \
	-m setup/gene-count.txt -f setup.dvc --overwrite-dvcfile \
	python $RUNDIR/setup_tests.py $expr_source $cohort $samp_cutoff $OUTDIR

gene_count=$( cat setup/gene-count.txt )
task_count=$(( $gene_count / $test_max + 1 ))

if [ -d .snakemake ]
then
	snakemake --unlock
	rm .snakemake/locks/*
fi

dvc run -d setup/cohort-data.p -d setup/gene-list.p -d $RUNDIR/fit_tests.py \
	-d $RUNDIR/models/${regress%%'__'*}.py -o $FINALDIR/out-data__${regress}.p \
	-f output.dvc --overwrite-dvcfile --remove-outs --no-commit \
	'snakemake -s $RUNDIR/Snakefile -j 102 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster \
	"sbatch -p {cluster.partition} -J {cluster.job-name} -t {cluster.time} \
	-o {cluster.output} -e {cluster.error} -n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config expr_source='"$expr_source"' cohort='"$cohort"' \
	samp_cutoff='"$samp_cutoff"' regress='"$regress"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${regress}.dvc

