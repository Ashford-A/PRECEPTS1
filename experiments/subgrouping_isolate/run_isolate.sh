#!/bin/bash
#SBATCH --job-name=subg-iso
#SBATCH --verbose


source activate HetMan
rewrite=false
count_only=false

# collect command line arguments
while getopts e:t:l:s:c:m:rn var
do
	case "$var" in
		e)      expr_source=$OPTARG;;
		t)	cohort=$OPTARG;;
		l)	mut_levels=$OPTARG;;
		s)	search=$OPTARG;;
		c)	classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)      rewrite=true;;
		n)      count_only=true;;
		[?])    echo "Usage: $0 " \
				"[-e] cohort expression source" \
				"[-t] tumour cohort" \
				"[-l] mutation annotation levels" \
				"[-s] mutation search parameters" \
				"[-c] mutation classifier" \
				"[-m] maximum number of tests per node" \
				"[-r] rewrite existing results?" \
				"[-n] only enumerate, don't classify?"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/HetMan/subgrouping_isolate/$expr_source/$cohort/$mut_levels/$search/$classif
FINALDIR=$DATADIR/HetMan/subgrouping_isolate/${expr_source}__${cohort}
export RUNDIR=$CODEDIR/HetMan/experiments/subgrouping_isolate
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored, move to working directory
mkdir -p $FINALDIR $TEMPDIR/HetMan/subgrouping_isolate/$expr_source/$cohort/setup
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

# initiate version control in this directory if it hasn't been already
if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

# enumerate the mutation types that will be tested in this experiment
dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $subtype_file \
	-d $RUNDIR/setup_isolate.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/muts-list.p -m setup/muts-count.txt \
	-f setup.dvc --overwrite-dvcfile python $RUNDIR/setup_isolate.py \
	$expr_source $cohort $mut_levels $search $OUTDIR

# if we are only enumerating, we quit before classification jobs are launched
if $count_only
then
	exit 0
fi

# calculate how many parallel tasks the mutations will be tested over
muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))

# remove the Snakemake locks on the working directory if present
if [ -d .snakemake ]
then
	snakemake --unlock
	rm -rf .snakemake/locks/*
fi

dvc run -d setup/muts-list.p -d $RUNDIR/fit_isolate.py -f output.dvc \
	--overwrite-dvcfile --no-commit 'snakemake -s $RUNDIR/Snakefile \
	-j 200 --latency-wait 120 --cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} \
	-t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config expr_source='"$expr_source"' cohort='"$cohort"' \
	mut_levels='"$mut_levels"' search='"$search"' \
	classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${mut_levels}__${search}__${classif}.dvc

