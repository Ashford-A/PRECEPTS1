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

cd $CODEDIR
eval $(python -m HetMan.experiments.subgrouping_isolate.data_dirs \
	$expr_source $cohort)

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored,
# move to working directory
mkdir -p $FINALDIR $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm $OUTDIR/merge
cd $OUTDIR

rm -rf .snakemake
dvc init --no-scm -f
export PYTHONPATH="$CODEDIR"

# enumerate the mutation types that will be tested in this experiment
dvc run -d $COH_DIR -d $GENCODE_DIR -d $ONCOGENE_LIST -d $SUBTYPE_LIST \
	-d $RUNDIR/setup_isolate.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/muts-list.p -m setup/muts-count.txt \
	-f setup.dvc --overwrite-dvcfile \
	python -m HetMan.experiments.subgrouping_isolate.setup_isolate \
	$expr_source $cohort $mut_levels $search $OUTDIR

# if we are only enumerating, we quit before classification jobs are launched
if $count_only
then
	exit 0
fi

# calculate how many parallel tasks the mutations will be tested over
merge_max=2000
muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))
merge_count=$(( $(( $muts_count - 1)) / $merge_max + 1 ))
xargs -n $merge_count <<< $(seq 0 $(( $task_count - 1 ))) > setup/tasks.txt

dvc run -d setup/muts-list.p -d $RUNDIR/fit_isolate.py -O out-siml.p.gz \
	-f output.dvc --overwrite-dvcfile --ignore-build-cache \
	'snakemake -s $RUNDIR/Snakefile \
	-j 200 --latency-wait 120 --cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} \
	-t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config expr_source='"$expr_source"' cohort='"$cohort"' \
	mut_levels='"$mut_levels"' search='"$search"' classif='"$classif"

cp output.dvc $FINALDIR/output__${mut_levels}__${search}__${classif}.dvc

