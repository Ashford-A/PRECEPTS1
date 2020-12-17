#!/bin/bash
#SBATCH --job-name=dyad-iso
#SBATCH --verbose


start_time=$( date +%s )
source activate research
rewrite=false
count_only=false

# collect command line arguments
while getopts :e:t:s:l:c:m:rn var
do
	case "$var" in
		e)  expr_source=$OPTARG;;
		t)  cohort=$OPTARG;;
		s)  search=$OPTARG;;
		l)  mut_lvls=$OPTARG;;
		c)  classif=$OPTARG;;
		m)  test_max=$OPTARG;;
		r)  rewrite=true;;
		n)  count_only=true;;
		[?])  echo "Usage: $0 " \
				"[-e] cohort expression source" \
				"[-t] tumour cohort" \
				"[-s] mutation search parameters" \
				"[-l] mutation annotation levels" \
				"[-c] mutation classifier" \
				"[-m] maximum number of tests per node" \
				"[-r] rewrite existing results?" \
				"[-n] only enumerate, don't classify?"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/dryads-research/dyad_isolate/$expr_source/$cohort/$search/$mut_lvls/$classif
FINALDIR=$DATADIR/dryads-research/dyad_isolate/${expr_source}__${cohort}
export RUNDIR=$CODEDIR/dryads-research/experiments/dyad_isolate
out_tag=${search}_${mut_lvls}_${classif}

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# move to working directory, clean up files from previous experiment runs, and
# create the directories where intermediate and final output will be stored
mkdir -p $FINALDIR $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm $OUTDIR/merge
cd $OUTDIR || exit

rm -rf .snakemake
dvc init --no-scm -f
export PYTHONPATH="$CODEDIR"

eval "$( python -m dryads-research.experiments.utilities.data_dirs \
	$cohort --expr_source $expr_source )"

# enumerate the mutation types that will be tested in this experiment
dvc run -d $COH_DIR -d $GENCODE_DIR -d $ONCOGENE_LIST -d $SUBTYPE_LIST \
	-d $RUNDIR/setup_isolate.py -d $CODEDIR/dryads-research/environment.yml \
	-o setup/muts-list.p -m setup/muts-count.txt \
	-f setup.dvc --overwrite-dvcfile \
	python -m dryads-research.experiments.dyad_isolate.setup_isolate \
	$expr_source $cohort $search $mut_lvls $OUTDIR

if [ -z ${SBATCH_TIMELIMIT+x} ]
then
	time_lim=2159
else
	time_lim=$SBATCH_TIMELIMIT
fi

cur_time=$( date +%s )
time_left=$(( time_lim - (cur_time - start_time) / 60 + 1 ))

if [ -z ${time_max+x} ]
then
	time_max=$(( $time_left * 11 / 13 ))
fi

if [ ! -f setup/tasks.txt ]
then
	merge_max=$(( $time_left - $time_max - 3 ))

  if [ $classif == 'Ridge' ]
	then
		task_size=1.13
		samp_exp=1
	elif [ $classif == 'RidgeRobust' ]
	then
		task_size=2.73
		samp_exp=1.07
	elif [ $classif == 'RidgeFlat' ]
	then
		task_size=1.71
		samp_exp=1.07
	fi

	eval "$( python -m dryads-research.experiments.utilities.pipeline_setup \
		$OUTDIR $time_max --merge_max=$merge_max \
		--task_size=$task_size --samp_exp=$samp_exp --merge_size=3.41 )"
fi

# if we are only enumerating, we quit before classification jobs are launched
if $count_only
then
	cp setup/cohort-data.p.gz $FINALDIR/cohort-data__${out_tag}.p.gz
	exit 0
fi

eval "$( tail -n 2 setup/tasks.txt | head -n 1 )"
eval "$( tail -n 1 setup/tasks.txt )"

dvc run -d setup/muts-list.p -d $RUNDIR/fit_isolate.py -O out-conf.p.gz \
	-f output.dvc --overwrite-dvcfile --ignore-build-cache \
	'snakemake -s $RUNDIR/Snakefile \
	-j 1200 --latency-wait 120 --cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} \
	-t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config expr_source='"$expr_source"' cohort='"$cohort"' \
	search='"$search"' mut_lvls='"$mut_lvls"' classif='"$classif"' \
	time_max='"$run_time"' merge_max='"$merge_time"

cp output.dvc $FINALDIR/output_${out_tag}.dvc

