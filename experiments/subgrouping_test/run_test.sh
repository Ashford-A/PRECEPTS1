#!/bin/bash
#SBATCH --job-name=subg-test
#SBATCH --verbose


# get current time, load conda environment, set default values for arguments
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
		s)  samp_cutoff=$OPTARG;;
		l)  mut_levels=$OPTARG;;
		c)  classif=$OPTARG;;
		m)  test_max=$OPTARG;;
		r)  rewrite=true;;
		n)  count_only=true;;
		[?])  echo "Usage: $0 " \
				"[-e] cohort expression source" \
				"[-t] tumour cohort" \
				"[-s] minimum samples for subgrouping search" \
				"[-l] mutation annotation levels" \
				"[-c] mutation classifier" \
				"[-m] maximum number of tests per node" \
				"[-r] rewrite existing results?" \
				"[-n] only enumerate, don't classify?"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
out_tag=${cohort}__samps-${samp_cutoff}
OUTDIR=$TEMPDIR/dryads-research/subgrouping_test/$expr_source/$out_tag/$mut_levels/$classif
FINALDIR=$DATADIR/dryads-research/subgrouping_test/${expr_source}__$out_tag
export RUNDIR=$CODEDIR/dryads-research/experiments/subgrouping_test

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored, move to working directory
mkdir -p $TEMPDIR/dryads-research/subgrouping_test/setup
mkdir -p $FINALDIR $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR || exit

rm -rf .snakemake
dvc init --no-scm -f
export PYTHONPATH="$CODEDIR"

eval "$( python -m dryads-research.experiments.utilities.data_dirs \
	$cohort --expr_source $expr_source )"

# enumerate the mutation types that will be tested in this experiment
dvc run -d $COH_DIR -d $GENCODE_DIR -d $ONCOGENE_LIST \
	-d $SUBTYPE_LIST -d $RUNDIR/setup_test.py \
	-d $CODEDIR/dryads-research/environment.yml \
	-o setup/muts-list.p -m setup/muts-count.txt \
	-f setup.dvc --overwrite-dvcfile \
	python -m dryads-research.experiments.subgrouping_test.setup_test \
	$expr_source $cohort $samp_cutoff $mut_levels $OUTDIR

# how long is this pipeline allowed to run for?
if [ -z ${SBATCH_TIMELIMIT+x} ]
then
	time_lim=2159
else
	time_lim=$SBATCH_TIMELIMIT
fi

# figure out how long it took to run the setup stages and how much time is
# left to run the remainder of the pipeline
cur_time=$( date +%s )
time_left=$(( time_lim - (cur_time - start_time) / 60 + 1 ))

if [ -z ${time_max+x} ]
then
	time_max=$(( $time_left * 11 / 13 ))
fi

# figure out the time left for the consolidation stage of the pipeline
if [ ! -f setup/tasks.txt ]
then
	merge_max=$(( $time_left - $time_max - 3 ))

  # based on the classifier used, set parameters for calculating the runtime
  # of a single classification task using the sample size of the cohort
	if [ $classif == 'Ridge' ]
	then
		task_size=1
		samp_exp=0.5
	elif [ $classif == 'RidgeMoreTune' ]
	then
		task_size=3.7
		samp_exp=0.5

	elif [ $classif == 'SVCrbf' ]
	then
		task_size=5
		samp_exp=1.13
	elif [ $classif == 'Forests' ]
	then
		task_size=11
		samp_exp=0.75
	fi

  # calculate the runtime of a single classification task
	eval "$( python -m dryads-research.experiments.utilities.pipeline_setup \
		$OUTDIR $time_max --merge_max=$merge_max \
		--task_size=$task_size --samp_exp=$samp_exp )"
fi

# if we are only enumerating, we quit before classification jobs are launched
if $count_only
then
	cp setup/cohort-data.p.gz $FINALDIR/cohort-data__${out_tag}.p.gz
	exit 0
fi

# get the `run_time` and `merge_time` parameters from the task manifest
eval "$( tail -n 2 setup/tasks.txt | head -n 1 )"
eval "$( tail -n 1 setup/tasks.txt )"

# launch the Snakemake pipeline for the classification and output
# consolidation stages of this experiment
dvc run -d setup/muts-list.p -d $RUNDIR/fit_test.py -O out-conf.p.gz \
	-O $FINALDIR/out-trnsf__${mut_levels}__${classif}.p.gz -f output.dvc \
	--overwrite-dvcfile --ignore-build-cache 'snakemake -s $RUNDIR/Snakefile \
	-j 800 --latency-wait 120 --cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} \
	-t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config expr_source='"$expr_source"' cohort='"$cohort"' \
	samp_cutoff='"$samp_cutoff"' mut_levels='"$mut_levels"' \
	classif='"$classif"' time_max='"$run_time"' merge_max='"$merge_time"

# final cleanup duties
rm $OUTDIR/setup/cohort-data__*
cp output.dvc $FINALDIR/output__${mut_levels}__${classif}.dvc

