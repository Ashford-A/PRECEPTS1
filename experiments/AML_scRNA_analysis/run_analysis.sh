#!/bin/bash
#SBATCH --job-name=AML_scRNA
#SBATCH --verbose

# Example usage:
#	export SBATCH_TIMELIMIT=500;
#	sbatch --mem-per-cpu=8000 --account='compbio' -c 4 --exclude=$ex_nodes \
#	--output=$slurm_dir/AML-scRNA.out --error=$slurm_dir/AML-scRNA.err \
#	dryads-research/experiments/AML_scRNA_analysis/run_analysis.sh \
#	-s default -l default -c Ridge -m 250 -r


start_time=$( date +%s )
source activate research
rewrite=false
count_only=false

# collect command line arguments
while getopts :s:l:c:m:rn var
do
	case "$var" in
		s)  search_params=$OPTARG;;
		l)  mut_lvls=$OPTARG;;
		c)  classif=$OPTARG;;
		m)  test_max=$OPTARG;;
		r)  rewrite=true;;
		n)  count_only=true;;
		[?])  echo "Usage: $0 " \
				"[-s] subgrouping search parameters" \
				"[-l] mutation annotation levels" \
				"[-c] mutation classifier" \
				"[-m] maximum number of tests per node" \
				"[-r] rewrite existing results?" \
				"[-n] only enumerate, don't classify?"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory
param_tag=${search_params}__${mut_lvls}
OUTDIR=$TEMPDIR/dryads-research/AML_scRNA_analysis/$param_tag/$classif
FINALDIR=$DATADIR/dryads-research/AML_scRNA_analysis/$classif
export RUNDIR=$CODEDIR/dryads-research/experiments/AML_scRNA_analysis

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored, move to working directory
mkdir -p $FINALDIR $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm $OUTDIR/merge
cd $OUTDIR || exit

# clean out last snakemake run; initialize DVC; point Python to where code is
rm -rf .snakemake
dvc init --no-scm -f
export PYTHONPATH="$CODEDIR"

# get input data files
eval "$( python -m dryads-research.experiments.utilities.data_dirs beatAML )"

# enumerate subgroupings to be tested
if $rewrite
then
	# enumerate the mutation types that will be tested in this experiment
	dvc run -d $COH_DIR -d $GENCODE_DIR -d $RUNDIR/setup_analysis.py \
		-d $CODEDIR/dryads-research/environment.yml \
		-o setup/muts-list.p -m setup/muts-count.txt \
		-f setup.dvc --overwrite-dvcfile \
		python -m dryads-research.experiments.AML_scRNA_analysis.setup_analysis \
		$search_params $mut_lvls $OUTDIR
fi

# what is the time limit for this job?
if [ -z ${SBATCH_TIMELIMIT+x} ]
then
	time_lim=2159
else
	time_lim=$SBATCH_TIMELIMIT
fi

# how much time do we have left to finish this job?
cur_time=$( date +%s )
time_left=$(( time_lim - (cur_time - start_time) / 60 + 1 ))

# allocate time for running classification tasks
if [ -z ${time_max+x} ]
then
	time_max=$(( $time_left * 11 / 13 ))
fi

# figure out how to distribute the classification and consolidation tasks across
# compute nodes in order to finish within the time left
if [ ! -f setup/tasks.txt ]
then
	merge_max=$(( $time_left - $time_max - 3 ))

	eval "$( python -m dryads-research.experiments.utilities.pipeline_setup \
		$OUTDIR $time_max --merge_max=$merge_max \
		--task_size=3 --merge_size=3 --samp_exp=0.43 )"
fi

# if we are only enumerating, quit before classification jobs are launched
if $count_only
then
	cp setup/cohort-data.p.gz $FINALDIR/cohort-data__${param_tag}.p.gz
	exit 0
fi

eval "$( tail -n 2 setup/tasks.txt | head -n 1 )"
eval "$( tail -n 1 setup/tasks.txt )"

# run the classification and output consolidation tasks
dvc run -d setup/muts-list.p -d $RUNDIR/fit_analysis.py -O out-conf.p.gz \
	-O $FINALDIR/out-conf__${param_tag}.p.gz -f output.dvc \
	--overwrite-dvcfile --ignore-build-cache 'snakemake -s $RUNDIR/Snakefile \
	-j 400 --latency-wait 120 --cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} \
	-t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config search='"$search_params"' mut_levels='"$mut_lvls"' \
	classif='"$classif"' time_max='"$run_time"' merge_max='"$merge_time"

cp output.dvc $FINALDIR/output__${param_tag}.dvc

