#!/bin/bash
#SBATCH --job-name=subv-thresh
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts t:g:c:m:r var
do
	case "$var" in
		t)	cohort=$OPTARG;;
		c)	classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)      rewrite=true;;
		[?])    echo "Usage: $0 [-t] tumour cohort " \
				"[-c] mutation classifier [-m] maximum tests per node" \
				"[-r] rewrite existing results?"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/HetMan/subvariant_threshold/$cohort/$classif
FINALDIR=$DATADIR/HetMan/subvariant_threshold
export RUNDIR=$CODEDIR/HetMan/experiments/subvariant_threshold
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored, move to working directory
mkdir -p $FINALDIR $TEMPDIR/HetMan/subvariant_threshold/setup
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

# initiate version control in this directory if it hasn't been already
if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

# enumerate the mutation types that will be tested in this experiment
dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $subtype_file \
	-d $RUNDIR/setup_threshold.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/muts-list.p -o setup/feat-list.p \
	-m setup/muts-count.txt -f setup.dvc --overwrite-dvcfile \
	python $RUNDIR/setup_threshold.py $cohort $classif \
	$OUTDIR $DATADIR/HetMan

# calculate how many parallel tasks the mutations will be tested over
muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))

# remove the Snakemake locks on the working directory if present
if [ -d .snakemake ]
then
	snakemake --unlock
	rm -rf .snakemake/locks/*
fi

dvc run -d setup/cohort-data.p -d setup/muts-list.p -d setup/feat-list.p \
	-d $RUNDIR/fit_threshold.py -o $FINALDIR/out-data__${cohort}__${classif}.p.gz \
	-f output.dvc --overwrite-dvcfile --no-commit \
	'snakemake -s $RUNDIR/Snakefile -j 120 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster "sbatch -p {cluster.partition} \
	-J {cluster.job-name} -t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} --mem-per-cpu {cluster.mem-per-cpu} \
	--exclude=$ex_nodes --no-requeue" --config cohort='"$cohort"' \
	classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${cohort}__${classif}.dvc

