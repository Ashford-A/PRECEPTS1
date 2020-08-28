#!/bin/bash
#SBATCH --job-name=subv-iso
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts t:g:l:s:c:m:r var
do
	case "$var" in
		t)	cohort=$OPTARG;;
		g)	gene=$OPTARG;;
		l)	mut_levels=$OPTARG;;
		s)	search=$OPTARG;;
		c)	classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)      rewrite=true;;
		[?])    echo "Usage: $0 [-t] tumour cohort [-g] mutated gene" \
				"[-c] mutation classifier [-m] maximum tests per node" \
				"[-r] rewrite existing results?"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/HetMan/subvariant_isolate/$gene/$cohort/$mut_levels/$search/$classif
FINALDIR=$DATADIR/HetMan/subvariant_isolate/$gene
export RUNDIR=$CODEDIR/HetMan/experiments/subvariant_isolate
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored, move to working directory
mkdir -p $FINALDIR $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
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
	$gene $cohort $search $mut_levels $OUTDIR

# calculate how many parallel tasks the mutations will be tested over
muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))
out_tag=${cohort}__${mut_levels}__${search}__${classif}

# remove the Snakemake locks on the working directory if present
if [ -d .snakemake ]
then
	snakemake --unlock
	rm -rf .snakemake/locks/*
fi

dvc run -d setup/muts-list.p -d $RUNDIR/fit_isolate.py \
	-o $FINALDIR/out-siml__${out_tag}.p.gz -f output.dvc --overwrite-dvcfile \
	--no-commit 'snakemake -s $RUNDIR/Snakefile -j 200 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} \
	-t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config cohort='"$cohort"' gene='"$gene"' mut_levels='"$mut_levels"' \
	search='"$search"' classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${out_tag}.dvc

