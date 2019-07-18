#!/bin/bash
#SBATCH --job-name=subv-infer
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts t:s:g:l:c:m:r var
do
	case "$var" in
		t)	cohort=$OPTARG;;
		s)	samp_cutoff=$OPTARG;;
		g)	gene=$OPTARG;;
		l)	mut_levels=$OPTARG;;
		c)	classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)      rewrite=true;;
		[?])    echo "Usage: $0 [-t] TCGA cohort [-s] minimum sample cutoff " \
			     "[-g] mutated gene [-l] mutation annotation levels " \
			     "[-c] mutation classifier [-m] maximum tests per node" \
			     "[-r] whether existing results should be rewritten"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/HetMan/subvariant_infer/${cohort}__samps-${samp_cutoff}/${gene}__${mut_levels}/$classif
FINALDIR=$DATADIR/HetMan/subvariant_infer/${cohort}__samps-${samp_cutoff}/${gene}
export RUNDIR=$CODEDIR/HetMan/experiments/subvariant_infer
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create the directories where intermediate and final output will be stored, move to working directory
mkdir -p $FINALDIR
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

# initiate version control in this directory if it hasn't been already
if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

# enumerate the mutation types that will be tested in this experiment
dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $subtype_file \
	-d $RUNDIR/setup_infer.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/muts-list.p -m setup/muts-count.txt \
	-f setup.dvc --overwrite-dvcfile python $RUNDIR/setup_infer.py \
	$cohort $gene $mut_levels --samp_cutoff=$samp_cutoff --setup_dir=$OUTDIR

# calculate how many parallel tasks the mutations will be tested over
muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))

# remove the Snakemake locks on the working directory if present
if [ -d .snakemake ]
then
	snakemake --unlock
fi

dvc run -d setup/cohort-data.p -d setup/muts-list.p -d $RUNDIR/fit_infer.py \
	-o $FINALDIR/out-data__${mut_levels}__${classif}.p.gz -f output.dvc \
	--overwrite-dvcfile 'snakemake -s $RUNDIR/Snakefile -j 50 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster "sbatch -p {cluster.partition} \
	-J {cluster.job-name} -t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} --mem-per-cpu {cluster.mem-per-cpu} \
	--exclude=$ex_nodes --no-requeue" --config cohort='"$cohort"' \
	samp_cutoff='"$samp_cutoff"' gene='"$gene"' mut_levels='"$mut_levels"' \
	classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${mut_levels}__${classif}.dvc

