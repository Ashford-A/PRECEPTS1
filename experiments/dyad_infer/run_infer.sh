#!/bin/bash
#SBATCH --job-name=dyad-infer
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts t:s:c:m:r var
do
	case "$var" in
		t)	export cohort=$OPTARG;;
		s)	export samp_cutoff=$OPTARG;;
		c)	export classif=$OPTARG;;
		m)	export test_max=$OPTARG;;
		r)	rewrite=true;;
		[?])	echo "Usage: $0 [-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation classifier [-m] maximum tests per node" \
			     "[-r] whether existing results should be rewritten"
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
OUTDIR=$TEMPDIR/HetMan/dyad_infer/${cohort}__samps-${samp_cutoff}/$classif
FINALDIR=$DATADIR/HetMan/dyad_infer/${cohort}__samps-${samp_cutoff}
export RUNDIR=$CODEDIR/HetMan/experiments/dyad_infer
source $RUNDIR/files.sh

# if we want to rewrite the experiment, remove the intermediate output directory
if $rewrite
then
	rm -rf $OUTDIR
fi

# create intermediate and final output directories, move to working directory
mkdir -p $FINALDIR
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

# initiate version control in this directory if it hasn't been already
if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

# enumerate the pairs of mutations that will be tested in this experiment
dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $gene_file -d $subtype_file \
	-d $RUNDIR/setup_infer.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/pairs-list.p -m setup/pairs-count.txt \
	-f setup.dvc --overwrite-dvcfile python $RUNDIR/setup_infer.py $cohort \
	$samp_cutoff --setup_dir=$OUTDIR

# calculate how many parallel tasks the pairs of mutations will tested over
pairs_count=$(cat setup/pairs-count.txt)
task_count=$(( $(( $pairs_count - 1 )) / $test_max + 1 ))

# remove the Snakemake locks on the working directory if present
if [ -d .snakemake ]
then
	snakemake --unlock
fi

dvc run -d setup/cohort-data.p -d setup/pairs-list.p -d $RUNDIR/fit_infer.py \
	-o $FINALDIR/out-data__${classif}.p.gz -f output.dvc --overwrite-dvcfile \
	--remove-outs --no-commit 'snakemake -s $RUNDIR/Snakefile -j 100 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster "sbatch -p {cluster.partition} \
	-J {cluster.job-name} -t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} --mem-per-cpu {cluster.mem-per-cpu} \
	--exclude=$ex_nodes --no-requeue" --config cohort='"$cohort"' \
	samp_cutoff='"$samp_cutoff"' classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${classif}.dvc

