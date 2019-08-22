#!/bin/bash
#SBATCH --job-name=dyad-tour
#SBATCH --verbose


source activate HetMan
rewrite=false

# collect command line arguments
while getopts e:t:s:l:c:m:r var
do
	case "$var" in
		e)      expr_source=$OPTARG;;
		t)	cohort=$OPTARG;;
		s)	samp_cutoff=$OPTARG;;
		l)      mut_levels=$OPTARG;;
		c)	classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)	rewrite=true;;
		[?])	echo "Usage: $0 [-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation classifier [-m] maximum tests per node";
			exit 1;;
	esac
done

# decide where intermediate files will be stored, find code source directory and input files
out_tag=${cohort}__samps-${samp_cutoff}
OUTDIR=$TEMPDIR/HetMan/dyad_tour/$expr_source/$out_tag/$mut_levels/$classif
FINALDIR=$DATADIR/HetMan/dyad_tour/${expr_source}__$out_tag

export RUNDIR=$CODEDIR/HetMan/experiments/dyad_tour
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

if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $gene_file -d $subtype_file \
	-d $RUNDIR/setup_tour.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/pairs-list.p -m setup/pairs-count.txt \
	-f setup.dvc --overwrite-dvcfile python $RUNDIR/setup_tour.py \
	$expr_source $cohort $samp_cutoff $mut_levels $OUTDIR

pairs_count=$(cat setup/pairs-count.txt)
task_count=$(( $(( $pairs_count - 1 )) / $test_max + 1 ))

if [ -d .snakemake ]
then
	snakemake --unlock
	rm -rf .snakemake/locks/*
fi

dvc run -d setup/cohort-data.p -d setup/pairs-list.p -d $RUNDIR/fit_tour.py \
	-o $FINALDIR/out-data__${mut_levels}__${classif}.p.gz -f output.dvc \
	--overwrite-dvcfile --remove-outs --no-commit \
	'snakemake -s $RUNDIR/Snakefile -j 100 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json --cluster "sbatch -p {cluster.partition} \
	-J {cluster.job-name} -t {cluster.time} -o {cluster.output} -e {cluster.error} \
	-n {cluster.ntasks} -c {cluster.cpus-per-task} --mem-per-cpu {cluster.mem-per-cpu} \
	--exclude=$ex_nodes --no-requeue" --config expr_source='"$expr_source"' \
	cohort='"$cohort"' samp_cutoff='"$samp_cutoff"' mut_levels='"$mut_levels"' \
	classif='"$classif"' task_count='"$task_count"

cp output.dvc $FINALDIR/output__${mut_levels}__${classif}.dvc

