#!/bin/bash

#SBATCH --job-name=subv-infer
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=8000
#SBATCH --time=2150


source activate HetMan
rewrite=false

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
			     "[-c] mutation classifier [-m] maximum tests per node";
			exit 1;;
	esac
done

OUTDIR=$TEMPDIR/HetMan/subvariant_infer/${cohort}__samps-${samp_cutoff}/${gene}__${mut_levels}/$classif
export RUNDIR=$CODEDIR/HetMan/experiments/subvariant_infer
source $RUNDIR/files.sh

rmv_str=""
if $rewrite
then
	rm -rf $OUTDIR
else
	rmv_str="--remove-outs "
fi

mkdir -p $DATADIR/HetMan/subvariant_infer/${cohort}__samps-${samp_cutoff}/${gene}
mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR

if [ ! -d .dvc ]
then
	dvc init --no-scm
fi

dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $subtype_file \
	-d $RUNDIR/setup_infer.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-data.p -o setup/muts-list.p -m setup/muts-count.txt \
	-f setup.dvc --overwrite-dvcfile python $RUNDIR/setup_infer.py \
	$cohort $gene $mut_levels --samp_cutoff=$samp_cutoff --setup_dir=$OUTDIR

muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))

if [ $task_count -gt 50 ]
then
	task_count=50
fi

if [ -d .snakemake ]
then
	snakemake --unlock
fi

dvc run -d setup/cohort-data.p -d setup/muts-list.p -d $RUNDIR/fit_infer.py \
	-d $CODEDIR/HetMan/experiments/utilities/classifiers.py -o out-data.p -f output.dvc \
	'snakemake -s $RUNDIR/Snakefile -j 50 --latency-wait 120 \
	--cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} -t {cluster.time} \
	-o {cluster.output} -e {cluster.error} -n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config cohort='"$cohort"' samp_cutoff='"$samp_cutoff"' gene='"$gene"' \
	mut_levels='"$mut_levels"' classif='"$classif"' task_count='"$task_count"

