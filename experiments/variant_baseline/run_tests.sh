#!/bin/bash

#SBATCH --job-name=var-baseline
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=8000
#SBATCH --time=2150


source activate HetMan
rewrite=false

while getopts e:t:s:c:m:r var
do
	case "$var" in
		e)	expr_source=$OPTARG;;
		t)	cohort=$OPTARG;;
		s)	samp_cutoff=$OPTARG;;
		c)	classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)	rewrite=true;;
		[?])	echo "Usage: $0 [-e] expression source directory" \
			     "[-t] TCGA cohort [-s] minimum sample cutoff" \
			     "[-c] mutation classifier [-m] maximum tests per node";
			exit 1;;
	esac
done

OUTDIR=$TEMPDIR/HetMan/variant_baseline/$expr_source/${cohort}__samps-${samp_cutoff}/$classif
export RUNDIR=$CODEDIR/HetMan/experiments/variant_baseline
source $RUNDIR/files.sh
out_tag=${expr_source}__${cohort}__samps-${samp_cutoff}

rmv_str=""
if $rewrite
then
	rm -rf $OUTDIR
else
	rmv_str="--remove-outs "
fi

mkdir -p $OUTDIR/setup $OUTDIR/output $OUTDIR/slurm
cd $OUTDIR
dvc init --no-scm
mkdir -p $DATADIR/HetMan/variant_baseline/$out_tag

dvc run -d $firehose_dir -d $mc3_file -d $gencode_file -d $gene_file -d $subtype_file \
	-d $RUNDIR/setup_tests.py -o setup/cohort-data_${out_tag}.p \
	-o setup/vars-list_${out_tag}.p -m setup/vars-count_${out_tag}.txt \
	-f setup.dvc --overwrite-dvcfile \
	python $RUNDIR/setup_tests.py $expr_source $cohort $samp_cutoff --setup_dir $OUTDIR

vars_count=$(cat setup/vars-count_${out_tag}.txt)
task_count=$(( $vars_count / $test_max + 1 ))

if [ $task_count -gt 12 ]
then
	task_count=12
fi

dvc run -d setup/cohort-data_${out_tag}.p -d setup/vars-list_${out_tag}.p \
       	-d $RUNDIR/fit_tests.py -d $RUNDIR/models/${classif%%'__'*}.py \
	-o out-data.p -f output.dvc \
	'snakemake -s $RUNDIR/Snakefile -j 100 --latency-wait 120 \
	--rerun-incomplete --cluster-config $RUNDIR/cluster.json \
	--cluster "sbatch -p {cluster.partition} -J {cluster.job-name} -t {cluster.time} \
	-o {cluster.output} -e {cluster.error} -n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config expr_source='"$expr_source"' cohort='"$cohort"' samp_cutoff='"$samp_cutoff"' \
	classif='"$classif"' task_count='"$task_count"

