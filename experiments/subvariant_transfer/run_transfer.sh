#!/bin/bash

#SBATCH --job-name=subv-trans
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=8000
#SBATCH --time=400


source activate HetMan
rewrite=false

# collect command line arguments
cohort_list=()
while getopts t:s:g:l:c:x:m:r var
do
	case "$var" in
		t)	cohort_list+=($OPTARG);;
		s)	samp_cutoff=$OPTARG;;
		l)	mut_levels=$OPTARG;;
		c)	classif=$OPTARG;;
		x)	ex_mtype=$OPTARG;;
		m)	test_max=$OPTARG;;
		r)      rewrite=true;;
		[?])    echo "Usage: $0 [-t] TCGA cohort [-s] minimum sample cutoff " \
			     "[-l] mutation annotation levels [-c] mutation classifier" \
			     "[-m] maximum tests per node";
			exit 1;;
	esac
done

IFS=$'\n'
export sorted_cohs=($(sort <<<"${cohort_list[*]}"))
unset IFS

coh_lbl=$( printf "__"%s "${sorted_cohs[@]}" )
export coh_lbl=${coh_lbl#__}

OUTDIR=$TEMPDIR/HetMan/subvariant_transfer/${coh_lbl}__samps-${samp_cutoff}/$mut_levels/${classif}_${ex_mtype}
FINALDIR=$DATADIR/HetMan/subvariant_transfer/${coh_lbl}__samps-${samp_cutoff}
export RUNDIR=$CODEDIR/HetMan/experiments/subvariant_transfer
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
	-d $RUNDIR/setup_transfer.py -d $CODEDIR/HetMan/environment.yml \
	-o setup/cohort-dict.p -o setup/cohort-data.p -o setup/muts-list.p \
	-m setup/muts-count.txt -f setup.dvc --overwrite-dvcfile \
	python $RUNDIR/setup_transfer.py $mut_levels $ex_mtype "${sorted_cohs[@]}" \
	--samp_cutoff=$samp_cutoff --setup_dir=$OUTDIR

muts_count=$(cat setup/muts-count.txt)
task_count=$(( $(( $muts_count - 1 )) / $test_max + 1 ))

if [ $task_count -gt 100 ]
then
	task_count=100
fi

if [ -d .snakemake ]
then
	snakemake --unlock
fi

dvc run -d setup/cohort-data.p -d setup/muts-list.p -d $RUNDIR/fit_transfer.py \
	-d $CODEDIR/HetMan/experiments/utilities/classifiers.py \
	-o $FINALDIR/out-data__${mut_levels}_${classif}_${ex_mtype}.p -f output.dvc \
	--overwrite-dvcfile --remove-outs 'snakemake -s $RUNDIR/Snakefile -j 50 \
	--latency-wait 120 --cluster-config $RUNDIR/cluster.json --cluster \
	"sbatch -p {cluster.partition} -J {cluster.job-name} -t {cluster.time} \
	-o {cluster.output} -e {cluster.error} -n {cluster.ntasks} -c {cluster.cpus-per-task} \
	--mem-per-cpu {cluster.mem-per-cpu} --exclude=$ex_nodes --no-requeue" \
	--config cohorts='"$coh_lbl"' samp_cutoff='"$samp_cutoff"' mut_levels='"$mut_levels"' \
	classif='"$classif"' ex_mtype='"$ex_mtype"' task_count='"$task_count"

