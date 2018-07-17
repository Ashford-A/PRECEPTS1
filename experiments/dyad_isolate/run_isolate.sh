#!/bin/bash

#SBATCH --job-name=dyad-isolate
#SBATCH --partition=exacloud
#SBATCH --verbose

#SBATCH --mem=1000
#SBATCH --time=500


# move to working directory, load required packages and modules
export BASEDIR=HetMan/experiments/dyad_isolate
source activate precepts

while getopts t:c:s:m: var
do
	case "$var" in
		t)	export cohort=$OPTARG;;
		c)	export classif=$OPTARG;;
		s)	export samp_cutoff=$OPTARG;;
		m)	export test_max=$OPTARG;;
		[?])    echo "Usage: $0 [-t] TCGA cohort [-c] mutation classifier" \
			     "[-s] minimum sample cutoff [-m] maximum tests per node";
			exit 1;;
	esac
done

# get the directory where the experiment results will be saved, removing it
# if it already exists
export OUTDIR=$BASEDIR/output/$cohort/$classif/samps_${samp_cutoff}
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

# setup the experiment by finding a list of mutation subtypes to be tested
if [ ! -e ${BASEDIR}/setup/${cohort}/pairs_list__samps_${samp_cutoff}.p ]
then

	srun -p=exacloud \
		--output=$BASEDIR/setup/slurm_${cohort}.txt \
		--error=$BASEDIR/setup/slurm_${cohort}.err \
		python $BASEDIR/setup_isolate.py -v $cohort --samp_cutoff=$samp_cutoff
fi

# find how large of a batch array to submit based on how many mutation types were
# found in the setup enumeration
mtypes_count=$(cat ${BASEDIR}/setup/${cohort}/pairs_count__samps_${samp_cutoff}.txt)
export array_size=$(( $mtypes_count / $test_max ))

if [ $array_size -gt 299 ]
then
	export array_size=299
fi

# run the subtype tests in parallel
sbatch --output=${slurm_dir}/dyad-iso.out \
	--error=${slurm_dir}/dyad-iso.err \
	--array=0-$(( $array_size )) $BASEDIR/fit_isolate.sh

