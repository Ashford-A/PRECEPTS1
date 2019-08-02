#!/bin/bash

#SBATCH --job-name=cna-isolate
#SBATCH --partition=exacloud
#SBATCH --mem=1000
#SBATCH --time=500

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/cna-isolate_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/cna-isolate_%j.err
#SBATCH --verbose


# move to working directory, load required packages and modules
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
source activate precepts

while getopts t:g:c:m: var
do
	case "$var" in
		t)	export cohort=$OPTARG;;
		g)	export gene=$OPTARG;;
		c)	export classif=$OPTARG;;
		m)	test_max=$OPTARG;;
		[?])	echo "Usage: $0 [-t] TCGA cohort [-g] mutated gene" \
			     "[-c] classifier [-m] maximum tests per node";
			exit 1;;
	esac
done

export BASEDIR=HetMan/experiments/cna_isolate
export OUTDIR=$BASEDIR/output/$cohort/$gene/$classif
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

if [ ! -e ${BASEDIR}/setup/ctf_lists/${cohort}_${gene}.p ]
then

	srun -p=exacloud \
		--output=$BASEDIR/setup/slurm_${cohort}.txt \
		--error=$BASEDIR/setup/slurm_${cohort}.err \
		python $BASEDIR/setup_isolate.py -v $cohort $gene
fi

mtypes_count=$(cat ${BASEDIR}/setup/ctf_counts/${cohort}_${gene}.txt)
export array_size=$(( $mtypes_count / $test_max ))

if [ $array_size -gt 199 ]
then
	export array_size=199
fi

# run the subtype tests in parallel
sbatch --array=0-$(( $array_size )) $BASEDIR/fit_isolate.sh

