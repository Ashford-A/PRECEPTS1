#!/bin/bash

#SBATCH --job-name=subv-cross
#SBATCH --partition=exacloud
#SBATCH --mem=1000
#SBATCH --time=500

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-cross_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-cross_%j.err
#SBATCH --verbose


# move to working directory, load required packages and modules
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
source activate precepts

# check for environment variables controlling the experiment, if these
# variables are not defined assign them default values
if [ -z ${cohort+x} ]
then
	echo "no cohort given, defaulting to TCGA-BRCA"
	export cohort="BRCA"
fi

if [ -z ${gene+x} ]
then
	echo "no gene given, defaulting to TP53"
	export gene="TP53"
fi

if [ -z ${classif+x} ]
then
	echo "no classifier given, defaulting to Lasso"
	export classif="Lasso"
fi

if [ -z ${samp_cutoff+x} ]
then
	echo "no mimimum sample size cutoff given, defaulting to twenty-five"
	export samp_cutoff=25
fi

if [ -z ${mut_levels+x} ]
then
	echo "no mutation levels given, defaulting to Form_base+Exon"
	export mut_levels="Form_base__Exon"
fi

if [ -z ${test_max+x} ]
then
	echo "limiting maximum number of tests per node to fifty"
	export test_max=25
fi 

export BASEDIR=HetMan/experiments/subvariant_cross
mkdir -p $BASEDIR/setup/${cohort}/${gene}

export OUTDIR=$BASEDIR/output/$cohort/$gene/$classif/samps_${samp_cutoff}/$mut_levels
rm -rf $OUTDIR
mkdir -p $OUTDIR/slurm

if [ ! -e ${BASEDIR}/setup/${cohort}/${gene}/pairs_list__samps_${samp_cutoff}__levels_${mut_levels}.p ]
then

	srun -p=exacloud \
		--output=$BASEDIR/setup/slurm_${cohort}.txt \
		--error=$BASEDIR/setup/slurm_${cohort}.err \
		python $BASEDIR/setup_cross.py -v \
		$cohort $gene $mut_levels --samp_cutoff=$samp_cutoff
fi

# finds how large of a batch array to submit based on how many mutation types were
# found in the setup enumeration
pairs_count=$(cat ${BASEDIR}/setup/${cohort}/${gene}/pairs_count__samps_${samp_cutoff}__levels_${mut_levels}.txt)
export array_size=$(( $pairs_count / $test_max ))

if [ $array_size -gt 299 ]
then
	export array_size=299
fi

sbatch --array=0-$(( $array_size )) $BASEDIR/fit_cross.sh

