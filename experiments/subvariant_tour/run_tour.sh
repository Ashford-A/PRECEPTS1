#!/bin/bash

#SBATCH --job-name=subv-tour
#SBATCH --partition=exacloud
#SBATCH --mem=8000
#SBATCH --time=1000

#SBATCH --output=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-tour_%j.out
#SBATCH --error=/home/exacloud/lustre1/CompBio/mgrzad/slurm/log-files/subv-tour_%j.err
#SBATCH --verbose


# move to working directory, load required packages and modules
cd /home/exacloud/lustre1/CompBio/mgrzad/bergamot
source activate precepts

# gets the name of the TCGA cohort to use
if [ -z ${cohort+x} ]
then
	echo "no cohort given, defaulting to TCGA-BRCA"
	export cohort="BRCA"
fi

# gets the name of the mutation prediction pipeline to use
if [ -z ${classif+x} ]
then
	echo "no classifier given, defaulting to Lasso"
	export classif="Lasso"
fi

# gets the minimum frequency of subtypes to consider
if [ -z ${freq_cutoff+x} ]
then
	echo "no mutation frequency cutoff given, defaulting to two percent"
	export freq_cutoff="0.02"
fi

# gets the mutation annotation levels to search over
if [ -z ${mut_levels+x} ]
then
	echo "no mutation levels given, defaulting to Form_base+Protein"
	export mut_levels="Form_base__Protein"
fi

export BASEDIR=HetMan/experiments/subvariant_tour
mkdir -p $BASEDIR/setup

export OUTDIR=$BASEDIR/output/$cohort/$classif/freq_${freq_cutoff}/$mut_levels
mkdir -p $OUTDIR/slurm

if [ ! -e ${BASEDIR}/setup/${cohort}/mtype_list__freq_${freq_cutoff}__levels_${mut_levels}.p ]
then

	srun -p=exacloud \
		--output=$BASEDIR/setup/slurm_${cohort}.txt \
		--error=$BASEDIR/setup/slurm_${cohort}.err \
		python $BASEDIR/setup_tour.py $cohort -v \
		--freq_cutoff=$freq_cutoff --mut_levels=$mut_levels
fi

# finds how large of a batch array to submit based on how many mutation types were
# found in the setup enumeration
mtype_count=$(cat ${BASEDIR}/setup/${cohort}/mtype_count__freq_${freq_cutoff}__levels_${mut_levels}.txt)
export array_size=$(( ($mtype_count / 50 + 1) * 5 - 1 ))

if [ $array_size -gt 199 ]
then
	export array_size=199
fi

sbatch --array=0-$(( $array_size )) $BASEDIR/fit_tour.sh

