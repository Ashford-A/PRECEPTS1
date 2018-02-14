
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.mut import VariantCohort as FirehoseVarCohort
from HetMan.features.cohorts.icgc import VariantCohort as ICGCVarCohort

import synapseclient
import dill as pickle

icgc_data_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'
firehose_data_dir = ('/home/exacloud/lustre1/share_your_data_here'
                     '/precepts/firehose/')


def main():
    cdata_icgc = ICGCVarCohort(icgc_data_dir, 'PACA-AU', cv_prop=1.0,
                               samp_cutoff=[1/12, 11/12])
    use_gns = [gn for gn, mut in cdata_icgc.train_mut]

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/'
                                'share_your_data_here/precepts/synapse')
    syn.login()

    use_cohorts = set(os.listdir(
        os.path.join(firehose_data_dir, 'stddata__2016_01_28')))
    use_cohorts -= {'FPPP'}

    cohort_genes = []
    for cohort in use_cohorts:

        cdata = FirehoseVarCohort(
            cohort=cohort, mut_genes=use_gns, mut_levels=['Gene'],
            expr_source='Firehose', data_dir=firehose_data_dir,
            cv_prop=1.0, syn=syn
            )

        for gn, mut in cdata.train_mut:
            if 10 <= len(mut) <= (len(cdata.samples) - 10):
                print('{} and {}'.format(cohort, gn))
                cohort_genes += [[cohort, gn]]
        
    pickle.dump(sorted(cohort_genes),
                open(os.path.join(base_dir, 'cohort_genes.p'), 'wb'))


if __name__ == "__main__":
    main()

