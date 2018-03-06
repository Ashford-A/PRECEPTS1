
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort as TCGAcohort
from HetMan.features.cohorts.icgc import MutationCohort as ICGCcohort

import synapseclient
import glob
import dill as pickle

toil_dir = "/home/exacloud/lustre1/CompBio/mgrzad/input-data/toil/processed"
icgc_data_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'


def main():
    cohort_genes = []

    # loads PACA-AU ICGC cohort, finds the frequently mutated genes
    cdata_icgc = ICGCcohort('PACA-AU', icgc_data_dir, mut_genes=None,
                            cv_prop=1.0, samp_cutoff=[1/12, 11/12])
    use_gns = [gn for gn, mut in cdata_icgc.train_mut]

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/'
                                'share_your_data_here/precepts/synapse')
    syn.login()

    # finds all the cohorts analyzed through Toil that have been downloaded
    use_cohorts = [os.path.basename(fl).replace('.txt.gz', '')
                   for fl in glob.glob(os.path.join(toil_dir, '*.txt.gz'))]

    # loads each TCGA Toil cohort, finds all the genes that are frequently
    # mutated both in it and the PACA-AU cohort
    for cohort in use_cohorts:
        cdata_tcga = TCGAcohort(
            cohort=cohort, mut_genes=use_gns, mut_levels=['Gene'],
            expr_source='toil', expr_dir=toil_dir, var_source='mc3',
            collapse_txs=False, syn=syn, cv_prop=1.0
            )

        for gn, mut in cdata_tcga.train_mut:
            if 10 <= len(mut) <= (len(cdata_tcga.samples) - 10):
                print('{} and {}'.format(cohort, gn))
                cohort_genes += [[cohort, gn]]

    # save list of genes found for each TCGA cohort for which we can
    # potentially transfer signatures to PACA-AU
    pickle.dump(sorted(cohort_genes),
                open(os.path.join(base_dir, 'setup', 'cohort_genes.p'), 'wb'))


if __name__ == "__main__":
    main()

