
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
import numpy as np
from itertools import combinations as combn

import argparse
import synapseclient
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"
copy_dir = '/home/users/grzadkow/compbio/input-data/firehose'


def main():
    parser = argparse.ArgumentParser(
        "Set up the copy number alteration expression effect isolation "
        "experiment by enumerating alteration score thresholds to be tested."
        )

    # create command line arguments
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which gene to consider")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command line arguments, create directory where found thresholds
    # and threshold counts will be stored
    args = parser.parse_args()
    os.makedirs(os.path.join(base_dir, 'setup', 'ctf_lists'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'setup', 'ctf_counts'), exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # load expression, variant call, and copy number alteration data for
    # the given TCGA cohort and mutated gene
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', var_source='mc3', expr_dir=firehose_dir,
        copy_source='Firehose', copy_dir=copy_dir, copy_discrete=False,
        cv_prop=1.0, syn=syn
        )

    ctf_list = []
    mut_stat = np.array(cdata.train_mut.status(cdata.copy_data.index))
    mut_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', args.gene): None})))

    copy_vals = cdata.copy_data.loc[~mut_stat, args.gene]
    loss_vals = copy_vals[copy_vals < 0]
    gain_vals = copy_vals[copy_vals > 0]

    loss_step = 20 / len(loss_vals)
    loss_ctfs = np.unique(loss_vals.quantile(np.arange(
        loss_step, 1, loss_step)))

    gain_step = 20 / len(gain_vals)
    gain_ctfs = np.unique(gain_vals.quantile(np.arange(
        gain_step, 1, gain_step)))[::-1]

    for low_ctf, high_ctf in combn(loss_ctfs, 2):
        cna_stat = (~mut_pheno
                    & cdata.train_pheno({'Gene': args.gene, 'CNA': 'Loss',
                                          'Cutoff': low_ctf}))

        wt_stat = (~mut_pheno
                   & ~cdata.train_pheno({'Gene': args.gene, 'CNA': 'Range',
                                         'Cutoff': (low_ctf, high_ctf)})
                   & ~cdata.train_pheno({'Gene': args.gene, 'CNA': 'Gain',
                                         'Cutoff': -high_ctf}))

        if (np.sum(cna_stat) >= 20) & (np.sum(wt_stat) >= 20):
            ctf_list += [(low_ctf, high_ctf)]

    for high_ctf, low_ctf in combn(gain_ctfs, 2):
        cna_stat = (~mut_pheno
                    & cdata.train_pheno({'Gene': args.gene, 'CNA': 'Gain',
                                         'Cutoff': high_ctf}))

        wt_stat = (~mut_pheno
                   & ~cdata.train_pheno({'Gene': args.gene, 'CNA': 'Range',
                                         'Cutoff': (low_ctf, high_ctf)})
                   & ~cdata.train_pheno({'Gene': args.gene, 'CNA': 'Loss',
                                         'Cutoff': -low_ctf}))

        if (np.sum(cna_stat) >= 20) & (np.sum(wt_stat) >= 20):
            ctf_list += [(low_ctf, high_ctf)]

    # save the list of found non-duplicate subtypes to file
    pickle.dump(
        sorted(ctf_list),
        open(os.path.join(base_dir, 'setup', 'ctf_lists',
                          '{}_{}.p'.format(args.cohort, args.gene)),
             'wb')
        )

    with open(os.path.join(base_dir, 'setup', 'ctf_counts',
                           '{}_{}.txt'.format(args.cohort, args.gene)),
              'w') as fl:

        fl.write(str(len(ctf_list)))


if __name__ == '__main__':
    main()

