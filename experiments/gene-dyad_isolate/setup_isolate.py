
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
import synapseclient
from itertools import combinations as combn
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def main():
    parser = argparse.ArgumentParser(
        "Set up the paired gene expression effect isolation experiment by "
        "enumerating the dyads of genes to be tested."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('--samp_cutoff', type=int, default=40,
                        help='subtype sample frequency threshold')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command line arguments, create directory where found pairs
    # will be stored
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup', args.cohort)
    os.makedirs(out_path, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()
    
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=None, mut_levels=['Gene'],
        expr_source='Firehose', var_source='mc3', expr_dir=firehose_dir,
        samp_cutoff=args.samp_cutoff, cv_prop=1.0, syn=syn
        )

    if args.verbose:
        print("Looking for pairs of mutated genes present in at least {} of "
              "the samples in TCGA cohort {} with {} total samples.".format(
                  args.samp_cutoff, args.cohort, len(cdata.samples))
                )

    gene_pairs = {
        (MuType({('Gene', gn1): None}), MuType({('Gene', gn2): None}))
        for (gn1, muts1), (gn2, muts2) in combn(cdata.train_mut, r=2)
        if (len(muts1 - muts2) >= args.samp_cutoff
            and len(muts2 - muts1) >= args.samp_cutoff
            and len(muts1 | muts2) <= (len(cdata.samples) - args.samp_cutoff))
        }

    if args.verbose:
        print("Found {} pairs of genes to isolate!".format(len(gene_pairs)))

    pickle.dump(
        sorted(gene_pairs),
        open(os.path.join(out_path,
                          'pairs_list__samps_{}.p'.format(args.samp_cutoff)),
             'wb')
        )

    with open(os.path.join(out_path,
                           'pairs_count__samps_{}.txt'.format(
                               args.samp_cutoff)),
              'w') as fl:

        fl.write(str(len(gene_pairs)))


if __name__ == '__main__':
    main()

