
"""Enumerating the pairs of subtypes of a gene in a cohort to be isolated.

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

import argparse
import synapseclient
from HetMan.features.cohorts.tcga import MutationCohort
from itertools import combinations as combn
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect cross-isolation "
        "experiment by enumerating the pairs of subtypes to be tested."
        )

    # create positional command line arguments
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which gene to consider")
    parser.add_argument('mut_levels', type=str,
                        help='the mutation property levels to consider')

    # create optional command line arguments
    parser.add_argument('--samp_cutoff', type=int, default=25,
                        help='subtype sample frequency threshold')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command line arguments, create directory where found subtypes
    # will be stored
    args = parser.parse_args()
    use_lvls = args.mut_levels.split('__')
    out_path = os.path.join(base_dir, 'setup', args.cohort, args.gene)
    os.makedirs(out_path, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()
    
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=use_lvls,
        expr_source='Firehose', var_source='mc3', expr_dir=firehose_dir,
        cv_prop=1.0, syn=syn
        )

    if args.verbose:
        print("Looking for combinations of subtypes of mutations in gene {} "
              "present in at least {} of the samples in TCGA cohort {} at "
              "annotation levels {}.\n".format(
                  args.gene, args.samp_cutoff, args.cohort, use_lvls)
             )

    cross_mtypes = cdata.train_mut.find_unique_subtypes(
        max_types=100, max_combs=10, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )

    mtype_samps = {mtype: mtype.get_samples(cdata.train_mut)
                   for mtype in cross_mtypes}
    cross_mtypes = {
        mtype for mtype in cross_mtypes
        if len(mtype_samps[mtype]) <= (len(cdata.samples) - args.samp_cutoff)
        }

    if args.verbose:
        print("\nFound {} total sub-types to cross!".format(
            len(cross_mtypes)))

    use_pairs = {(mtype1, mtype2) for mtype1, mtype2 in combn(cross_mtypes, 2)
                 if ((len(mtype_samps[mtype1] - mtype_samps[mtype2])
                      >= args.samp_cutoff)
                     and (len(mtype_samps[mtype2] - mtype_samps[mtype1])
                          >= args.samp_cutoff)
                     and (len(mtype_samps[mtype1] | mtype_samps[mtype2])
                          <= (len(cdata.samples) - args.samp_cutoff))
                     and (mtype1 & mtype2).is_empty())}

    if args.verbose:
        print("\nFound {} non-overlapping sub-type pairs!".format(
            len(use_pairs)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(
        sorted(use_pairs),
        open(os.path.join(out_path,
                          'pairs_list__samps_{}__levels_{}.p'.format(
                              args.samp_cutoff, args.mut_levels)),
             'wb')
        )

    with open(os.path.join(out_path,
                           'pairs_count__samps_{}__levels_{}.txt'.format(
                               args.samp_cutoff, args.mut_levels)),
              'w') as fl:

        fl.write(str(len(use_pairs)))


if __name__ == '__main__':
    main()

