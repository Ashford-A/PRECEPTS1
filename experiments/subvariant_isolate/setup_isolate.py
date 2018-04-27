
"""Enumerating the sub-types in a cohort to be tested by a classifier.

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
import synapseclient
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description='Set up touring for sub-types to detect.'
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which gene to consider")

    parser.add_argument(
        'mut_levels', type=str,
        help='the mutation property levels to consider, in addition to `Gene`'
        )

    parser.add_argument('--samp_cutoff', type=int, default=20,
                        help='subtype sample frequency threshold')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup', args.cohort, args.gene)
    os.makedirs(out_path, exist_ok=True)
    use_lvls = args.mut_levels.split('__')

    # log into Synapse using locally-stored credentials
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
    
    use_mtypes = set()
    use_sampsets = set()
    use_combs = 1

    while len(use_mtypes) <= 2000 and use_combs <= 10:
        cur_mtypes = cdata.train_mut.combtypes(
            comb_sizes=(use_combs, ), sub_levels=use_lvls,
            min_type_size=args.samp_cutoff
            )
        
        # finds the samples belonging to each enumerated sub-type that
        # hasn't already been found
        cur_sampsets = {mtype: frozenset(mtype.get_samples(cdata.train_mut))
                        for mtype in cur_mtypes - use_mtypes}
        cur_sampsets = {
            mtype: sampset for mtype, sampset in cur_sampsets.items()
            if len(sampset) <= (len(cdata.samples) - args.samp_cutoff)
            }

        # ensures that when two sub-types have the same samples the one
        # further down the sort order gets removed
        if args.verbose:
            print("Found {} new sub-types!\n".format(len(cur_sampsets)))

        if len(cur_sampsets) <= 2000:
            for i, (mtype, sampset) in enumerate(cur_sampsets.items()):
                if args.verbose and (i % 200) == 100:
                        print("\nchecked {} sub-types\n".format(i))

                # ...we remove each one whose set of mutated samples is
                # identical to that of a sub-type that was already found
                if sampset in use_sampsets:
                    if args.verbose:
                        print("Removing functionally duplicate MuType {}"\
                                .format(mtype))

                else:
                    use_mtypes.update({mtype})
                    use_sampsets.update({sampset})

            use_combs += 1

        else:
            use_combs = 20

    if args.verbose:
        print("\nFound {} total sub-types!".format(len(use_mtypes)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(sorted(MuType({('Gene', args.gene): mtype})
                       for mtype in use_mtypes),
                open(os.path.join(
                    out_path, 'mtypes_list__samps_{}__levels_{}.p'.format(
                        args.samp_cutoff, args.mut_levels)
                    ), 'wb'))

    with open(os.path.join(
            out_path,
            'mtypes_count__samps_{}__levels_{}.txt'.format(
                args.samp_cutoff, args.mut_levels)), 'w') as fl:

        fl.write(str(len(use_mtypes)))


if __name__ == '__main__':
    main()

