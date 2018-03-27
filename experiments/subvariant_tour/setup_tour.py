
"""Enumerating the sub-types in a cohort to be tested by a classifier.

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.variants import MuType

import argparse
import synapseclient
from itertools import chain, combinations
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def powerset_plus(iterable):
    s = list(iterable)

    return chain.from_iterable(combinations(s, r) for r in
                               range(1, len(s) + 1))


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description='Set up touring for sub-types to detect.'
        )
    parser.add_argument('cohort', type=str, help="Which TCGA cohort to use.")

    # optional command line arguments controlling the thresholds for which
    # individual mutations and how many genes' mutations are considered
    parser.add_argument(
        '--freq_cutoff', type=float, default=0.02,
        help='sub-type sample frequency threshold'
        )

    # optional command line arguments for what kinds of mutation sub-types to
    # look for in terms of properties and number of mutations to combine
    parser.add_argument(
        '--mut_levels', type=str, default='Form_base__Exon__Protein',
        help='the mutation property levels to consider in addition to `Genes`'
        )

    # optional command line argument controlling verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup', args.cohort)
    os.makedirs(out_path, exist_ok=True)

    # log into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()
    
    cdata = MutationCohort(cohort=args.cohort, mut_genes=None,
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', var_source='mc3', syn=syn,
                           expr_dir=firehose_dir, cv_prop=1.0,
                           samp_cutoff=args.freq_cutoff)

    if args.verbose:
        print("Found {} candidate genes with mutations in at least {:.1f}% "
              "of samples.".format(len(tuple(cdata.train_mut)),
                                   args.freq_cutoff * 100)
             )

    # intializes the list of found sub-types and the list of samples each
    # sub-type appears in
    use_mtypes = {MuType({('Gene', gn): None}) for gn, _ in cdata.train_mut}
    use_sampsets = {frozenset(mtype.get_samples(cdata.train_mut))
                    for mtype in use_mtypes}

    for lvl_combn in powerset_plus(args.mut_levels.split('__')):
        use_lvls = ['Gene'] + list(lvl_combn)
        mtype_sampsets = dict()

        if args.verbose:
            print("Looking for sub-types that are combinations of up to "
                  "two mutations at levels {}...\n".format(use_lvls))

        for gn, mut in cdata.train_mut:
            cur_mtypes = {
                MuType({('Gene', gn): mtype})
                for mtype in mut.combtypes(
                    comb_sizes=(1, 2), sub_levels=use_lvls,
                    min_type_size=args.freq_cutoff * len(cdata.samples)
                    )
                }

            # finds the samples belonging to each enumerated sub-type that
            # hasn't already been found
            cur_sampsets = {mtype: frozenset(mtype.get_samples(cdata.train_mut))
                            for mtype in cur_mtypes - use_mtypes}

            # removes the sub-types with so many mutated samples that there
            # are not enough negatively-labelled samples for classification
            mtype_sampsets.update({
                mtype: sampset for mtype, sampset in cur_sampsets.items()
                if len(sampset) <= (len(cdata.samples)
                                    * (1 - args.freq_cutoff))
                })

        # ensures that when two sub-types have the same samples the one
        # further down the sort order gets removed
        sub_mtypes = sorted(list(mtype_sampsets))
        if args.verbose:
            print("Found {} new sub-types!\n".format(len(sub_mtypes)))

            for i, mtype in enumerate(sub_mtypes):

                if args.verbose and (i % 200) == 100:
                    print("\nchecked {} sub-types\n".format(i))

                # ...we remove each one whose set of mutated samples is
                # identical to that of a sub-type that was already found
                if mtype_sampsets[mtype] in use_sampsets:
                    if args.verbose:
                        print("Removing functionally duplicate MuType {}"\
                                .format(mtype))

                else:
                    use_mtypes.update({mtype})
                    use_sampsets.update({mtype_sampsets[mtype]})

    if args.verbose:
        print("\nFound {} total sub-types!".format(len(use_mtypes)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(
        sorted(list(use_mtypes)),
        open(os.path.join(
            out_path, 'mtype_list__freq_{}__levels_{}.p'.format(
                args.freq_cutoff, args.mut_levels)
            ), 'wb')
        )

    pickle.dump({'Samps': cdata.samples},
                open(os.path.join(out_path, 'cohort_info.p'), 'wb'))


if __name__ == '__main__':
    main()

