
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
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description='Set up touring for sub-types to detect.'
        )
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")

    # optional command line arguments controlling the thresholds for which
    # individual mutations and how many genes' mutations are considered
    parser.add_argument('--freq_cutoff', type=float, default=0.02,
                        help='subtype sample frequency threshold')

    # optional command line arguments for what kinds of mutation sub-types to
    # look for in terms of properties and number of mutations to combine
    parser.add_argument('--mut_levels', type=str, default='Gene',
                        help='the mutation property levels to consider')

    # optional command line argument controlling verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup', args.cohort)
    os.makedirs(out_path, exist_ok=True)
    use_lvls = args.mut_levels.split('__')

    # log into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()
    
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=None, mut_levels=use_lvls,
        expr_source='Firehose', var_source='mc3', expr_dir=firehose_dir,
        cv_prop=1.0, samp_cutoff=args.freq_cutoff, syn=syn
        )

    if args.verbose:
        print("Found {} candidate genes with mutations in at least "
              "{:.1f}% of the samples in TCGA cohort {}.\nLooking for "
              "subtypes of these genes that are combinations of up to two "
              "mutations at annotation levels {} ...\n".format(
                  len(tuple(cdata.train_mut)), args.freq_cutoff * 100,
                  args.cohort, use_lvls
                )
             )
    
    min_samps = args.freq_cutoff * len(cdata.samples)
    if use_lvls == ['Gene']:

        use_mtypes = {MuType({('Gene', gn): None})
                      for gn, mut in cdata.train_mut
                      if len(mut) >= min_samps}

    elif use_lvls[0] == 'Gene':
        use_lvls = use_lvls[1:]

        use_mtypes = set()
        use_sampsets = set()
        mtype_sampsets = dict()

        for gn, mut in cdata.train_mut:
            cur_mtypes = {
                MuType({('Gene', gn): mtype})
                for mtype in mut.combtypes(comb_sizes=(1, 2),
                                           sub_levels=use_lvls,
                                           min_type_size=min_samps)
                }

            # finds the samples belonging to each enumerated sub-type that
            # hasn't already been found
            cur_sampsets = {
                mtype: frozenset(mtype.get_samples(cdata.train_mut))
                for mtype in cur_mtypes - use_mtypes}

            # removes the sub-types with so many mutated samples that there
            # are not enough negatively-labelled samples for classification
            mtype_sampsets.update({
                mtype: sampset for mtype, sampset in cur_sampsets.items()
                if len(sampset) <= (len(cdata.samples) - min_samps)
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

    else:
        cur_mtypes = cdata.train_mut.combtypes(comb_sizes=(1, 2),
                                               sub_levels=use_lvls,
                                               min_type_size=min_samps)

        use_mtypes = set()
        use_sampsets = set()
        mtype_sampsets = dict()

        cur_sampsets = {mtype: frozenset(mtype.get_samples(cdata.train_mut))
                        for mtype in cur_mtypes - use_mtypes}

        # removes the sub-types with so many mutated samples that there
        # are not enough negatively-labelled samples for classification
        mtype_sampsets.update({
            mtype: sampset for mtype, sampset in cur_sampsets.items()
            if len(sampset) <= (len(cdata.samples) - min_samps)
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

    with open(os.path.join(
            out_path,
            'mtype_count__freq_{}__levels_{}.txt'.format(
                args.freq_cutoff, args.mut_levels)), 'w') as fl:

        fl.write(str(len(use_mtypes)))


if __name__ == '__main__':
    main()

