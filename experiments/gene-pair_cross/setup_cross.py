
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
import synapseclient
from itertools import product
import dill as pickle

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description='Set up touring for sub-types to detect.'
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene1', type=str, help="which gene to consider")
    parser.add_argument('gene2', type=str, help="which gene to consider")

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
    out_path = os.path.join(base_dir, 'setup', args.cohort,
                            '{}_{}'.format(args.gene1, args.gene2))

    os.makedirs(out_path, exist_ok=True)
    use_lvls = args.mut_levels.split('__')

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()
    
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene1, args.gene2],
        mut_levels=['Gene'] + use_lvls, expr_source='Firehose',
        var_source='mc3', expr_dir=firehose_dir, cv_prop=1.0, syn=syn
        )

    cross_mtypes1 = cdata.train_mut[args.gene1].find_unique_subtypes(
        max_types=40, max_combs=50, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )
    cross_mtypes2 = cdata.train_mut[args.gene2].find_unique_subtypes(
        max_types=40, max_combs=50, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )

    if args.verbose:
        print("Found {} sub-types of {} and {} sub-types of {} "
              "to cross!".format(len(cross_mtypes1), args.gene1,
                                 len(cross_mtypes2), args.gene2))

    cross_mtypes1 = {MuType({('Gene', args.gene1): mtype})
                     for mtype in cross_mtypes1}
    cross_mtypes2 = {MuType({('Gene', args.gene2): mtype})
                     for mtype in cross_mtypes2}

    samps1 = {mtype: mtype.get_samples(cdata.train_mut)
              for mtype in cross_mtypes1}
    samps2 = {mtype: mtype.get_samples(cdata.train_mut)
              for mtype in cross_mtypes2}

    use_pairs = sorted(
        (mtype1, mtype2) for mtype1, mtype2 in product(cross_mtypes1,
                                                       cross_mtypes2)
        if (len(samps1[mtype1] - samps2[mtype2]) >= args.samp_cutoff
            and len(samps2[mtype2] - samps1[mtype1]) >= args.samp_cutoff)
        )

    if args.verbose:
        print("\nSaving {} pairs with sufficient "
              "exclusivity...".format(len(use_pairs)))

    pickle.dump(use_pairs,
                open(os.path.join(
                    out_path, 'pairs_list__samps_{}__levels_{}.p'.format(
                        args.samp_cutoff, args.mut_levels)
                    ), 'wb'))

    pickle.dump({(mtype1, mtype2): cdata.mutex_test(mtype1, mtype2)
                 for mtype1, mtype2 in use_pairs},
                open(os.path.join(
                    out_path, 'pairs_mutex__samps_{}__levels_{}.p'.format(
                        args.samp_cutoff, args.mut_levels)
                    ), 'wb'))

    pickle.dump({'Samps': cdata.samples},
                open(os.path.join(out_path, 'cohort_info.p'), 'wb'))

    with open(os.path.join(
            out_path,
            'pairs_count__samps_{}__levels_{}.txt'.format(
                args.samp_cutoff, args.mut_levels)), 'w') as fl:

        fl.write(str(len(use_pairs)))


if __name__ == '__main__':
    main()

