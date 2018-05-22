
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
    parser = argparse.ArgumentParser(
        "Set up the paired-gene subtype expression effect isolation "
        "experiment by enumerating the subtypes to be tested."
        )

    # create positional command line arguments
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene1', type=str, help="the first gene to consider")
    parser.add_argument('gene2', type=str, help="the second gene to consider")
    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")

    # create optional command line arguments
    parser.add_argument('--samp_cutoff', type=int, default=25,
                        help='subtype sample frequency threshold')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command line arguments, create directory where found subtypes
    # will be stored
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

    if args.verbose:
        print("Looking for combinations of subtypes of mutations in genes {} "
              "and {} present in at least {} of the samples in TCGA cohort "
              "{} at annotation levels {}.\n".format(
                  args.gene1, args.gene2,
                  args.samp_cutoff, args.cohort, use_lvls
                ))
 
    iso_mtypes1 = cdata.train_mut[args.gene1].find_unique_subtypes(
        max_types=750, max_combs=10, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )
    iso_mtypes2 = cdata.train_mut[args.gene2].find_unique_subtypes(
        max_types=750, max_combs=10, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )

    use_mtypes1 = {
        MuType({('Gene', args.gene1): mtype}) for mtype in iso_mtypes1
        if (len(mtype.get_samples(cdata.train_mut))
            <= (len(cdata.samples) - args.samp_cutoff))
        }
    use_mtypes2 = {
        MuType({('Gene', args.gene2): mtype}) for mtype in iso_mtypes2
        if (len(mtype.get_samples(cdata.train_mut))
            <= (len(cdata.samples) - args.samp_cutoff))
        }

    if args.verbose:
        print("\nFound {} subtypes of gene {} and {} subtypes of "
              "gene {} to isolate!".format(len(use_mtypes1), args.gene1,
                                           len(use_mtypes2), args.gene2))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(
        sorted(use_mtypes1 | use_mtypes2),
        open(os.path.join(out_path,
                          'mtypes_list__samps_{}__levels_{}.p'.format(
                              args.samp_cutoff, args.mut_levels)),
             'wb')
        )

    with open(os.path.join(out_path,
                           'mtypes_count__samps_{}__levels_{}.txt'.format(
                               args.samp_cutoff, args.mut_levels)),
              'w') as fl:

        fl.write(str(len(use_mtypes1 | use_mtypes2)))


if __name__ == '__main__':
    main()

