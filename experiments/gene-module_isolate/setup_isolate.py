
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
    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
    parser.add_argument('genes', type=str, nargs='+',
                        help="a list of mutated genes")

    # create optional command line arguments
    parser.add_argument('--samp_cutoff', type=int, default=25,
                        help='subtype sample frequency threshold')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command line arguments, create directory where found subtypes
    # will be stored
    args = parser.parse_args()
    use_lvls = args.mut_levels.split('__')
    out_path = os.path.join(base_dir, 'setup', args.cohort,
                            '_'.join(args.genes))
    os.makedirs(out_path, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()
    
    cdata = MutationCohort(cohort=args.cohort, mut_genes=args.genes,
                           mut_levels=['Gene'] + use_lvls,
                           expr_source='Firehose', var_source='mc3',
                           expr_dir=firehose_dir, cv_prop=1.0, syn=syn)

    iso_mtypes = set()
    for gene in args.genes:
        if args.verbose:
            print("Looking for combinations of subtypes of mutations in gene "
                  "{} present in at least {} of the samples in TCGA cohort "
                  "{} at annotation levels {}.\n".format(
                      gene, args.samp_cutoff, args.cohort, use_lvls)
                    )

        gene_mtypes = cdata.train_mut[gene].find_unique_subtypes(
            max_types=1500 / len(args.genes), max_combs=4, verbose=2,
            sub_levels=use_lvls, min_type_size=args.samp_cutoff
            )

        if args.verbose:
            print("\nFound {} subtypes of gene {} to isolate!".format(
                len(gene_mtypes), gene))

        iso_mtypes |= {
            MuType({('Gene', gene): mtype}) for mtype in gene_mtypes
            if (len(mtype.get_samples(cdata.train_mut[gene]))
                <= (len(cdata.samples) - args.samp_cutoff))
            }

    if args.verbose:
        print("\nFound {} total sub-types to isolate!".format(
            len(iso_mtypes)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(
        sorted(iso_mtypes),
        open(os.path.join(out_path,
                          'mtypes_list__samps_{}__levels_{}.p'.format(
                              args.samp_cutoff, args.mut_levels)),
             'wb')
        )

    with open(os.path.join(out_path,
                           'mtypes_count__samps_{}__levels_{}.txt'.format(
                               args.samp_cutoff, args.mut_levels)),
              'w') as fl:

        fl.write(str(len(iso_mtypes)))


if __name__ == '__main__':
    main()

