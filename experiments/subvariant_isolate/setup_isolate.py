
"""Enumerating the subtypes of a gene in a cohort to be isolated.

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_isolate import *
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
from itertools import combinations as combn
import dill as pickle


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    # create positional command line arguments
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which gene to consider")
    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")

    # create optional command line arguments
    parser.add_argument('--samp_cutoff', type=int, default=20,
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
    syn.cache.cache_root_dir = syn_root
    syn.login()

    # load expression and variant call data for the given TCGA cohort
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=use_lvls,
        expr_source='Firehose', var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=firehose_dir, cv_prop=1.0, syn=syn
        )

    if args.verbose:
        print("Looking for combinations of subtypes of mutations in gene {} "
              "present in at least {} of the samples in TCGA cohort {} at "
              "annotation levels {}.\n".format(
                  args.gene, args.samp_cutoff, args.cohort, use_lvls)
             )

    # find mutation subtypes present in enough samples in the TCGA cohort
    pnt_mtypes = cdata.train_mut['Point'].find_unique_subtypes(
        max_types=500, max_combs=2, verbose=2,
        sub_levels=use_lvls, min_type_size=args.samp_cutoff
        )

    # filter out the subtypes that appear in too many samples for there to
    # be a wild-type class of sufficient size for classification
    pnt_mtypes = {MuType({('Scale', 'Point'): mtype}) for mtype in pnt_mtypes
                  if (len(mtype.get_samples(cdata.train_mut['Point']))
                      <= (len(cdata.samples) - args.samp_cutoff))}

    cna_mtypes = cdata.train_mut['Copy'].branchtypes(min_size=5)
    cna_mtypes = {MuType({('Scale', 'Copy'): mtype}) for mtype in cna_mtypes
                  if (len(mtype.get_samples(cdata.train_mut['Copy']))
                      <= (len(cdata.samples) - args.samp_cutoff))}

    all_mtype = MuType(cdata.train_mut.allkey())
    use_mtypes = pnt_mtypes | cna_mtypes

    only_mtypes = {(mtype, ) for mtype in use_mtypes
                   if (len(mtype.get_samples(cdata.train_mut)
                           - (all_mtype - mtype).get_samples(cdata.train_mut))
                       >= args.samp_cutoff)}

    comb_mtypes = {(mtype1, mtype2) for mtype1, mtype2 in combn(use_mtypes, 2)
                   if ((mtype1 & mtype2).is_empty()
                       and (len((mtype1.get_samples(cdata.train_mut)
                                 & mtype2.get_samples(cdata.train_mut))
                                - (mtype1.get_samples(cdata.train_mut)
                                   ^ mtype2.get_samples(cdata.train_mut))
                                - (all_mtype - mtype1 - mtype2).get_samples(
                                    cdata.train_mut))
                            >= args.samp_cutoff))}

    if args.verbose:
        print("\nFound {} exclusive sub-types and {} combination sub-types "
              "to isolate!".format(len(only_mtypes), len(comb_mtypes)))

    # save the list of found non-duplicate subtypes to file
    pickle.dump(
        sorted(only_mtypes | comb_mtypes),
        open(os.path.join(out_path,
                          'mtypes_list__samps_{}__levels_{}.p'.format(
                              args.samp_cutoff, args.mut_levels)),
             'wb')
        )

    # save the number of found subtypes to file
    with open(os.path.join(out_path,
                           'mtypes_count__samps_{}__levels_{}.txt'.format(
                               args.samp_cutoff, args.mut_levels)),
              'w') as fl:

        fl.write(str(len(only_mtypes | comb_mtypes)))


if __name__ == '__main__':
    main()

