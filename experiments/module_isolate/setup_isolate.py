
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.module_isolate import *
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import dill as pickle

from functools import reduce
from operator import or_
from itertools import combinations as combn
from itertools import chain


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
    parser.add_argument('--samp_cutoff', type=int, default=20,
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
    syn.cache.cache_root_dir = syn_root
    syn.login()
 
    cdata = MutationCohort(cohort=args.cohort, mut_genes=args.genes,
                           mut_levels=['Gene'] + use_lvls,
                           expr_source='Firehose', var_source='mc3',
                           copy_source='Firehose', annot_file=annot_file,
                           expr_dir=expr_dir, cv_prop=1.0, syn=syn)

    iso_mtypes = set()
    for gene in args.genes:
        other_samps = reduce(or_, [cdata.train_mut[other_gn].get_samples()
                                   for other_gn in set(args.genes) - {gene}])

        if args.verbose:
            print("Looking for combinations of subtypes of mutations in gene "
                  "{} present in at least {} of the samples in TCGA cohort "
                  "{} at annotation levels {}.\n".format(
                      gene, args.samp_cutoff, args.cohort, use_lvls)
                    )

        pnt_mtypes = cdata.train_mut[gene]['Point'].find_unique_subtypes(
            max_types=500, max_combs=2, verbose=2,
            sub_levels=use_lvls, min_type_size=args.samp_cutoff
            )

        # filter out the subtypes that appear in too many samples for there to
        # be a wild-type class of sufficient size for classification
        pnt_mtypes = {MuType({('Scale', 'Point'): mtype}) for mtype in pnt_mtypes
                      if (len(mtype.get_samples(cdata.train_mut[gene]['Point']))
                          <= (len(cdata.samples) - args.samp_cutoff))}

        cna_mtypes = cdata.train_mut[gene]['Copy'].branchtypes(min_size=5)
        cna_mtypes = {MuType({('Scale', 'Copy'): mtype}) for mtype in cna_mtypes
                      if (len(mtype.get_samples(cdata.train_mut[gene]['Copy']))
                          <= (len(cdata.samples) - args.samp_cutoff))}

        all_mtype = MuType(cdata.train_mut[gene].allkey())
        use_mtypes = pnt_mtypes | cna_mtypes

        only_mtypes = {
            (MuType({('Gene', gene): mtype}), ) for mtype in use_mtypes
            if (len(mtype.get_samples(cdata.train_mut[gene])
                    - (all_mtype - mtype).get_samples(cdata.train_mut[gene])
                    - other_samps)
                >= args.samp_cutoff)
            }

        comb_mtypes = {
            (MuType({('Gene', gene): mtype1}),
             MuType({('Gene', gene): mtype2}))
            for mtype1, mtype2 in combn(use_mtypes, 2)
            if ((mtype1 & mtype2).is_empty()
                and (len((mtype1.get_samples(cdata.train_mut[gene])
                          & mtype2.get_samples(cdata.train_mut[gene]))
                         - (mtype1.get_samples(cdata.train_mut[gene])
                            ^ mtype2.get_samples(cdata.train_mut[gene]))
                         - (all_mtype - mtype1 - mtype2).get_samples(
                             cdata.train_mut[gene])
                         - other_samps)
                     >= args.samp_cutoff))
            }

        iso_mtypes |= only_mtypes | comb_mtypes
        if args.verbose:
            print("\nFound {} exclusive sub-types and {} combination sub-types "
                  "to isolate!".format(len(only_mtypes), len(comb_mtypes)))

    for cur_genes in chain.from_iterable(combn(args.genes, r)
                                         for r in range(1, len(args.genes))):
        gene_mtype = MuType({('Gene', cur_genes): None})
        rest_mtype = MuType({('Gene',
                              tuple(set(args.genes) - set(cur_genes))): None})
 
        if (args.samp_cutoff <= len(gene_mtype.get_samples(cdata.train_mut)
                                    - rest_mtype.get_samples(cdata.train_mut))
                <= (len(cdata.samples) - args.samp_cutoff)):
            iso_mtypes |= {(gene_mtype, )}

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

