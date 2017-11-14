
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.expression import get_expr_firehose
from HetMan.features.variants import get_variants_mc3

from HetMan.features.cohorts import VariantCohort
from HetMan.features.variants import MuType

import synapseclient
import dill as pickle
import argparse

from itertools import chain, islice, combinations
from copy import deepcopy

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def rev_powerset_slice(iterable, stop):
    s = list(iterable)
    return reversed(list(
        islice(chain.from_iterable(combinations(s, r)
                                   for r in range(0, len(s) + 1)),
               stop)
        ))


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description='Set up searching for sub-types to detect.'
        )

    # positional command line arguments
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')
    parser.add_argument('base_gene', type=str,
                        help='a gene to cross with respect to')

    # optional command line arguments controlling the thresholds for which
    # individual mutations and how many genes' mutations are considered
    parser.add_argument(
        '--freq_cutoff', type=int, default=20,
        help='sub-type sample frequency threshold'
        )
    parser.add_argument(
        '--max_genes', type=int, default=200,
        help='maximum number of mutated genes to consider'
        )

    # optional command line arguments for what kinds of mutation sub-types to
    # look for in terms of properties and number of mutations to combine
    parser.add_argument(
        '--mut_levels', type=str, nargs='+',
        default=['Form_base', 'Exon', 'Protein'],
        help='the mutation property levels to consider in addition to `Genes`'
        )
    parser.add_argument(
        '--comb_size', type=int, default=2,
        help='maximum number of individual mutations to combine'
             'when searching for mutation sub-types'
        )

    # optional command line argument controlling verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(
        base_dir, 'output',
        args.cohort, args.classif, 'cross', args.base_gene
        )

    if args.verbose:
        print("Looking for mutation sub-types in cohort {} composed of at "
              "most {} individual mutations with at least {} "
              "samples in total.\n".format(
                  args.cohort, args.comb_size, args.freq_cutoff))

    # log into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # load the expression matrix for the given cohort from Broad Firehose,
    # load the MC3 variant call set from Synapse, find the mutations for the
    # samples that are in both datasets
    expr_data = get_expr_firehose(args.cohort, firehose_dir)
    mc3_data = get_variants_mc3(syn)
    expr_mc3 = mc3_data.loc[mc3_data['Sample'].isin(expr_data.index), :]

    # get the genes whose mutations appear in enough samples to pass the
    # frequency threshold
    gene_counts = expr_mc3.groupby(by='Gene').Sample.nunique()
    count_cutoff = int(args.freq_cutoff / args.comb_size)
    common_genes = set(gene_counts.index[gene_counts >= count_cutoff])

    if args.verbose:
        print("Found {} candidate genes with at least {} potential "
              "mutated samples.".format(len(common_genes), count_cutoff))

    if len(common_genes) >= args.max_genes:
        gene_counts = gene_counts[common_genes].sort_values(ascending=False)
        common_genes = set(gene_counts[:args.max_genes].index)

        if args.verbose:
            print("Too many genes found, culling list to {} genes which each "
                  "have at least {} mutated samples.".format(
                      args.max_genes, min(gene_counts[common_genes])))


    cdata = VariantCohort(cohort=args.cohort, mut_genes=common_genes,
                          mut_levels=['Gene'] + args.mut_levels,
                          expr_source='Firehose', data_dir=firehose_dir,
                          cv_prop=1.0, syn=syn)

    base_mtype = MuType({('Gene', args.base_gene): None})
    base_samps = base_mtype.get_samples(cdata.train_mut)

    with_muts = deepcopy(cdata.train_mut).subtree(base_samps)
    without_muts = deepcopy(cdata.train_mut).subtree(
        cdata.samples - base_samps)

    # intializes the list of found sub-types and the list of samples each
    # sub-type appears in
    use_mtypes = set()
    use_sampsets = set()

    search_level = 1
    break_status = False

    # until we have not reached the limit of sub-type enumeration or run out
    # property level combinations to test...
    while (len(use_mtypes) < 10000 and not break_status
           and search_level <= 2 ** len(args.mut_levels)):

        # try a list of property level combinations and number of individual
        # variants to combine, where the complexity of the level combination
        # plus the variant count is held constant
        for lvl_combn, comb_size in zip(
                rev_powerset_slice(args.mut_levels, search_level),
                range(1, min(search_level + 1, args.comb_size + 1))
                ):
            use_lvls = ['Gene'] + list(lvl_combn)

            if args.verbose:
                print("\nLooking for sub-types that are combinations "
                      "of {} mutation(s) at levels {}...\n".format(
                          comb_size, use_lvls))

            # enumerates the sub-types consisting of a combination of the given
            # number of individual mutations at the given property levels
            sub_mtypes = with_muts.combtypes(
                comb_sizes=(comb_size, ), sub_levels=use_lvls,
                min_type_size=int(args.freq_cutoff / 2)
                )
            sub_mtypes |= without_muts.combtypes(
                comb_sizes=(comb_size, ), sub_levels=use_lvls,
                min_type_size=int(args.freq_cutoff / 2)
                )

            # finds the samples belonging to each enumerated sub-type that
            # hasn't already been found
            mtype_sampsets = {mtype: frozenset(mtype.get_samples(cdata.train_mut))
                              for mtype in sub_mtypes - use_mtypes
                              if (mtype & base_mtype).is_empty()}

            # removes the sub-types with so many mutated samples that there
            # are not enough negatively-labelled samples for classification
            mtype_sampsets = {
                mtype: sampset for mtype, sampset in mtype_sampsets.items()
                if len(sampset) <= (len(cdata.samples) - args.freq_cutoff)
                }

            sub_mtypes = sorted(list(mtype_sampsets))
            if args.verbose:
                print("Found {} new sub-types!\n".format(len(sub_mtypes)))

            # if the list of remaining sub-types isn't too long...
            if len(sub_mtypes) < 8000:
                add_mtypes = set()

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
                        add_mtypes.update({mtype})
                        use_sampsets.update({mtype_sampsets[mtype]})

                use_mtypes |= add_mtypes

            elif len(sub_mtypes) > 100000:
                break_status = True
        
        search_level += 1
    
    if args.verbose:
        print("\nFound {} total sub-types!".format(len(use_mtypes)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(sorted(list(use_mtypes)),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == '__main__':
    main()

