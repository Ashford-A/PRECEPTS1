
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.expression import get_expr_firehose
from HetMan.features.variants import get_variants_mc3
from HetMan.features.cohorts import VariantCohort

import synapseclient
import dill as pickle
import argparse

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description='Set up searching for sub-types to detect.'
        )

    # positional command line arguments
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')

    # optional command line arguments controlling the thresholds for which
    # individual mutations and how many genes' mutations are considered
    parser.add_argument(
        '--freq_cutoff', type=int, default=20,
        help='sub-type sample frequency threshold'
        )
    parser.add_argument(
        '--max_genes', type=int, default=10,
        help='maximum number of mutated genes to consider'
        )

    # optional command line argument controlling verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'output',
                            args.cohort, args.classif, 'comb')

    if args.verbose:
        print("Looking for mutation sub-types in cohort {} with at least {} "
              "samples in total.\n".format(
                  args.cohort, args.freq_cutoff))

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
    common_genes = set(gene_counts.index[gene_counts >= args.freq_cutoff])

    if args.verbose:
        print("Found {} candidate genes with at least {} potential "
              "mutated samples.".format(len(common_genes), args.freq_cutoff))

    # if too many genes passed the frequency cutoff, use only the top n by
    # frequency - note that ties are broken arbitrarily and so the list of
    # genes chosen will differ slightly between runs
    if len(common_genes) >= args.max_genes:
        gene_counts = gene_counts[common_genes].sort_values(ascending=False)
        common_genes = set(gene_counts[:args.max_genes].index)

        if args.verbose:
            print("Too many genes found, culling list to {} genes which each "
                  "have at least {} mutated samples.".format(
                      args.max_genes, min(gene_counts[common_genes])))

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=common_genes, mut_levels=['Gene'],
        expr_source='Firehose', data_dir=firehose_dir, cv_prop=1.0, syn=syn
        )

    use_mtypes = cdata.train_mut.branchtypes(sub_levels=['Gene'],
                                             min_size=args.freq_cutoff)

    if args.verbose:
        print("\nFound {} total sub-types!".format(len(use_mtypes)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(sorted(list(use_mtypes)),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == '__main__':
    main()

