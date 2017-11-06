
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

    # optional command line arguments controlling how mutation sub-types are
    # combined and filtered during enumeration
    parser.add_argument(
        '--freq_cutoff', type=int, default=15,
        help='sub-type sample frequency threshold'
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
    out_path = os.path.join(base_dir, 'output',
                            args.cohort, args.classif, 'search')

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

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=common_genes,
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        #mut_levels=['Gene', 'Form_base'],
        expr_source='Firehose', data_dir=firehose_dir,
        cv_prop=1.0, syn=syn
        )

    # intializes the list of found sub-types, the list of samples each sub-
    # type appears in, and the number of variants to combine in each sub-type
    use_mtypes = set()
    use_sampsets = set()
    comb_size = 1

    # finds the sub-types satisfying the sample frequency criterion, starting
    # with the ones that consist of a single type of mutation variant
    while len(use_mtypes) < 8000 and comb_size <= args.comb_size:
        if args.verbose:
            print("\nLooking for sub-types that are combinations "
                  "of {} branch(es)...\n".format(comb_size))

        # enumerates the sub-types consisting of a combination of the given
        # number of variant types, finds which samples have each sub-type
        sub_mtypes = cdata.train_mut.combtypes(comb_sizes=(comb_size, ),
                                               min_type_size=args.freq_cutoff)
        mtype_sampsets = {mtype: frozenset(mtype.get_samples(cdata.train_mut))
                          for mtype in sub_mtypes}

        mtype_sampsets = {
            mtype: sampset for mtype, sampset in mtype_sampsets.items()
            if len(sampset) <= (len(cdata.samples) - args.freq_cutoff)
            }
        sub_mtypes = sorted(list(mtype_sampsets))

        # removes the sub-types that have the same set of samples as
        # another already-found sub-type
        if args.verbose:
            print("Found {} new sub-types!\n".format(len(sub_mtypes)))

        if len(sub_mtypes) < 7500:
            add_mtypes = set()
            if comb_size == 1:
                print(sub_mtypes)

            for i, mtype in enumerate(sub_mtypes):
                if args.verbose and (i % 200) == 100:
                    print("\nchecked {} sub-types\n".format(i))

                if mtype_sampsets[mtype] in use_sampsets:
                    if args.verbose:
                        print("Removing functionally duplicate MuType {}"\
                                .format(mtype))

                else:
                    add_mtypes.update({mtype})
                    use_sampsets.update({mtype_sampsets[mtype]})

            use_mtypes |= add_mtypes

        else:
            break
        
        comb_size += 1
    
    if args.verbose:
        print("Found {} total sub-types!".format(len(use_mtypes)))

    # save the list of found non-duplicate sub-types to file
    pickle.dump(sorted(list(use_mtypes)),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == '__main__':
    main()

