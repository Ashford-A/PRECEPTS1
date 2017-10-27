
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

    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')
    parser.add_argument('freq_cutoff', type=int,
                        help='sub-type sample frequency threshold')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'output',
                            args.cohort, args.classif, 'search')

    # log into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # load the expression matrix for the given cohort from Broad Firehose and
    # load the MC3 variant call set from Synapse
    expr_data = get_expr_firehose(args.cohort, firehose_dir)
    mc3_data = get_variants_mc3(syn)

    # find the variant calls for the cohort's samples, get the set of genes
    # whose mutations appear in enough samples to pass the frequency threshold
    expr_mc3 = mc3_data.loc[mc3_data['Sample'].isin(expr_data.index), :]
    gene_counts = expr_mc3.groupby(by='Gene').count()['Sample']
    common_genes = set(gene_counts[gene_counts >= args.freq_cutoff].index)

    if args.verbose:
        print("Found {} candidate genes in cohort {} with at least {} "
              "mutated samples.".format(
                  len(common_genes), args.cohort, args.freq_cutoff))

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=common_genes,
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
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
    while len(use_mtypes) < 8000 and comb_size <= 2:
        if args.verbose:
            print("Looking for sub-types that are combinations "
                  "of {} branch(es).".format(comb_size))

        # enumerates the sub-types consisting of a combination of the given
        # number of variant types, finds which samples have each sub-type
        sub_mtypes = cdata.train_mut.combtypes(comb_sizes=(comb_size, ),
                                               min_type_size=args.freq_cutoff)
        mtype_sampsets = {mtype: frozenset(mtype.get_samples(cdata.train_mut))
                          for mtype in sub_mtypes}

        # removes the sub-types that have the same set of samples as
        # another already-found sub-type
        if args.verbose:
            print("Found {} new sub-types!".format(len(sub_mtypes)))

        if len(sub_mtypes) < 7500:
            add_mtypes = set()

            for i, mtype in enumerate(sub_mtypes):
                if args.verbose and (i % 200) == 100:
                    print("Checked {} sub-types...".format(i))

                if mtype_sampsets[mtype] in use_sampsets:
                    if args.verbose:
                        print("Removing functionally duplicate MuType {}"
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
    pickle.dump(list(use_mtypes),
                open(os.path.join(out_path, 'tmp/mtype_list.p'), 'wb'))


if __name__ == '__main__':
    main()

