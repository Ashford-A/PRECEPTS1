
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_tour import *
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.cohorts.beatAML import BeatAmlCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_
from itertools import combinations as combn
from itertools import product


def get_cohort_data(cohort, expr_source, mut_levels):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                'FOUNDATION ONE', 'MSK-IMPACT']]
            == 'Yes').sum(axis=1) >= 1
        ]

    if cohort == 'beatAML':
        if expr_source != 'toil__gns':
            raise ValueError("Only gene-level Kallisto calls are available "
                             "for the beatAML cohort!")

        cdata = BeatAmlCohort(mut_levels=['Gene'] + list(mut_levels),
                              mut_genes=use_genes.tolist(),
                              expr_source=expr_source,
                              expr_file=beatAML_files['expr'],
                              samp_file=beatAML_files['samps'], syn=syn,
                              annot_file=annot_file, cv_seed=8713, test_prop=0)

    else:
        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        cdata = MutationCohort(cohort=cohort.split('_')[0],
                               mut_levels=['Gene'] + list(mut_levels),
                               mut_genes=use_genes.tolist(),
                               expr_source=source_base, var_source='mc3',
                               copy_source='Firehose', annot_file=annot_file,
                               domain_dir=domain_dir, type_file=type_file,
                               expr_dir=expr_sources[source_base],
                               copy_dir=copy_dir, collapse_txs=collapse_txs,
                               syn=syn, cv_seed=8713, test_prop=0,
                               annot_fields=['transcript'],
                               use_types=parse_subtypes(cohort))

    return cdata


def main():
    parser = argparse.ArgumentParser(
        "Set up the gene subtype expression effect isolation experiment by "
        "enumerating the subtypes to be tested."
        )

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of affected samples needed to test a mutation"
        )

    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
    parser.add_argument('out_dir', type=str, default=base_dir)

    # parse command line arguments
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    use_lvls = args.mut_levels.split('__')

    cdata = get_cohort_data(args.cohort, args.expr_source, use_lvls)
    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f)

    use_mtypes = set()
    for gene, muts in cdata.mtree:
        if len(pnt_mtype.get_samples(muts)) >= args.samp_cutoff:
            gene_mtypes = {
                mtype for mtype in muts['Point'].combtypes(
                    comb_sizes=(1, 2, 3), min_type_size=args.samp_cutoff)
                if (args.samp_cutoff <= len(mtype.get_samples(muts))
                    <= (len(cdata.get_samples()) - args.samp_cutoff))
                }

            gene_mtypes -= {
                mtype1 for mtype1, mtype2 in product(gene_mtypes, repeat=2)
                if mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                and (mtype1.get_samples(cdata.mtree)
                     == mtype2.get_samples(cdata.mtree))
                }

            if args.mut_levels == 'Exon__Location__Protein':
                gene_mtypes |= {pnt_mtype}

            use_mtypes |= {MuType({('Gene', gene): mtype})
                           for mtype in gene_mtypes}

    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))


if __name__ == '__main__':
    main()

