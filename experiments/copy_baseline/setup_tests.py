
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.copy_baseline import *
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import CopyCohort
from HetMan.features.data.copies import get_copies_firehose

import argparse
import pandas as pd
import dill as pickle


def get_cohort_data(cohort, expr_source, samp_cutoff,
                    cv_seed=None, test_prop=0):
    coh_base = cohort.split('_')[0]

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    copy_mat = get_copies_firehose(coh_base, copy_dir, discrete=True)

    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                'FOUNDATION ONE', 'MSK-IMPACT']]
            == 'Yes').sum(axis=1) > 1
        ]

    deep_copies = copy_mat.isin({-2, 2}).sum()
    use_genes &= set(deep_copies[deep_copies >= samp_cutoff].index)

    source_info = expr_source.split('__')
    src_base = source_info[0]
    collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

    cdata = CopyCohort(cohort=coh_base, copy_genes=use_genes.tolist(),
                       expr_source=src_base, var_source='mc3',
                       copy_source='Firehose', annot_file=annot_file,
                       type_file=type_file, expr_dir=expr_sources[src_base],
                       copy_dir=copy_dir, collapse_txs=collapse_txs,
                       cv_seed=cv_seed, test_prop=test_prop,
                       annot_fields=['transcript'],
                       use_types=parse_subtypes(cohort))

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of affected samples needed to test a mutation"
        )

    parser.add_argument('--setup_dir', type=str, default=base_dir)
    args = parser.parse_args()
    out_path = os.path.join(args.setup_dir, 'setup')

    cdata = get_cohort_data(args.cohort, args.expr_source, args.samp_cutoff)
    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f)

    # save the enumerated altered genes, and the number of such genes, to file
    with open(os.path.join(out_path, "gene-list.p"), 'wb') as fl:
        pickle.dump(sorted(cdata.copy_mat.columns), fl)
    with open(os.path.join(out_path, "gene-count.txt"), 'w') as f:
        f.write(str(cdata.copy_mat.shape[1]))


if __name__ == '__main__':
    main()

