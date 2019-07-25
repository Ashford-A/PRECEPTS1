
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.copy_baseline import *
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import CopyCohort, list_cohorts
from HetMan.features.data.copies import get_copies_firehose

import argparse
import pandas as pd
import dill as pickle
import random


def get_cohort_data(cohort, expr_source, samp_cutoff, cv_seed=None):
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
                       cv_seed=cv_seed, test_prop=0,
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

    parser.add_argument('out_dir', type=str)
    args = parser.parse_args()
    base_path = os.path.join(args.out_dir.split('copy_baseline')[0],
                             'copy_baseline')
    coh_path = os.path.join(base_path, args.expr_source, 'setup')

    out_tag = "{}__samps-{}".format(args.cohort, args.samp_cutoff)
    gene_path = os.path.join(base_path, args.expr_source, out_tag, 'setup')
    out_path = os.path.join(args.out_dir, 'setup')

    coh_list = list_cohorts(
        args.expr_source.split('__')[0],
        expr_dir=expr_sources[args.expr_source.split('__')[0]],
        copy_dir=copy_dir
        ) | {args.cohort}

    use_feats = None
    for coh in random.sample(coh_list, k=len(coh_list)):
        coh_tag = "{}__cohort-data.p".format(coh)

        if coh == args.cohort:
            copy_tag = "cohort-data.p"
        else:
            copy_tag = "{}__cohort-data.p".format(coh)

        if os.path.exists(os.path.join(coh_path, coh_tag)):
            os.system("cp {} {}".format(os.path.join(coh_path, coh_tag),
                                        os.path.join(out_path, copy_tag)))

            with open(os.path.join(out_path, copy_tag), 'rb') as f:
                cdata = pickle.load(f)

        else:
            cdata = get_cohort_data(coh, args.expr_source, args.samp_cutoff)

            with open(os.path.join(coh_path, coh_tag), 'wb') as f:
                pickle.dump(cdata, f)
            with open(os.path.join(out_path, copy_tag), 'wb') as f:
                pickle.dump(cdata, f)

        if use_feats is None:
            use_feats = set(cdata.get_features())
        else:
            use_feats &= set(cdata.get_features())

    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f)

    if os.path.exists(os.path.join(gene_path, "gene-list.p")):
        os.system("cp {} {}".format(os.path.join(gene_path, "gene-list.p"),
                                    os.path.join(out_path, "gene-list.p")))

    else:
        with open(os.path.join(out_path, "cohort-data.p".format(args.cohort)),
                  'rb') as f:
            cdata = pickle.load(f)

        # save the enumerated altered genes, and the number of such genes, to file
        with open(os.path.join(out_path, "gene-list.p"), 'wb') as fl:
            pickle.dump(sorted(cdata.copy_mat.columns), fl)
        with open(os.path.join(out_path, "gene-count.txt"), 'w') as f:
            f.write(str(cdata.copy_mat.shape[1]))

    if os.path.exists(os.path.join(gene_path, "gene-count.txt")):
        os.system("cp {} {}".format(os.path.join(gene_path, "gene-count.txt"),
                                    os.path.join(out_path, "gene-count.txt")))

    else:
        with open(os.path.join(out_path, "gene-list.p"), 'rb') as f:
            gene_list = pickle.load(f)

        with open(os.path.join(gene_path, "gene-count.txt"), 'w') as f:
            f.write(str(len(gene_list)))
        with open(os.path.join(out_path, "gene-count.txt"), 'w') as f:
            f.write(str(len(gene_list)))


if __name__ == '__main__':
    main()

