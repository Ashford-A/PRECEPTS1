
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_isolate')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'mutex')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.subvariant_isolate.setup_isolate import load_cohort
from HetMan.experiments.subvariant_isolate.utils import compare_scores
from HetMan.experiments.utilities import load_infer_output

import argparse
import numpy as np
import pandas as pd

from itertools import combinations as combn
from scipy.stats import fisher_exact

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def plot_similarity_scatter(simil_df, auc_list, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10, 6))

    mutex_pvals = []
    simil_vals = []
    auc_vals = []
    size_vals = []

    for mtypes1, mtypes2 in combn(auc_list[auc_list > 0.6].index, 2):
        if (len(mtypes1) > 1 or len(mtypes2) > 1
                or (mtypes1[0] & mtypes2[0]).is_empty()):

            mutex_pvals += [
                -np.log10(fisher_exact(table=pd.crosstab(pheno_dict[mtypes1],
                                                         pheno_dict[mtypes2]),
                                       alternative='less')[1])
                ]

            siml_adj1 = np.clip(auc_list[mtypes1] - 0.5, 0, 1) ** 2
            siml_adj2 = np.clip(auc_list[mtypes2] - 0.5, 0, 1) ** 2

            siml_val = siml_adj1 * simil_df.loc[
                [mtypes1], [mtypes2]].iloc[0, 0]
            siml_val += siml_adj2 * simil_df.loc[
                [mtypes2], [mtypes1]].iloc[0, 0]
            simil_vals += [siml_val / (siml_adj1 + siml_adj2)]

            auc_vals += [max(siml_adj1, siml_adj2)]
            size_vals += [np.sum(pheno_dict[mtypes1])
                          + np.sum(pheno_dict[mtypes2])]

    for mutex_pval, simil_val, auc_val, size_val in zip(
            mutex_pvals, simil_vals, auc_vals, size_vals):
        ax.scatter(mutex_pval, simil_val, marker='o',
                   s=size_val / 13, alpha=(auc_val - 0.01) ** 0.61)

    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.xlabel("Mutual Exclusivity",
               size=23, weight='semibold')
    plt.ylabel("Inferred Similarity", size=23, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, '{}_{}'.format(args.cohort, args.gene),
        "simil-scatter__{}__samps_{}__{}.png".format(
            args.classif, args.samp_cutoff, args.mut_levels)
            ),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')

    parser.add_argument('mut_levels', default='Form_base__Exon',
                        help='a set of mutation annotation levels')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '{}_{}'.format(args.cohort, args.gene)),
                exist_ok=True)

    cdata = load_cohort(args.cohort, [args.gene], args.mut_levels.split('__'))
    simil_df, auc_list, pheno_dict = compare_scores(
        load_infer_output(
            os.path.join(base_dir, 'output',
                         args.cohort, args.gene, args.classif,
                         'samps_{}'.format(args.samp_cutoff), args.mut_levels)
            ), cdata
        )

    plot_similarity_scatter(simil_df.copy(), auc_list.copy(),
                            pheno_dict.copy(), args)


if __name__ == '__main__':
    main()

