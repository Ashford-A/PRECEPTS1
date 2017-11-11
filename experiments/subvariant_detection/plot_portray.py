
"""Plots the results of describing the classification signatures of sub-types.

Examples:
    plot_portray.py LUAD ElasticNet
    plot_portray.py BRCA Ridge
    plot_portray.py THCA GradBoost

"""

import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots', 'portray')
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities import depict_output
from HetMan.features.variants import MuType

import numpy as np
import pandas as pd
import argparse

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_coef_heatmap(coef_data, args, auc_cutoff=None, acc_data=None):

    if auc_cutoff is None != acc_data is None:
        raise ValueError(
            "If an AUC cutoff is specified, sub-type classification "
            "performance must be specified as well!"
            )

    if auc_cutoff is None:
        plot_file = "mtype-coefs_{}-{}.png".format(args.cohort, args.classif)
        use_mtypes = coef_data.index

    else:
        plot_file = "mtype-coefs_{}-{}_auc-cutoff-{}.png".format(
                args.cohort, args.classif, auc_cutoff)
        use_mtypes = acc_data.index[acc_data.mean(axis=1) > auc_cutoff]

    use_data = coef_data.loc[use_mtypes, :]
    use_data = (use_data.transpose() / use_data.abs().max(axis=1)).transpose()

    gene_means = use_data.abs().mean()
    gene_vars = use_data.var()

    if use_data.shape[1] > 120:
        use_genes = set(
            gene_means.sort_values(ascending=False)[:80].index.tolist()
            + gene_vars.sort_values(ascending=False)[:80].index.tolist()
            )

    else:
        use_genes = use_data.columns

    use_data = use_data.loc[:, use_genes]
    sns.set_context("paper")

    sig_plot = sns.clustermap(use_data,
                              method='centroid', center=0, cmap="coolwarm_r",
                              figsize=(33, 17))

    sig_plot.savefig(os.path.join(plot_dir, plot_file))


def main():
    """Creates plots for the given combination of cohort and classifier."""

    # parses command line arguments
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')
    args = parser.parse_args()

    acc_data, coef_data = depict_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'portray'))

    plot_coef_heatmap(coef_data, args)
    plot_coef_heatmap(coef_data, args, auc_cutoff=0.85, acc_data=acc_data)
    plot_coef_heatmap(coef_data, args, auc_cutoff=0.9, acc_data=acc_data)


if __name__ == '__main__':
    main()

