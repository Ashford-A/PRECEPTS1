
import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots')
sys.path.extend([os.path.join(base_dir, '../../..')])

import HetMan.experiments.utilities import depict_output
from HetMan.features.variants import MuType

import numpy as np
import pandas as pd
import argparse

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


def plot_coef_heatmap(coef_data, args, auc_cutoff=None, acc_data=None):

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

    if use_data.shape[1] > 100:
        use_genes = set(
            gene_means.sort_values(ascending=False)[:75].index.tolist()
            + gene_vars.sort_values(ascending=False)[:75].index.tolist()
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

