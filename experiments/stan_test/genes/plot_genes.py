
"""Plotting the results of testing Toil Kallisto expression data.

"""

import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots')
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.experiments.stan_test.genes.fit_models import load_accuracies
from HetMan.experiments.stan_test.genes.setup_baseline import (
    load_accuracies as load_base_accuracies)

from HetMan.features.variants import MuType
from HetMan.features.cohorts.mut import VariantCohort

import numpy as np
from scipy.stats import wilcoxon

import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'
toil_dir = '/home/exacloud/lustre1/CompBio/data/toil_tcga'


def plot_auc_comp(base_df, model_df, cdata, args):

    # turns on drawing borders around the edges of the plotting area
    mpl.rcParams['axes.linewidth'] = 2.7
    mpl.rcParams['axes.edgecolor'] = '0.59'

    # instantiates the plot, decides the file where it will be saved
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_fl = 'base-comp_{}-{}-{}-{}.png'.format(
        args.model_name, args.cohort, args.solve_type, args.baseline)

    # gets the 1st quartile of classifier performance on each input dataset
    base_acc = base_df.quantile(axis=1, q=0.25)
    model_acc = model_df.quantile(axis=1, q=0.25)
    plot_min = min(np.min(base_acc), np.min(model_acc)) - 0.01

    # gets the difference in performance according to the input dataset used,
    # calculates the statistical significane of this difference
    acc_diff = model_acc - base_acc
    pair_test = wilcoxon(acc_diff)[1]
    acc_delta = (acc_diff / base_acc).quantile(q=0.5)

    # initializes variables determining point label position
    lbl_offset = {mtype: None for mtype in base_df.index}
    lbl_align = {mtype: 'left' for mtype in base_df.index}

    for mtype in base_df.index:
        if not ((base_acc > base_acc[mtype])
                & (base_acc < base_acc[mtype] + 0.04)
                & (model_acc > model_acc[mtype] - 0.01)
                & (model_acc < model_acc[mtype] + 0.03)).any():
            lbl_offset[mtype] = (1, 1)

    for mtype in base_df.index:
        if not lbl_offset[mtype]:
            if not ((base_acc > base_acc[mtype])
                    & (base_acc < base_acc[mtype] + 0.04)
                    & (model_acc > model_acc[mtype] - 0.03)
                    & (model_acc < model_acc[mtype] + 0.01)).any():
                lbl_offset[mtype] = (1, -1)

    for mtype in base_df.index:
        if not lbl_offset[mtype]:
            if not ((base_acc > base_acc[mtype] - 0.04)
                    & (base_acc < base_acc[mtype])
                    & (model_acc > model_acc[mtype] - 0.01)
                    & (model_acc < model_acc[mtype] + 0.03)).any():
                lbl_offset[mtype] = (-1, 1)
                lbl_align[mtype] = 'right'

    for mtype in base_df.index:
        if not lbl_offset[mtype]:
            if not ((base_acc > base_acc[mtype] - 0.04)
                    & (base_acc < base_acc[mtype])
                    & (model_acc > model_acc[mtype] - 0.03)
                    & (model_acc < model_acc[mtype] + 0.01)).any():
                lbl_offset[mtype] = (-1, -1)
                lbl_align[mtype] = 'right'

    for mtype in base_df.index:
        mtype_size = (
            np.sum(np.array(cdata.train_pheno(mtype)))
            * (975 / len(cdata.samples))
            )

        ax.scatter(base_acc[mtype], model_acc[mtype],
                   s=mtype_size, c='black', marker='o', alpha=0.27)

        if lbl_offset[mtype]:
            ax.text(x=(base_acc[mtype]
                       + lbl_offset[mtype][0] * (mtype_size / 4e6) ** 0.5),
                    y=(model_acc[mtype]
                       + lbl_offset[mtype][1] * (mtype_size / 4e6) ** 0.5),
                    s=str(mtype), size=8, va='center', ha=lbl_align[mtype],
                    stretch=10)

    ax.text(x=0.82, y=0.52,
            s="median change: {:+.2%}\np-val: {:.2e}".format(
                acc_delta, pair_test),
            size=14)

    plt.axhline(color='r', y=0.5, xmin=-2, xmax=2,
                linewidth=1.1, linestyle=':')
    plt.axvline(color='r', x=0.5, ymin=-2, ymax=2,
                linewidth=1.1, linestyle=':')
    plt.plot([-1, 2], [-1, 2], linewidth=1.7, linestyle='--')

    # creates axis titles and labels, sets limits to ensure plot is square
    plt.xlabel('Baseline {} AUC'.format(args.baseline), fontsize=21)
    plt.ylabel("Stan '{}' Model AUC".format(args.model_name), fontsize=21)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(plot_min, 1)
    ax.set_ylim(plot_min, 1)

    # removes plot border on the labelled sides
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # sets the size of the plotting area and saves the plot to file
    fig.set_size_inches(10, 10)
    plt.savefig(os.path.join(plot_dir, plot_fl),
                dpi=500, bbox_inches='tight')
    plt.close()


def main():
    """Creates plots for a given combination of cohort and classifier."""
    parser = argparse.ArgumentParser()

    # parses command line arguments
    parser.add_argument('model_name', help='a Stan model')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('solve_type',
                        help='a way of solving the above Stan model')
    parser.add_argument('--baseline', default='Lasso',
                        help='algorithm to use for baseline comparison')
    args = parser.parse_args()

    # loads output of the experiment, extracts the genes that were tested
    base_df = load_base_accuracies(args.baseline, args.cohort)
    acc_df = load_accuracies(args.model_name, args.cohort, args.solve_type)
    use_genes = [mtype.subtype_list()[0][0] for mtype in acc_df.index]

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/mgrzad"
                                "/input-data/synapse")
    syn.login()

    # loads expression and mutation data using Firehose as a source
    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=use_genes, mut_levels=['Gene'],
        expr_source='Firehose', data_dir=firehose_dir, cv_prop=1.0, syn=syn
        )

    # creates plots
    plot_auc_comp(base_df, acc_df, cdata, args)


if __name__ == '__main__':
    main()

