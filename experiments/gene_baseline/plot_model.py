
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'model')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

from HetMan.experiments.gene_baseline.fit_tests import load_output
from HetMan.experiments.gene_baseline.setup_tests import get_cohort_data
from HetMan.experiments.utilities import auc_cmap

import argparse
import synapseclient
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_auc_distribution(acc_df, args, cdata):
    fig, ax = plt.subplots(figsize=(15, 14))

    auc_means = acc_df['AUC'].mean(axis=1).sort_values(ascending=False)
    auc_clrs = auc_means.apply(auc_cmap)
    flier_props = dict(marker='o', markerfacecolor='black', markersize=6,
                       markeredgecolor='none', alpha=0.4)

    sns.boxplot(data=acc_df['AUC'].transpose(), order=auc_means.index,
                palette=auc_clrs, linewidth=1.7, boxprops=dict(alpha=0.68),
                flierprops=flier_props)
 
    plt.axhline(color='#550000', y=0.5, xmin=-2, xmax=acc_df.shape[0] + 2,
                linewidth=3.7, alpha=0.32)

    plt.xticks(rotation=38, ha='right', size=11)
    plt.ylabel('AUC', fontsize=26, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     'auc-distribution__{}-{}_{}-{}.png'.format(
                         args.expr_source, args.cohort,
                         args.cohort, args.model_name
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_aupr_quartile(acc_df, args, cdata):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'

    fig, ax = plt.subplots(figsize=(13, 12))
    
    mtype_sizes = [len(cdata.train_mut[gene]) / len(cdata.samples)
                   for gene in acc_df.index]

    ax.scatter(mtype_sizes, acc_df['AUPR'].quantile(q=0.25, axis=1),
               s=21, c='black', alpha=0.32)

    plt.plot([-1, 2], [-1, 2],
             linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.xlabel('Proportion of {} Samples Mutated'.format(args.cohort),
               fontsize=21, weight='semibold')
    plt.ylabel('1st Quartile AUPR', fontsize=21, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     'aupr-quartile__{}-{}_{}-{}.png'.format(
                         args.expr_source, args.cohort,
                         args.cohort, args.model_name
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance and tuning characteristics of a model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('expr_source', type=str, choices=['Firehose', 'toil'],
                        help="which TCGA expression data source was used")
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")

    parser.add_argument(
        'syn_root', type=str,
        help="the root cache directory for data downloaded from Synapse"
        )

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    cdata = get_cohort_data(args.expr_source, args.syn_root,
                            args.cohort, args.samp_cutoff)
    acc_df, time_df, par_df, mut_clf = load_output(
        args.expr_source, args.cohort, args.samp_cutoff, args.model_name)

    plot_auc_distribution(acc_df, args, cdata)
    plot_aupr_quartile(acc_df, args, cdata)


if __name__ == "__main__":
    main()

