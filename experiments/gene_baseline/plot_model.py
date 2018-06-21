
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
from HetMan.experiments.utilities.scatter_plotting import place_annot

import argparse
import synapseclient
import numpy as np
import pandas as pd

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
                     '{}__auc-distribution__{}-{}_samps-{}.png'.format(
                         args.model_name, args.expr_source,
                         args.cohort, args.samp_cutoff
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
    aupr_vals = acc_df['AUPR'].quantile(q=0.25, axis=1)

    ax.scatter(mtype_sizes, acc_df['AUPR'].quantile(q=0.25, axis=1),
               s=15, c='black', alpha=0.47)

    annot_placed = place_annot(mtype_sizes, aupr_vals.values.tolist(),
                               size_vec=[15 for _ in mtype_sizes],
                               annot_vec=aupr_vals.index, x_range=1, y_range=1)

    for annot_x, annot_y, annot, halign in annot_placed:
        ax.text(annot_x, annot_y, annot, size=11, ha=halign)

    plt.plot([-1, 2], [-1, 2],
             linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.xlabel('Proportion of {} Samples Mutated'.format(args.cohort),
               fontsize=21, weight='semibold')
    plt.ylabel('1st Quartile AUPR', fontsize=21, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     '{}__aupr-quartile__{}-{}_samps-{}.png'.format(
                         args.model_name, args.expr_source,
                         args.cohort, args.samp_cutoff
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_distribution(par_df, acc_df, use_clf, args, cdata):
    fig = plt.figure(figsize=(11, 5 * len(use_clf.tune_priors)))
    plt_grid = plt.GridSpec(len(use_clf.tune_priors), 4,
                            wspace=0.1, hspace=0.3)
    
    for i, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        if isinstance(tune_distr, tuple):
            main_ax = fig.add_subplot(plt_grid[i, :])

            use_df = pd.DataFrame({'Acc': acc_df['AUC'].values.flatten(),
                                   'Par': par_df[par_name].values.flatten()})
            use_df['Acc'] += np.random.normal(loc=0.0, scale=1e-4,
                                              size=use_df.shape[0])

            sns.violinplot(data=use_df, x='Par', y='Acc', ax=main_ax,
                           order=tune_distr, cut=0, scale='count',
                           linewidth=1.7)
 
            main_ax.axhline(y=0.5, xmin=-2, xmax=len(tune_distr),
                            color='#550000', linewidth=2.9, alpha=0.32)

            main_ax.tick_params(labelsize=13)
            main_ax.set_xticklabels(['{:.1e}'.format(par)
                                     for par in tune_distr])

            main_ax.tick_params(axis='x', labelrotation=38)
            for label in main_ax.get_xticklabels():
                label.set_horizontalalignment('right')

            main_ax.set_xlabel('Tuned Hyper-Parameter Value',
                               size=17, weight='semibold')
            main_ax.set_ylabel('AUC', size=17, weight='semibold')

        elif tune_distr.dist.name == 'lognorm':
            main_ax = fig.add_subplot(plt_grid[:-1, i])
            plt.ylabel('AUC', size=16)

            sns.kdeplot(np.log10(par_df[par_name].values.flatten()),
                        acc_df['AUC'].values.flatten(),
                        ax=main_ax, gridsize=250, n_levels=23,
                        linewidths=0.9, alpha=0.5)

            dist_ax = fig.add_subplot(plt_grid[-1, i])
            dist_ax.hist(np.log10(tune_distr.rvs(100000)), bins=100,
                         normed=True, histtype='stepfilled', alpha=0.6)
            dist_ax.set_xlim(*np.log10(tune_distr.interval(0.9999)))
            plt.ylabel('Density', size=16)

            plt.xlabel('Hyper-Parameter Value', size=17)
            dist_ax.set_xticklabels(["$10^{" + str(int(x)) + "}$"
                                     for x in dist_ax.get_xticks()])

            main_ax.set_xlim(*np.log10(tune_distr.interval(0.9999)))
            main_ax.xaxis.set_ticklabels([])

        main_ax.set_title(par_name, size=21, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     '{}__tuning-distribution__{}-{}_samps-{}.png'.format(
                         args.model_name, args.expr_source,
                         args.cohort, args.samp_cutoff
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_gene(par_df, acc_df, use_clf, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 12 * len(use_clf.tune_priors)),
                              nrows=1, ncols=len(use_clf.tune_priors),
                              squeeze=False)

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):

        par_vals = np.log10(par_df[par_name].groupby(level=0).median())
        acc_vals = acc_df['AUC'].quantile(q=0.25, axis=1)
        size_vec = [1073 * len(cdata.train_mut[gene]) / len(cdata.samples)
                    for gene in acc_vals.index]

        plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
        plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])
        par_vals += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tune_distr) * 19), acc_df.shape[0])

        ax.scatter(par_vals, acc_vals, s=size_vec, c='black', alpha=0.23)
        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0, 1)

        ax.axhline(y=0.5, xmin=plt_xmin - 1, xmax=plt_xmax + 1,
                   color='#550000', linewidth=3.1, linestyle='--', alpha=0.32)
        annot_placed = place_annot(
            par_vals, acc_vals.values.tolist(), size_vec=size_vec,
            annot_vec=acc_vals.index, x_range=plt_xmax - plt_xmin, y_range=1
            )
 
        for annot_x, annot_y, annot, halign in annot_placed:
            ax.text(annot_x, annot_y, annot, size=11, ha=halign)

        ax.set_xlabel('Median Tuned {} Value'.format(par_name),
                      fontsize=21, weight='semibold')
        ax.set_ylabel('1st Quartile AUC', fontsize=21, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     '{}__tuning-gene__{}-{}_samps-{}.png'.format(
                         args.model_name, args.expr_source,
                         args.cohort, args.samp_cutoff
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
    plot_tuning_distribution(par_df, acc_df, mut_clf, args, cdata)
    plot_tuning_gene(par_df, acc_df, mut_clf, args, cdata)


if __name__ == "__main__":
    main()

