
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'model')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.gene_baseline.fit_tests import load_output
from HetMan.experiments.gene_baseline.setup_tests import get_cohort_data
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.utilities.scatter_plotting import place_annot

import argparse
import numpy as np
import pandas as pd
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def detect_log_distr(tune_distr):
    distr_diff = np.array(tune_distr[1:]) - np.array(tune_distr[:-1])
    diff_min = np.log10(np.min(distr_diff))
    diff_max = np.log10(np.max(distr_diff))

    return (diff_max - diff_min) > 1


def plot_auc_distribution(acc_df, args, cdata):
    fig, ax = plt.subplots(figsize=(acc_df.shape[0] / 6 + 2, 11))

    auc_means = acc_df['AUC'].mean(axis=1).sort_values(ascending=False)
    auc_clrs = auc_means.apply(auc_cmap)
    flier_props = dict(marker='o', markerfacecolor='black', markersize=4,
                       markeredgecolor='none', alpha=0.4)

    sns.boxplot(data=acc_df['AUC'].transpose(), order=auc_means.index,
                palette=auc_clrs, linewidth=1.7, boxprops=dict(alpha=0.68),
                flierprops=flier_props)
 
    plt.axhline(color='#550000', y=0.5, linewidth=3.7, alpha=0.32)
    plt.xticks(rotation=90, ha='right', size=12)
    plt.yticks(np.linspace(0, 1, 11), size=17)
    plt.ylabel('AUC', fontsize=26, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, args.model_name.split('__')[0],
                     '{}__auc-distribution__{}-{}_samps-{}.png'.format(
                         args.model_name.split('__')[1], args.expr_source,
                         args.cohort, args.samp_cutoff
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_acc_quartiles(acc_df, args, cdata):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, (ax_auc, ax_aupr) = plt.subplots(figsize=(22, 10), ncols=2)

    mtype_sizes = [len(cdata.train_mut[gene]) / len(cdata.samples)
                   for gene in acc_df.index]
    auc_vals = acc_df['AUC'].quantile(q=0.25, axis=1)
    aupr_vals = acc_df['AUPR'].quantile(q=0.25, axis=1)

    ax_auc.scatter(mtype_sizes, auc_vals, s=15, c='black', alpha=0.47)
    ax_aupr.scatter(mtype_sizes, aupr_vals, s=15, c='black', alpha=0.47)

    auc_annot = place_annot(mtype_sizes, auc_vals.values.tolist(),
                            size_vec=[15 for _ in mtype_sizes],
                            annot_vec=aupr_vals.index, x_range=1, y_range=1)
    for annot_x, annot_y, annot, halign in auc_annot:
        ax_auc.text(annot_x, annot_y, annot, size=11, ha=halign)

    aupr_annot = place_annot(mtype_sizes, aupr_vals.values.tolist(),
                             size_vec=[15 for _ in mtype_sizes],
                             annot_vec=aupr_vals.index, x_range=1, y_range=1)
    for annot_x, annot_y, annot, halign in aupr_annot:
        ax_aupr.text(annot_x, annot_y, annot, size=11, ha=halign)

    for ax in (ax_auc, ax_aupr):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax_auc.plot([-1, 2], [0.5, 0.5],
                linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)
    ax_aupr.plot([-1, 2], [-1, 2],
                 linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)

    fig.text(0.5, -0.03,
             'Proportion of {} Samples Mutated'.format(args.cohort),
             ha='center', va='center', fontsize=22, weight='semibold')
    ax_auc.set_ylabel('1st Quartile AUC', fontsize=22, weight='semibold')
    ax_aupr.set_ylabel('1st Quartile AUPR', fontsize=22, weight='semibold')

    fig.tight_layout(w_pad=2.2, h_pad=5.1)
    fig.savefig(
        os.path.join(plot_dir, args.model_name.split('__')[0],
                     '{}__acc-quartiles__{}-{}_samps-{}.png'.format(
                         args.model_name.split('__')[1], args.expr_source,
                         args.cohort, args.samp_cutoff
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_distribution(par_df, acc_df, use_clf, args, cdata):
    fig, axarr = plt.subplots(
        figsize=(17, 0.3 + 7 * len(use_clf.tune_priors)),
        nrows=len(use_clf.tune_priors), ncols=1, squeeze=False
        )

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        ax.set_title(par_name, size=29, weight='semibold')

        use_df = pd.DataFrame({'Acc': acc_df['AUC'].values.flatten(),
                               'Par': par_df[par_name].values.flatten()})
        use_df['Acc'] += np.random.normal(loc=0.0, scale=1e-4,
                                          size=use_df.shape[0])
 
        sns.violinplot(data=use_df, x='Par', y='Acc', ax=ax, order=tune_distr,
                       cut=0, scale='count', linewidth=1.7)

        ax.axhline(y=0.5, color='#550000', linewidth=2.9, alpha=0.32)
        ax.set_xticklabels(['{:.1e}'.format(par) for par in tune_distr])

        ax.tick_params(labelsize=18)
        ax.set_xlabel("")
        ax.set_ylabel("")
 
        ax.tick_params(axis='x', labelrotation=38)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    ax.set_xlabel("Tuned Hyper-Parameter Value", size=26, weight='semibold')
    fig.text(-0.01, 0.5, 'AUC', ha='center', va='center',
             fontsize=26, weight='semibold', rotation='vertical')

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, args.model_name.split('__')[0],
                     '{}__tuning-distribution__{}-{}_samps-{}.png'.format(
                         args.model_name.split('__')[1], args.expr_source,
                         args.cohort, args.samp_cutoff
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_gene(par_df, acc_df, use_clf, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 12 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)
    
    acc_vals = acc_df['AUC'].quantile(q=0.25, axis=1)
    size_vec = [1073 * len(cdata.train_mut[gene]) / len(cdata.samples)
                for gene in acc_vals.index]

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        par_vals = par_df[par_name].groupby(level=0).median()

        if detect_log_distr(tune_distr):
            par_vals = np.log10(par_vals)
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        par_vals += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tune_distr) * 19), acc_df.shape[0])
        ax.scatter(par_vals, acc_vals, s=size_vec, c='black', alpha=0.23)

        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=3.1, linestyle='--', alpha=0.32)

        annot_placed = place_annot(
            par_vals, acc_vals.values.tolist(), size_vec=size_vec,
            annot_vec=acc_vals.index, x_range=plt_xmax - plt_xmin, y_range=1
            )
 
        for annot_x, annot_y, annot, halign in annot_placed:
            ax.text(annot_x, annot_y, annot, size=11, ha=halign)

        ax.set_xlabel('Median Tuned {} Value'.format(par_name),
                      fontsize=26, weight='semibold')
        ax.set_ylabel('1st Quartile AUC', fontsize=26, weight='semibold')

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, args.model_name.split('__')[0],
                     '{}__tuning-gene__{}-{}_samps-{}.png'.format(
                         args.model_name.split('__')[1], args.expr_source,
                         args.cohort, args.samp_cutoff
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_gene_grid(par_df, acc_df, use_clf, args, cdata):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    acc_vals = acc_df['AUC'].quantile(q=0.25, axis=1)
    acc_clrs = acc_vals.apply(auc_cmap)
    size_vec = [493 * len(cdata.train_mut[gene]) /
                (len(cdata.samples) * par_count)
                for gene in acc_vals.index]

    for i, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        axarr[i, i].grid(False)

        if detect_log_distr(tune_distr):
            use_distr = [np.log10(par_val) for par_val in tune_distr]
        else:
            use_distr = tune_distr

        plt_min = 2 * use_distr[0] - use_distr[1]
        plt_max = 2 * use_distr[-1] - use_distr[-2]
        axarr[i, i].set_xlim(plt_min, plt_max)
        axarr[i, i].set_ylim(plt_min, plt_max)

        title_pos = (max(use_distr) + min(use_distr)) / 2
        axarr[i, i].text(title_pos, title_pos, par_name,
                         ha='center', fontsize=28, weight='semibold')

        for par_val in use_distr:
            axarr[i, i].axhline(y=par_val, color='#550000',
                                ls='--', linewidth=4.1, alpha=0.27)
            axarr[i, i].axvline(x=par_val, color='#550000',
                                ls='--', linewidth=4.1, alpha=0.27)

    for (i, (par_name1, tn_distr1)), (j, (par_name2, tn_distr2)) in combn(
            enumerate(use_clf.tune_priors), 2):

        if detect_log_distr(tn_distr1):
            par_meds1 = np.log10(par_df[par_name1]).groupby(level=0).median()
            par_means1 = np.log10(par_df[par_name1]).groupby(level=0).mean()
            plt_ymin = 2 * np.log10(tn_distr1[0]) - np.log10(tn_distr1[1])
            plt_ymax = 2 * np.log10(tn_distr1[-1]) - np.log10(tn_distr1[-2])

        else:
            par_meds1 = par_df[par_name1].groupby(level=0).median()
            par_means1 = par_df[par_name1].groupby(level=0).mean()
            plt_ymin = 2 * tn_distr1[0] - tn_distr1[1]
            plt_ymax = 2 * tn_distr1[-1] - tn_distr1[-2]

        if detect_log_distr(tn_distr2):
            par_meds2 = np.log10(par_df[par_name2]).groupby(level=0).median()
            par_means2 = np.log10(par_df[par_name2]).groupby(level=0).mean()
            plt_xmin = 2 * np.log10(tn_distr2[0]) - np.log10(tn_distr2[1])
            plt_xmax = 2 * np.log10(tn_distr2[-1]) - np.log10(tn_distr2[-2])

        else:
            par_meds2 = par_df[par_name2].groupby(level=0).median()
            par_means2 = par_df[par_name2].groupby(level=0).mean()
            plt_xmin = 2 * tn_distr2[0] - tn_distr2[1]
            plt_xmax = 2 * tn_distr2[-1] - tn_distr2[-2]

        par_meds1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 19), acc_df.shape[0])
        par_meds2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 19), acc_df.shape[0])

        par_means1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 19), acc_df.shape[0])
        par_means2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 19), acc_df.shape[0])

        axarr[i, j].scatter(par_meds2, par_meds1, s=size_vec, c=acc_clrs,
                            alpha=0.29, edgecolor='black')
        axarr[i, j].set_xlim(plt_xmin, plt_xmax)
        axarr[i, j].set_ylim(plt_ymin, plt_ymax)

        axarr[j, i].scatter(par_means1, par_means2, s=size_vec, c=acc_clrs,
                            alpha=0.29, edgecolor='black')
        axarr[j, i].set_ylim(plt_xmin, plt_xmax)
        axarr[j, i].set_xlim(plt_ymin, plt_ymax)

        annot_placed = place_annot(par_meds2, par_meds1,
                                   size_vec=size_vec,
                                   annot_vec=acc_vals.index,
                                   x_range=plt_xmax - plt_xmin,
                                   y_range=plt_ymax - plt_ymin)
 
        for annot_x, annot_y, annot, halign in annot_placed:
            axarr[i, j].text(annot_x, annot_y, annot, size=11, ha=halign)

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, args.model_name.split('__')[0],
                     '{}__tuning-gene-grid__{}-{}_samps-{}.png'.format(
                         args.model_name.split('__')[1], args.expr_source,
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
    os.makedirs(os.path.join(plot_dir, args.model_name.split('__')[0]),
                exist_ok=True)

    cdata = get_cohort_data(args.expr_source, args.syn_root,
                            args.cohort, args.samp_cutoff)
    acc_df, time_df, par_df, mut_clf = load_output(
        args.expr_source, args.cohort, args.samp_cutoff, args.model_name)

    plot_auc_distribution(acc_df, args, cdata)
    plot_acc_quartiles(acc_df, args, cdata)
    plot_tuning_distribution(par_df, acc_df, mut_clf, args, cdata)
    plot_tuning_gene(par_df, acc_df, mut_clf, args, cdata)

    if len(mut_clf.tune_priors) > 1:
        plot_tuning_gene_grid(par_df, acc_df, mut_clf, args, cdata)


if __name__ == "__main__":
    main()

