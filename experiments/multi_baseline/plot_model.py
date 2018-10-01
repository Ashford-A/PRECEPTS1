
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'model')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.multi_baseline import *
from HetMan.experiments.multi_baseline.fit_tests import load_output
from HetMan.experiments.multi_baseline.setup_tests import get_cohort_data
from HetMan.experiments.mut_baseline.plot_model import detect_log_distr

from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.utilities.scatter_plotting import place_annot

import argparse
import numpy as np
import pandas as pd
from itertools import combinations as combn
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_auc_quartiles(auc_df, args):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, ax = plt.subplots(figsize=(14, 13))

    test_aucs = auc_df.applymap(itemgetter('test'))
    quart_df = pd.DataFrame(index=auc_df.index, columns=['Min', 'Max'])
    new_indx = ['' for _ in auc_df.index]

    for i, (mtypes, auc_dicts) in enumerate(test_aucs.iterrows()):
        auc_quants = pd.DataFrame(np.stack(auc_dicts.values)).quantile(q=0.25)

        if auc_quants[0] >= auc_quants[1]:
            new_indx[i] = ' + '.join([str(mtype) for mtype in mtypes])
        else:
            new_indx[i] = ' + '.join([str(mtype) for mtype in mtypes[::-1]])

        quart_df.iloc[i, :] = auc_quants.sort_values().values

    quart_df.index = new_indx
    plot_min = np.min(quart_df.values) - 0.01
    ax.scatter(quart_df.Min, quart_df.Max, s=15, c='black', alpha=0.47)

    for annot_x, annot_y, annot, halign in place_annot(
            quart_df.Min.tolist(), quart_df.Max.tolist(),
            size_vec=[15 for _ in auc_df.index], annot_vec=quart_df.index,
            x_range=1 - plot_min, y_range=1 - plot_min, gap_adj=79
            ):
        ax.text(annot_x, annot_y, annot, size=11, ha=halign)

    ax.tick_params(pad=5.1)
    ax.set_xlim(plot_min, 1)
    ax.set_ylim(plot_min, 1)

    ax.set_xlabel('1st Qrt. AUC, min mut.', fontsize=22, weight='semibold')
    ax.set_ylabel('1st Qrt. AUC, max mut.', fontsize=22, weight='semibold')
    ax.plot([-1, 2], [-1, 2],
            linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)

    fig.savefig(
        os.path.join(plot_dir, '{}__{}'.format(args.expr_source, args.cohort),
                     args.model_name.split('__')[0],
                     '{}__acc-quartiles.png'.format(
                         args.model_name.split('__')[1])),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_mtype_grid(par_df, auc_df, use_clf, args):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(0.5 + 7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    auc_vals = auc_df.apply(
        lambda x: pd.DataFrame(y['test'] for y in x).quantile(q=0.25).min(),
        axis=1
        )
    auc_clrs = auc_vals.apply(auc_cmap)

    for i, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        axarr[i, i].grid(False)

        if detect_log_distr(tune_distr):
            use_distr = [np.log10(par_val) for par_val in tune_distr]
            par_lbl = par_name + '\n(log-scale)'

        else:
            use_distr = tune_distr
            par_lbl = par_name

        distr_diff = np.mean(np.array(use_distr[1:])
                             - np.array(use_distr[:-1]))
        plt_min = use_distr[0] - distr_diff / 2
        plt_max = use_distr[-1] + distr_diff / 2

        axarr[i, i].set_xlim(plt_min, plt_max)
        axarr[i, i].set_ylim(plt_min, plt_max)
        axarr[i, i].text(
            (plt_min + plt_max) / 2, (plt_min + plt_max) / 2, par_lbl,
            ha='center', fontsize=28, weight='semibold'
            )

        for par_val in use_distr:
            axarr[i, i].axhline(y=par_val, color='#116611',
                                ls='--', linewidth=4.1, alpha=0.27)
            axarr[i, i].axvline(x=par_val, color='#116611',
                                ls='--', linewidth=4.1, alpha=0.27)

    for (i, (par_name1, tn_distr1)), (j, (par_name2, tn_distr2)) in combn(
            enumerate(use_clf.tune_priors), 2):

        if detect_log_distr(tn_distr1):
            par_meds1 = np.log10(par_df[par_name1]).groupby(
                level=[0, 1]).median()
            par_means1 = np.log10(par_df[par_name1]).groupby(
                level=[0, 1]).mean()
            
            distr_diff = np.mean(np.log10(np.array(tn_distr1[1:]))
                                 - np.log10(np.array(tn_distr1[:-1])))
            plt_ymin = np.log10(tn_distr1[0]) - distr_diff / 2
            plt_ymax = np.log10(tn_distr1[-1]) + distr_diff / 2

        else:
            par_meds1 = par_df[par_name1].groupby(level=[0, 1]).median()
            par_means1 = par_df[par_name1].groupby(level=[0, 1]).mean()

            distr_diff = np.mean(np.array(tn_distr1[1:])
                                 - np.array(tn_distr1[:-1]))
            plt_ymin = tn_distr1[0] - distr_diff / 2
            plt_ymax = tn_distr1[-1] + distr_diff / 2

        if detect_log_distr(tn_distr2):
            par_meds2 = np.log10(par_df[par_name2]).groupby(
                level=[0, 1]).median()
            par_means2 = np.log10(par_df[par_name2]).groupby(
                level=[0, 1]).mean()

            distr_diff = np.mean(np.log10(np.array(tn_distr2[1:]))
                                 - np.log10(np.array(tn_distr2[:-1])))
            plt_xmin = np.log10(tn_distr2[0]) - distr_diff / 2
            plt_xmax = np.log10(tn_distr2[-1]) + distr_diff / 2

        else:
            par_meds2 = par_df[par_name2].groupby(level=[0, 1]).median()
            par_means2 = par_df[par_name2].groupby(level=[0, 1]).mean()

            distr_diff = np.mean(np.array(tn_distr2[1:])
                                 - np.array(tn_distr2[:-1]))
            plt_xmin = tn_distr2[0] - distr_diff / 2
            plt_xmax = tn_distr2[-1] + distr_diff / 2

        par_meds1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 17), auc_df.shape[0])
        par_meds2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 17), auc_df.shape[0])

        par_means1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 23), auc_df.shape[0])
        par_means2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 23), auc_df.shape[0])

        axarr[i, j].scatter(
            par_meds2[auc_clrs.index], par_meds1[auc_clrs.index],
            s=17, c=auc_clrs, alpha=0.43, edgecolor='black'
            )

        axarr[j, i].scatter(
            par_means1[auc_clrs.index], par_means2[auc_clrs.index],
            s=17, c=auc_clrs, alpha=0.43, edgecolor='black'
            )

        axarr[i, j].set_xlim(plt_xmin, plt_xmax)
        axarr[i, j].set_ylim(plt_ymin, plt_ymax)
        axarr[j, i].set_ylim(plt_xmin, plt_xmax)
        axarr[j, i].set_xlim(plt_ymin, plt_ymax)

        #annot_placed = place_annot(par_meds2, par_meds1,
        #                           size_vec=[13 for _ in par_meds],
        #                           annot_vec=auc_vals.index,
        #                           x_range=plt_xmax - plt_xmin,
        #                           y_range=plt_ymax - plt_ymin)
 
        #for annot_x, annot_y, annot, halign in annot_placed:
        #    axarr[i, j].text(annot_x, annot_y, annot, size=11, ha=halign)

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, '{}__{}'.format(args.expr_source, args.cohort),
                     args.model_name.split('__')[0],
                     '{}__tuning-mtype.png'.format(
                         args.model_name.split('__')[1])),
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
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    os.makedirs(os.path.join(
        plot_dir, '{}__{}'.format(args.expr_source, args.cohort),
        args.model_name.split('__')[0]
        ), exist_ok=True
        )

    #cdata = get_cohort_data(args.expr_source, args.cohort, args.samp_cutoff)
    auc_df, aupr_df, time_df, par_df, mut_clf = load_output(
        args.expr_source, args.cohort, args.samp_cutoff, args.model_name)

    plot_auc_quartiles(auc_df, args)
    if len(mut_clf.tune_priors) > 1:
        plot_tuning_mtype_grid(par_df, auc_df, mut_clf, args)


if __name__ == "__main__":
    main()

