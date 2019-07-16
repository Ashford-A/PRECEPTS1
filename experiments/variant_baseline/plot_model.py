
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'variant_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'model')

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.variant_baseline.plot_tuning import detect_log_distr

from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.utilities.scatter_plotting import place_annot
from sklearn.kernel_ridge import KernelRidge

import argparse
from pathlib import Path
import dill as pickle
import bz2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'

cv_clrs = {'random': '#806915', 'fivefold': '#0E5342', 'infer': '#5E104A'}


def plot_label_stability(score_dict, auc_df, args, cdata):
    fig, axarr = plt.subplots(figsize=(16, 9),
                              nrows=2, ncols=5, sharex=True, sharey=True)

    auc_bins = pd.qcut(auc_df.quantile(q=0.25, axis=1), 5)
    pred_vec = np.linspace(0, 1, 1000)
    pheno_df = pd.DataFrame({mtype: np.array(cdata.train_pheno(mtype))
                             for mtype in auc_bins.index},
                            index=sorted(cdata.get_train_samples()))

    stat_dict = {
        cv_mth: {
            'Mean': score_dict[cv_mth].applymap(
                lambda x: np.mean([y for y in x if y == y])),
            'Var': score_dict[cv_mth].applymap(
                lambda x: np.var([y for y in x if y == y], ddof=1))
            }
        for cv_mth in ['random', 'fivefold']
        }

    for i, (phn, phn_df) in enumerate([('WT', ~pheno_df), ('Mut', pheno_df)]):
        for j, (abin, auc_vals) in enumerate(auc_bins.groupby(by=auc_bins)):

            val_dict = {
                cv_mth: pd.DataFrame({
                    stat_lbl: stat_df[auc_vals.index].where(
                        phn_df.loc[stat_df.index,
                                   auc_vals.index]).melt().dropna().value
                    for stat_lbl, stat_df in cv_dict.items()
                    })
                for cv_mth, cv_dict in stat_dict.items()
                }

            val_dict = {cv_mth: val_df.loc[~val_df.isna().any(axis=1), :]
                        for cv_mth, val_df in val_dict.items()}

            clf_dict = {
                cv_mth: KernelRidge(alpha=1e-4, kernel='rbf').fit(
                    val_df.Mean.values.reshape(-1, 1),
                    val_df.Var.values.reshape(-1, 1)
                    )
                for cv_mth, val_df in val_dict.items()
                }

            for cv_mth, cv_clf in clf_dict.items():
                axarr[i, j].plot(pred_vec,
                                 cv_clf.predict(pred_vec.reshape(-1, 1)),
                                 color=cv_clrs[cv_mth], linewidth=2.3,
                                 alpha=0.61)

            axarr[i, j].set_ylim([0, 13/11])

    lgnd_ptchs = [Patch(color=cv_clrs['random'], alpha=0.51, label="random"),
                  Patch(color=cv_clrs['fivefold'],
                        alpha=0.51, label="five-fold")]

    fig.legend(handles=lgnd_ptchs, frameon=False, fontsize=23, ncol=2, loc=8,
               handletextpad=0.7, bbox_to_anchor=(0.5, -1/33))

    fig.tight_layout(w_pad=1.7, h_pad=2.3)
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__label-stability.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_label_correlation(score_dict, auc_df, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))

    score_vals = {
        'fold': score_dict['fivefold'].applymap(
            lambda x: [y for y in x if y == y]),
        'infer': score_dict['infer']
        }

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    corr_dict = {mtype: {'inter': [-1] * 5, 'intra': np.empty((5, 5))}
                 for mtype in score_dict['fivefold']}

    for mtype in score_dict['fivefold']:
        fold_vals = np.stack(score_vals['fold'][mtype])

        for i in range(5):
            corr_dict[mtype]['inter'][i] = spearmanr(
                fold_vals[:, i], score_vals['infer'][mtype]).correlation

            for j in set(range(5)) - set(range(i + 1)):
                corr_dict[mtype]['intra'][i, j] = spearmanr(
                    fold_vals[:, i], fold_vals[:, j]).correlation

        inter_corr = np.median(corr_dict[mtype]['inter'])
        intra_corr = np.median(
            corr_dict[mtype]['intra'][np.triu_indices(5, k=1)])

        ax.scatter(auc_vals[mtype], inter_corr, s=67, c=cv_clrs['infer'],
                   marker='o', alpha=0.37, edgecolors='none')
        ax.scatter(auc_vals[mtype], intra_corr, s=83, c=cv_clrs['fivefold'],
                   marker='o', alpha=0.37, edgecolors='none')

        if inter_corr > intra_corr:
            ax.axvline(auc_vals[mtype], ymin=intra_corr, ymax=inter_corr,
                       color=cv_clrs['infer'], alpha=0.23, linewidth=2.9)
        else:
            ax.axvline(auc_vals[mtype], ymin=inter_corr, ymax=intra_corr,
                       color=cv_clrs['fivefold'], alpha=0.23, linewidth=2.9)

    ax.set_ylim([0, 1])
    plt.xlabel('Median Correlation', fontsize=27, weight='semibold')
    plt.ylabel('AUC', fontsize=27, weight='semibold')

    lgnd_ptchs = [Patch(color=cv_clrs['fivefold'],
                        alpha=0.51, label="within folds"),
                  Patch(color=cv_clrs['infer'],
                        alpha=0.51, label="each fold with inferred")]

    fig.legend(handles=lgnd_ptchs, frameon=False, fontsize=23, ncol=2, loc=8,
               handletextpad=0.7, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(pad=2)
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__label-correlation.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_auc_distribution(auc_df, args):
    fig, ax = plt.subplots(figsize=(auc_df.shape[0] / 4.3 + 2, 11))

    auc_means = auc_df.mean(axis=1).sort_values(ascending=False)
    auc_clrs = auc_means.apply(auc_cmap)
    flier_props = dict(marker='o', markerfacecolor='black', markersize=4,
                       markeredgecolor='none', alpha=0.4)

    sns.boxplot(data=auc_df.transpose(), order=auc_means.index,
                palette=auc_clrs, linewidth=1.7, boxprops=dict(alpha=0.68),
                flierprops=flier_props)
 
    plt.axhline(color='#550000', y=0.5, linewidth=3.7, alpha=0.32)
    plt.ylabel('AUC', fontsize=26, weight='semibold')
    flr_locs = np.array([[ax.lines[i * 6]._yorig[1],
                          ax.lines[i * 6 + 1]._yorig[1]]
                         for i in range(len(auc_means))])

    plt.xticks([])
    plt.yticks(size=17)
    ax.tick_params(axis='y', length=11, width=2)

    for i, mtype in enumerate(auc_means.index):
        str_len = min(len(str(mtype)) // 3 + 2, 8)

        if i < 8 or ((i % 2) == 1 and i < (len(auc_means) - 8)):
            txt_pos = np.max(flr_locs[i:(i + str_len), 1]) + 0.004
            ax.text(i - 0.4, txt_pos, str(mtype),
                    rotation=41, ha='left', va='bottom', size=10)
            flr_locs[i, 1] = txt_pos

        else:
            txt_pos = np.min(flr_locs[(i - str_len):(i + 1), 0]) - 0.004
            ax.text(i + 0.4, txt_pos, str(mtype),
                    rotation=41, ha='right', va='top', size=10)
            flr_locs[i, 0] = txt_pos

    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__auc-distribution.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_acc_quartiles(auc_df, aupr_df, args, cdata):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, (ax_auc, ax_aupr) = plt.subplots(figsize=(22, 10), ncols=2)

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    aupr_vals = aupr_df.quantile(q=0.25, axis=1)

    mtype_sizes = [
        len(mtype.get_samples(cdata.mtree)) / len(cdata.get_samples())
        for mtype in auc_df.index
        ]

    ax_auc.scatter(mtype_sizes, auc_vals, s=17, c='black', alpha=0.47)
    ax_aupr.scatter(mtype_sizes, aupr_vals, s=17, c='black', alpha=0.47)

    for annot_x, annot_y, annot, halign in place_annot(
            mtype_sizes, auc_vals.values.tolist(),
            size_vec=[15 for _ in mtype_sizes], annot_vec=aupr_vals.index,
            x_range=max(mtype_sizes) * 1.03, y_range=1, gap_adj=53
            ):
        ax_auc.text(annot_x, annot_y, annot, size=11, ha=halign)

    for annot_x, annot_y, annot, halign in place_annot(
            mtype_sizes, aupr_vals.values.tolist(),
            size_vec=[15 for _ in mtype_sizes], annot_vec=aupr_vals.index,
            x_range=1, y_range=1, gap_adj=53
            ):
        ax_aupr.text(annot_x, annot_y, annot, size=11, ha=halign)

    ax_auc.set_xlim(0, max(mtype_sizes) * 1.03)
    ax_aupr.set_xlim(0, 1)
    for ax in (ax_auc, ax_aupr):
        ax.tick_params(pad=3.9)
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
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__acc-quartiles.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_mtype(par_df, auc_df, use_clf, args, cdata):
    fig, axarr = plt.subplots(figsize=(1 + 9 * len(use_clf.tune_priors), 13),
                              nrows=3, ncols=len(use_clf.tune_priors),
                              gridspec_kw={'height_ratios': [1, 0.3, 1]},
                              squeeze=False, sharex=False, sharey=True)

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    size_vec = [
        198 * len(mtype.get_samples(cdata.mtree)) / len(cdata.get_samples())
        for mtype in auc_vals.index
        ]

    for i, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        axarr[1, i].set_axis_off()
        axarr[2, i].tick_params(length=6)

        if detect_log_distr(tune_distr):
            med_vals = np.log10(par_df[par_name]).median(axis=1)
            mean_vals = np.log10(par_df[par_name]).mean(axis=1)
            use_distr = [np.log10(par_val) for par_val in tune_distr]
            par_lbl = par_name + '\n(log-scale)'

        else:
            med_vals = par_df[par_name].median(axis=1)
            mean_vals = par_df[par_name].mean(axis=1)
            use_distr = tune_distr
            par_lbl = par_name

        med_vals = med_vals[auc_vals.index]
        mean_vals = mean_vals[auc_vals.index]
        distr_diff = np.mean(np.array(use_distr[1:])
                             - np.array(use_distr[:-1]))

        for j in range(3):
            axarr[j, i].set_xlim(use_distr[0] - distr_diff / 2,
                                 use_distr[-1] + distr_diff / 2)

        axarr[1, i].text((use_distr[0] + use_distr[-1]) / 2, 0.5, par_lbl,
                         ha='center', va='center', fontsize=25,
                         weight='semibold')

        med_vals += np.random.normal(0,
                                     (use_distr[-1] - use_distr[0])
                                     / (len(tune_distr) * 17),
                                     auc_df.shape[0])
        mean_vals += np.random.normal(0,
                                     (use_distr[-1] - use_distr[0])
                                      / (len(tune_distr) * 23),
                                      auc_df.shape[0])

        axarr[0, i].scatter(med_vals, auc_vals,
                            s=size_vec, c='black', alpha=0.23)
        axarr[2, i].scatter(mean_vals, auc_vals,
                            s=size_vec, c='black', alpha=0.23)

        axarr[0, i].set_ylim(0, 1)
        axarr[2, i].set_ylim(0, 1)
        axarr[0, i].set_ylabel("1st Quartile AUC", size=19, weight='semibold')
        axarr[2, i].set_ylabel("1st Quartile AUC", size=19, weight='semibold')

        axarr[0, i].axhline(y=0.5, color='#550000',
                            linewidth=2.3, linestyle='--', alpha=0.32)
        axarr[2, i].axhline(y=0.5, color='#550000',
                            linewidth=2.3, linestyle='--', alpha=0.32)

        for par_val in use_distr:
            axarr[1, i].axvline(x=par_val, color='#116611',
                                ls='--', linewidth=3.4, alpha=0.27)

            axarr[0, i].axvline(x=par_val, color='#116611',
                                ls=':', linewidth=1.3, alpha=0.16)
            axarr[2, i].axvline(x=par_val, color='#116611',
                                ls=':', linewidth=1.3, alpha=0.16)

        annot_placed = place_annot(
            med_vals, auc_vals.values.tolist(),
            size_vec=size_vec, annot_vec=auc_vals.index,
            x_range=use_distr[-1] - use_distr[0] + 2 * distr_diff, y_range=1
            )
 
        for annot_x, annot_y, annot, halign in annot_placed:
            axarr[0, i].text(annot_x, annot_y, annot, size=8, ha=halign)
 
    plt.tight_layout(h_pad=0)
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__tuning-mtype.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_mtype_grid(par_df, auc_df, use_clf, args, cdata):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(0.5 + 7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    auc_clrs = auc_vals.apply(auc_cmap)
    size_vec = [461 * sum(cdata.train_pheno(mtype))
                / (len(cdata.get_samples()) * par_count)
                for mtype in auc_vals.index]

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
            use_distr1 = [np.log10(par_val) for par_val in tn_distr1]
            par_meds1 = np.log10(par_df[par_name1]).median(axis=1)
            par_means1 = np.log10(par_df[par_name1]).mean(axis=1)
            
            distr_diff = np.mean(np.log10(np.array(tn_distr1[1:]))
                                 - np.log10(np.array(tn_distr1[:-1])))
            plt_ymin = np.log10(tn_distr1[0]) - distr_diff / 2
            plt_ymax = np.log10(tn_distr1[-1]) + distr_diff / 2

        else:
            use_distr1 = tn_distr1
            par_meds1 = par_df[par_name1].median(axis=1)
            par_means1 = par_df[par_name1].mean(axis=1)

            distr_diff = np.mean(np.array(tn_distr1[1:])
                                 - np.array(tn_distr1[:-1]))
            plt_ymin = tn_distr1[0] - distr_diff / 2
            plt_ymax = tn_distr1[-1] + distr_diff / 2

        if detect_log_distr(tn_distr2):
            use_distr2 = [np.log10(par_val) for par_val in tn_distr2]
            par_meds2 = np.log10(par_df[par_name2]).median(axis=1)
            par_means2 = np.log10(par_df[par_name2]).mean(axis=1)

            distr_diff = np.mean(np.log10(np.array(tn_distr2[1:]))
                                 - np.log10(np.array(tn_distr2[:-1])))
            plt_xmin = np.log10(tn_distr2[0]) - distr_diff / 2
            plt_xmax = np.log10(tn_distr2[-1]) + distr_diff / 2

        else:
            use_distr2 = tn_distr2
            par_meds2 = par_df[par_name2].median(axis=1)
            par_means2 = par_df[par_name2].mean(axis=1)

            distr_diff = np.mean(np.array(tn_distr2[1:])
                                 - np.array(tn_distr2[:-1]))
            plt_xmin = tn_distr2[0] - distr_diff / 2
            plt_xmax = tn_distr2[-1] + distr_diff / 2

        par_meds1 = par_meds1[auc_clrs.index]
        par_meds2 = par_meds2[auc_clrs.index]
        y_adj = (plt_ymax - plt_ymin) / len(tn_distr1)
        x_adj = (plt_xmax - plt_xmin) / len(tn_distr2)
        plt_adj = (plt_xmax - plt_xmin) / (plt_ymax - plt_ymin)

        for med1, med2 in set(zip(par_meds1, par_meds2)):
            use_indx = (par_meds1 == med1) & (par_meds2 == med2)

            cnt_adj = use_indx.sum() ** 0.49
            use_sizes = [s for s, ix in zip(size_vec, use_indx) if ix]
            sort_indx = sorted(enumerate(use_sizes),
                               key=lambda x: x[1], reverse=True)

            from circlify import circlify
            mpl.use('Agg')

            for k, circ in enumerate(circlify([s for _, s in sort_indx])):
                axarr[i, j].scatter(
                    med2 + (1 / 23) * cnt_adj * circ.y * plt_adj,
                    med1 + (1 / 23) * cnt_adj * circ.x * plt_adj ** -1,
                    s=sort_indx[k][1], c=auc_clrs[use_indx][sort_indx[k][0]],
                    alpha=0.36, edgecolor='black'
                    )

        par_means1 += np.random.normal(0, y_adj / 27, auc_df.shape[0])
        par_means2 += np.random.normal(0, x_adj / 27, auc_df.shape[0])
        axarr[j, i].scatter(
            par_means1[auc_clrs.index], par_means2[auc_clrs.index],
            s=size_vec, c=auc_clrs, alpha=0.36, edgecolor='black'
            )

        axarr[i, j].set_xlim(plt_xmin, plt_xmax)
        axarr[i, j].set_ylim(plt_ymin, plt_ymax)
        axarr[j, i].set_ylim(plt_xmin, plt_xmax)
        axarr[j, i].set_xlim(plt_ymin, plt_ymax)

        annot_placed = place_annot(par_meds2, par_meds1,
                                   size_vec=size_vec,
                                   annot_vec=auc_vals.index,
                                   x_range=plt_xmax - plt_xmin,
                                   y_range=plt_ymax - plt_ymin)
 
        for annot_x, annot_y, annot, halign in annot_placed:
            axarr[i, j].text(annot_x, annot_y, annot, size=11, ha=halign)

        for par_val1 in use_distr1:
            axarr[i, j].axhline(y=par_val1, color='#116611',
                                ls=':', linewidth=2.3, alpha=0.19)
            axarr[j, i].axvline(x=par_val1, color='#116611',
                                ls=':', linewidth=2.3, alpha=0.19)

        for par_val2 in use_distr2:
            axarr[i, j].axvline(x=par_val2, color='#116611',
                                ls=':', linewidth=2.3, alpha=0.19)
            axarr[j, i].axhline(y=par_val2, color='#116611',
                                ls=':', linewidth=2.3, alpha=0.19)

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__tuning-mtype-grid.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance and tuning characteristics of a model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source was used")
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort]),
        args.model_name.split('__')[0]
        ), exist_ok=True)

    use_ctf = min(
        int(out_file.parts[-2].split('__samps-')[1])
        for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-data__{}.p.gz".format(
                args.expr_source, args.cohort, args.model_name)
            )
        )

    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, use_ctf)
    cdata = merge_cohort_data(os.path.join(base_dir, out_tag))

    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-data__{}.p.gz".format(
                                      args.model_name)),
                     'r') as fl:
        out_dict = pickle.load(fl)

    plot_label_stability(out_dict['Scores'], out_dict['Fit']['test'].AUC,
                         args, cdata)
    plot_label_correlation(out_dict['Scores'], out_dict['Fit']['test'].AUC,
                           args, cdata)

    plot_auc_distribution(out_dict['Fit']['test'].AUC, args)
    plot_acc_quartiles(out_dict['Fit']['test'].AUC,
                       out_dict['Fit']['test'].AUPR, args, cdata)

    plot_tuning_mtype(out_dict['Params'], out_dict['Fit']['test'].AUC,
                      out_dict['Clf'], args, cdata)
    if len(out_dict['Clf'].tune_priors) > 1:
        plot_tuning_mtype_grid(out_dict['Params'],
                               out_dict['Fit']['test'].AUC, out_dict['Clf'],
                               args, cdata)


if __name__ == "__main__":
    main()

