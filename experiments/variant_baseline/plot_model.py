
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'variant_baseline')
plot_dir = os.path.join(base_dir, 'plots', 'model')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.utilities.scatter_plotting import place_annot

import argparse
import dill as pickle

import numpy as np
import pandas as pd
from itertools import combinations as combn
from itertools import product

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def detect_log_distr(tune_distr):
    distr_diff = np.array(tune_distr[1:]) - np.array(tune_distr[:-1])
    diff_min = np.log10(np.min(distr_diff))
    diff_max = np.log10(np.max(distr_diff))

    return (diff_max - diff_min) > 4./3


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
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__auc-distribution.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_acc_quartiles(auc_df, aupr_df, args, cdata):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, (ax_auc, ax_aupr) = plt.subplots(figsize=(22, 10), ncols=2)

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    aupr_vals = aupr_df.quantile(q=0.25, axis=1)

    mtype_sizes = [
        len(mtype.get_samples(cdata.train_mut)) / len(cdata.samples)
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
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__acc-quartiles.svg".format(
                         args.model_name.split('__')[1])),
        dpi=350, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_generalization_error(train_aucs, test_aucs, args):
    plot_min = min(train_aucs.min().min(), test_aucs.min().min()) - 0.01

    if np.min(train_aucs.values) > 0.999:
        train_aucs += np.random.randn(train_aucs.shape[1]) / 500

    g = sns.JointGrid(train_aucs.values.flatten(), test_aucs.values.flatten(),
                      xlim=(plot_min, 1.01), ylim=(plot_min, 1.01), height=9)
    g = g.plot_joint(sns.kdeplot,
                     shade=True, shade_lowest=False, bw=0.01, cut=0)
    g = g.plot_marginals(sns.distplot, kde=False)

    g.ax_joint.tick_params(pad=3.9)
    g.ax_joint.plot([-1, 2], [-1, 2],
                    linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)
    g.ax_joint.set_xlabel('Training AUC', fontsize=22, weight='semibold')
    g.ax_joint.set_ylabel('Testing AUC', fontsize=22, weight='semibold')

    g.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__generalization.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_profile(tune_dict, use_clf, args, cdata):
    fig, axarr = plt.subplots(
        figsize=(17, 0.3 + 7 * len(use_clf.tune_priors)),
        nrows=len(use_clf.tune_priors), ncols=1, squeeze=False
        )

    tune_df = tune_dict['mean'] - tune_dict['std']
    tune_df.columns.names = [par for par, _ in use_clf.tune_priors]

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        tune_vals = tune_df.groupby(axis=1, level=par_name).quantile(q=0.25)

        if detect_log_distr(tune_distr):
            use_distr = [np.log10(par_val) for par_val in tune_distr]
            par_lbl = par_name + '\n(log-scale)'

        else:
            use_distr = tune_distr
            par_lbl = par_name

        ax.axhline(color='#550000', y=0.5, linewidth=3.1, alpha=0.32)
        ax.set_xlabel(par_lbl, fontsize=22, weight='semibold')
        ax.set_ylabel('Training AUC', fontsize=22, weight='semibold')

        for vals in tune_vals.values:
            ax.plot(use_distr, vals, '-',
                    linewidth=1.3, alpha=0.23, color=auc_cmap(np.max(vals)))

            diffs = np.argsort(np.abs(vals[1:] - vals[:-1]))
            chng_indx = np.argmax(sum(diffs[i:(i + 6)])
                                  for i in range(len(diffs) - 5))

            ax.plot(use_distr[chng_indx:(chng_indx + 6)],
                    vals[chng_indx:(chng_indx + 6)], '-', linewidth=2.7,
                    alpha=0.39, color=auc_cmap(np.max(vals)))

        for par_val in use_distr:
            ax.axvline(x=par_val, color='#116611',
                       ls=':', linewidth=1.3, alpha=0.16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__tuning-profile.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_distribution(par_df, auc_df, use_clf, args, cdata):
    fig, axarr = plt.subplots(
        figsize=(17, 0.3 + 7 * len(use_clf.tune_priors)),
        nrows=len(use_clf.tune_priors), ncols=1, squeeze=False
        )

    auc_vals = auc_df.values.flatten()
    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        ax.set_title(par_name, size=29, weight='semibold')

        use_df = pd.DataFrame({'Acc': auc_vals,
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
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__tuning-distribution.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_mtype(par_df, auc_df, use_clf, args, cdata):
    fig, axarr = plt.subplots(figsize=(1 + 9 * len(use_clf.tune_priors), 13),
                              nrows=3, ncols=len(use_clf.tune_priors),
                              gridspec_kw={'height_ratios': [1, 0.3, 1]},
                              squeeze=False, sharex=False, sharey=True)

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    size_vec = [
        198 * len(mtype.get_samples(cdata.train_mut)) / len(cdata.samples)
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
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__tuning-mtype.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_mtype_grid(par_df, auc_df, use_clf, args, cdata):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(0.5 + 7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    auc_vals = auc_df.quantile(q=0.25, axis=1)
    auc_clrs = auc_vals.apply(auc_cmap)
    size_vec = [461 * len(mtype.get_samples(cdata.train_mut)) /
                (len(cdata.samples) * par_count)
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
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__tuning-mtype-grid.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_profile_grid(tune_dict, use_clf, args, cdata):
    fig, axarr = plt.subplots(
        figsize=(0.1 + 2.3 * len(use_clf.tune_priors[1][1]),
                 0.1 + 2.3 * len(use_clf.tune_priors[0][1])),
        nrows=len(use_clf.tune_priors[0][1]),
        ncols=len(use_clf.tune_priors[1][1]),
        sharex=True, sharey=True
        )

    tune_grps = (tune_dict['mean'] - tune_dict['std']).groupby(
        axis=1, level=tune_dict['mean'].columns.names)
    ylim = max(tune_grps.min().values.min(), 0)
    mtype_order = tune_grps.quantile(q=0.25).max(axis=1).sort_values(
        ascending=False).index

    for (i, par_val1), (j, par_val2) in product(
            enumerate(use_clf.tune_priors[0][1]),
            enumerate(use_clf.tune_priors[1][1])
            ):

        if i == 0:
            axarr[i, j].text(0.5, 1.03, format(par_val2, '.1g'),
                             size=16, weight='semibold', ha='center',
                             va='bottom', transform=axarr[i, j].transAxes)

        if j == 0:
            axarr[i, j].text(-0.24, 0.5, format(par_val1, '.1g'),
                             size=16, weight='semibold', rotation=90,
                             ha='right', va='center',
                             transform=axarr[i, j].transAxes)

        axarr[i, j].plot(
            tune_grps.quantile(q=0.25).loc[
                mtype_order, (par_val1, par_val2)].values,
            linewidth=3.1, color='blue', alpha=0.9
            )

        axarr[i, j].fill_between(
            list(range(len(mtype_order))),
            y1=tune_grps.quantile(q=0.5).loc[
                mtype_order, (par_val1, par_val2)].values,
            y2=tune_grps.min().loc[mtype_order, (par_val1, par_val2)].values,
            facecolor='blue', alpha=0.3, interpolate=True
            )

        axarr[i, j].set_xticks([])
        axarr[i, j].set_ylim(ylim, 1.01)

    fig.text(0.5, 60/59, use_clf.tune_priors[1][0],
             size=21, weight='semibold', ha='center', va='bottom')
    fig.text(-0.03, 0.5, use_clf.tune_priors[0][0], size=21,
             weight='semibold', rotation=90, ha='right', va='center')

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__tuning-profile-grid.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance and tuning characteristics of a model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('expr_source', type=str,
                        choices=list(expr_sources.keys()),
                        help="which TCGA expression data source was used")
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, args.samp_cutoff)

    os.makedirs(os.path.join(plot_dir, out_tag,
                             args.model_name.split('__')[0]),
                exist_ok=True)

    cdata = merge_cohort_data(os.path.join(base_dir, 'output', out_tag))
    out_dict = pickle.load(open(os.path.join(base_dir, 'output', out_tag,
                                             "out-data__{}.p".format(
                                                 args.model_name)),
                                'rb'))

    plot_auc_distribution(out_dict['Fit']['test'].AUC, args)
    plot_acc_quartiles(out_dict['Fit']['test'].AUC,
                       out_dict['Fit']['test'].AUPR, args, cdata)
    plot_generalization_error(out_dict['Fit']['train'].AUC,
                              out_dict['Fit']['test'].AUC, args)

    plot_tuning_profile(out_dict['Tune']['Acc'], out_dict['Clf'], args, cdata)
    plot_tuning_distribution(out_dict['Params'], out_dict['Fit']['test'].AUC,
                             out_dict['Clf'], args, cdata)
    plot_tuning_mtype(out_dict['Params'], out_dict['Fit']['test'].AUC,
                      out_dict['Clf'], args, cdata)

    if len(out_dict['Clf'].tune_priors) > 1:
        plot_tuning_mtype_grid(out_dict['Params'],
                               out_dict['Fit']['test'].AUC, out_dict['Clf'],
                               args, cdata)

    if len(out_dict['Clf'].tune_priors) == 2:
        plot_tuning_profile_grid(out_dict['Tune']['Acc'], out_dict['Clf'],
                                 args, cdata)


if __name__ == "__main__":
    main()

