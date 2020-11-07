
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.data_dirs import choose_source
from ..utilities.metrics import calc_conf
from ..utilities.misc import choose_label_colour
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.label_placement import place_scatter_labels
from .plot_ccle import load_response_data

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import permutations, product

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'cohorts')


def plot_auc_comparison(auc_dfs, pheno_dicts, args):
    fig, ax = plt.subplots(figsize=(12, 12))

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty())}

    plt_min = 0.83
    for mtype in use_mtypes:
        auc_val1 = auc_dfs[args.cohorts[0]].loc[mtype, 'mean']
        auc_val2 = auc_dfs[args.cohorts[1]].loc[mtype, 'mean']

        mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                    * np.mean(pheno_dicts[args.cohorts[1]][mtype])) ** 0.5
        plt_min = min(plt_min, auc_val1 - 0.01, auc_val2 - 0.01)

        ax.scatter(auc_val1, auc_val2,
                   c=[choose_label_colour(tuple(mtype.label_iter())[0])],
                   s=1003 * mtype_sz, alpha=0.23, edgecolor='none')

    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    ax.grid(linewidth=0.83, alpha=0.41)

    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    lbl_base = "AUC in training cohort\n{}"
    xlbl = lbl_base.format(get_cohort_label(args.cohorts[0]))
    ylbl = lbl_base.format(get_cohort_label(args.cohorts[1]))
    ax.set_xlabel(xlbl, size=27, weight='semibold')
    ax.set_ylabel(ylbl, size=27, weight='semibold')

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "auc-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_coh_comparison(auc_dfs, conf_dfs, pheno_dicts, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    use_aucs = {
        coh: auc_df['mean'][[mtype for mtype in auc_df.index
                             if (not isinstance(mtype, RandomType)
                                 and (tuple(mtype.subtype_iter())[0][1]
                                      & copy_mtype).is_empty())]]
        for coh, auc_df in auc_dfs.items()
        }

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.83
    coh1, coh2 = args.cohorts

    for gene, auc_vec1 in use_aucs[coh1].groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        if len(auc_vec1) > 1:
            auc_vec2 = use_aucs[coh2][[
                mtype for mtype in use_aucs[coh2].index
                if tuple(mtype.label_iter())[0] == gene
                ]]

            if len(auc_vec2) > 1:
                base_mtype = MuType({('Gene', gene): pnt_mtype})

                for coh, auc_vec in zip(args.cohorts, [auc_vec1, auc_vec2]):
                    base_indx = auc_vec.index.get_loc(base_mtype)
                    best_subtype = auc_vec[:base_indx].append(
                        auc_vec[(base_indx + 1):]).idxmax()

                    auc_tupl = tuple(auc_vec[[base_mtype, best_subtype]])
                    use_clr = choose_label_colour(gene)
                    line_dict[auc_tupl] = dict(c=use_clr)

                    base_size = np.mean(pheno_dicts[coh][base_mtype])
                    plt_size = 0.07 * base_size ** 0.5
                    plot_dict[auc_tupl] = [plt_size,
                                           (gene, get_cohort_label(coh))]
                    plt_min = min(plt_min, min(auc_tupl) - 0.03)

                    best_prop = np.mean(
                        pheno_dicts[coh][best_subtype]) / base_size
                    auc_bbox = (auc_tupl[0] - plt_size / 2,
                                auc_tupl[1] - plt_size / 2,
                                plt_size, plt_size)

                    pie_ax = inset_axes(
                        ax, width='100%', height='100%',
                        bbox_to_anchor=auc_bbox, bbox_transform=ax.transData,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    if coh == coh1:
                        pie_args = dict(colors=[use_clr + (0.77, ),
                                                use_clr + (0.29, )])

                    else:
                        pie_args = dict(colors=['none', 'none'],
                                        wedgeprops=dict(edgecolor=use_clr,
                                                        linewidth=7 / 3,
                                                        alpha=0.37))

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.37, 0], startangle=90, **pie_args)

    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    ax.grid(linewidth=0.83, alpha=0.41)

    ax.plot(plt_lims, [0.5, 0.5], color='black',
            linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims, color='black',
            linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims, color='#550000',
            linewidth=2.1, linestyle='--', alpha=0.37)

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=27, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=27, weight='semibold')

    #TODO: have one label per gene, with lines to both cohort pies?
    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[plt_lims, plt_lims],
                                       line_dict=line_dict, font_size=15)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "coh-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_sub_comparison(auc_dfs, trnsf_dicts, pheno_dicts, conf_dfs, args):
    fig, axarr = plt.subplots(figsize=(12.9, 14), nrows=2, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty())}

    auc_dicts = {
        train_coh: {
            'Wthn': auc_dfs[train_coh]['mean'],
            'Trnsf': pd.concat([
                trnsf_data[train_coh]['AUC']['mean']
                for trnsf_data in trnsf_dicts[other_coh].values()
                ])
            }
        for train_coh, other_coh in permutations(args.cohorts)
        }

    plot_dicts = {(i, j): dict() for i, j in product(range(2), repeat=2)}
    line_dicts = {(i, j): dict() for i, j in product(range(2), repeat=2)}
    plt_min = 0.83

    for i, auc_lbl in enumerate(['Wthn', 'Trnsf']):
        for j, coh in enumerate(args.cohorts):
            for gene, auc_vec in auc_dicts[coh][auc_lbl][use_mtypes].groupby(
                    lambda mtype: tuple(mtype.label_iter())[0]):

                if len(auc_vec) > 1:
                    base_mtype = MuType({('Gene', gene): pnt_mtype})

                    base_indx = auc_vec.index.get_loc(base_mtype)
                    best_subtype = auc_vec[:base_indx].append(
                        auc_vec[(base_indx + 1):]).idxmax()

                    if auc_vec[best_subtype] > 0.6:
                        auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
                        use_clr = choose_label_colour(gene)
                        line_dicts[i, j][auc_tupl] = dict(c=use_clr)

                        base_size = np.mean(pheno_dicts[coh][base_mtype])
                        plt_size = 0.07 * base_size ** 0.5
                        plot_dicts[i, j][auc_tupl] = [plt_size, ('', '')]
                        plt_min = min(plt_min, min(auc_tupl) - 0.03)

                        best_prop = np.mean(pheno_dicts[coh][best_subtype])
                        best_prop /= base_size
                        conf_sc = calc_conf(conf_dfs[coh].loc[best_subtype],
                                            conf_dfs[coh].loc[base_mtype])

                        if conf_sc > 0.9:
                            plot_dicts[i, j][auc_tupl][1] = (
                                gene, get_fancy_label(
                                    tuple(best_subtype.subtype_iter())[0][1],
                                    pnt_link='\n', phrase_link=' '
                                    )
                                )

                        else:
                            plot_dicts[i, j][auc_tupl][1] = gene, ''

                        auc_bbox = (auc_tupl[0] - plt_size / 2,
                                    auc_tupl[1] - plt_size / 2,
                                    plt_size, plt_size)

                        pie_ax = inset_axes(
                            axarr[i, j], width='100%', height='100%',
                            bbox_to_anchor=auc_bbox,
                            bbox_transform=axarr[i, j].transData,
                            axes_kwargs=dict(aspect='equal'), borderpad=0
                            )

                        pie_ax.pie(x=[best_prop, 1 - best_prop],
                                   colors=[use_clr + (0.77, ),
                                           use_clr + (0.29, )],
                                   explode=[0.29, 0], startangle=90)

    plt_lims = plt_min, 1 + (1 - plt_min) / 151
    for i, j in product(range(2), repeat=2):
        axarr[i, j].grid(linewidth=0.83, alpha=0.41)

        axarr[i, j].plot(plt_lims, [0.5, 0.5], color='black',
                         linewidth=1.3, linestyle=':', alpha=0.71)
        axarr[i, j].plot([0.5, 0.5], plt_lims, color='black',
                         linewidth=1.3, linestyle=':', alpha=0.71)

        axarr[i, j].plot(plt_lims, [1, 1], color='black',
                         linewidth=1.9, alpha=0.89)
        axarr[i, j].plot([1, 1], plt_lims, color='black',
                         linewidth=1.9, alpha=0.89)
        axarr[i, j].plot(plt_lims, plt_lims, color='#550000',
                         linewidth=2.1, linestyle='--', alpha=0.41)

        axarr[i, j].set_xlabel("AUC using all point mutations",
                               size=23, weight='semibold')

        if j == 0:
            axarr[i, j].set_ylabel("AUC of best found subgrouping",
                                   size=23, weight='semibold')

        if plot_dicts[i, j]:
            lbl_pos = place_scatter_labels(plot_dicts[i, j], axarr[i, j],
                                           plt_lims=[plt_lims, plt_lims],
                                           line_dict=line_dicts[i, j])

        axarr[i, j].set_xlim(plt_lims)
        axarr[i, j].set_ylim(plt_lims)

    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "sub-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_cross_sub_comparison(auc_dfs, pheno_dicts, conf_dfs, args):
    fig, axarr = plt.subplots(figsize=(14.3, 7), nrows=1, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty())}
    plt_min = 0.83

    for i, (base_coh, other_coh) in enumerate(permutations(args.cohorts)):
        plot_dict = dict()
        line_dict = dict()

        for gene, auc_vec in auc_dfs[base_coh].loc[
                use_mtypes, 'mean'].groupby(
                    lambda mtype: tuple(mtype.label_iter())[0]):

            if len(auc_vec) > 1:
                base_mtype = MuType({('Gene', gene): pnt_mtype})

                base_indx = auc_vec.index.get_loc(base_mtype)
                best_subtype = auc_vec[:base_indx].append(
                    auc_vec[(base_indx + 1):]).idxmax()

                if auc_vec[best_subtype] > 0.6:
                    plt_x = auc_dfs[other_coh].loc[base_mtype, 'mean']
                    plt_y = auc_dfs[other_coh].loc[best_subtype, 'mean']
                    use_clr = choose_label_colour(gene)
                    line_dict[plt_x, plt_y] = dict(c=use_clr)

                    base_size = np.mean(pheno_dicts[other_coh][base_mtype])
                    plt_size = 0.07 * base_size ** 0.5
                    plot_dict[plt_x, plt_y] = [plt_size, ('', '')]
                    plt_min = min(plt_min, plt_x - 0.01, plt_y - 0.01)

                    best_prop = np.mean(pheno_dicts[other_coh][best_subtype])
                    best_prop /= base_size

                    conf_sc = calc_conf(conf_dfs[other_coh][best_subtype],
                                        conf_dfs[other_coh][base_mtype])

                    if conf_sc > 0.8:
                        plot_dict[plt_x, plt_y][1] = gene, get_fancy_label(
                            tuple(best_subtype.subtype_iter())[0][1],
                            pnt_link='\n', phrase_link=' '
                            )

                    else:
                        plot_dict[plt_x, plt_y][1] = gene, ''

                    auc_bbox = (plt_x - plt_size / 2, plt_y - plt_size / 2,
                                plt_size, plt_size)

                    pie_ax = inset_axes(axarr[i], width='100%', height='100%',
                                        bbox_to_anchor=auc_bbox,
                                        bbox_transform=axarr[i].transData,
                                        axes_kwargs=dict(aspect='equal'),
                                        borderpad=0)

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               colors=[use_clr + (0.77, ),
                                       use_clr + (0.29, )],
                               explode=[0.29, 0], startangle=90)

        if plot_dict:
            lbl_pos = place_scatter_labels(
                plot_dict, axarr[i],
                plt_lims=[[0.48, 1.01], [0.48, 1.01]], line_dict=line_dict
                )

        xlbl = "AUC in {} using\nall point mutations".format(
            get_cohort_label(other_coh))
        ylbl = "AUC in {} of best\nfound subgrouping in {}".format(
            get_cohort_label(other_coh), get_cohort_label(base_coh))

        axarr[i].set_xlabel(xlbl, size=19, weight='semibold')
        axarr[i].set_ylabel(ylbl, size=19, weight='semibold')

    plt_lims = plt_min, 1 + (1 - plt_min) / 73
    for ax in axarr:
        ax.grid(linewidth=0.83, alpha=0.41)

        ax.plot(plt_lims, [0.5, 0.5], color='black',
                linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims, color='black',
                linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
        ax.plot(plt_lims, plt_lims, color='#550000',
                linewidth=2.1, linestyle='--', alpha=0.41)

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "cross-sub-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_metrics_comparison(auc_dfs, pheno_dicts, args):
    fig, axarr = plt.subplots(figsize=(17.5, 6), nrows=1, ncols=3)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if not isinstance(mtype, RandomType)}

    plt_min = 0.83
    for ax, auc_lbl in zip(axarr, ['mean', 'all', 'CV']):
        ax.set_title(auc_lbl, size=23, weight='semibold')

        if auc_lbl == 'CV':
            auc_vals1 = auc_dfs[args.cohorts[0]]['CV'].apply(
                np.quantile, q=0.25)
            auc_vals2 = auc_dfs[args.cohorts[1]]['CV'].apply(
                np.quantile, q=0.25)

        else:
            auc_vals1 = auc_dfs[args.cohorts[0]][auc_lbl]
            auc_vals2 = auc_dfs[args.cohorts[1]][auc_lbl]

        for mtype in use_mtypes:
            use_gene = tuple(mtype.label_iter())[0]
            gene_clr = choose_label_colour(use_gene)

            plt_min = min(plt_min,
                          auc_vals1[mtype] - 0.02, auc_vals2[mtype] - 0.02)
            mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                        * np.mean(pheno_dicts[args.cohorts[1]][mtype])) ** 0.5

            ax.scatter(auc_vals1[mtype], auc_vals2[mtype], c=[gene_clr],
                       s=408 * mtype_sz, alpha=0.19, edgecolor='none')

        corr_val = spearmanr(auc_vals1[use_mtypes], auc_vals2[use_mtypes])[0]
        ax.text(0.97, 0.1, "Spearman: {:.3f}".format(corr_val),
                size=15, ha='right', va='bottom', transform=ax.transAxes)

        rmse_val = ((auc_vals1[use_mtypes]
                     - auc_vals2[use_mtypes]) ** 2).mean() ** 0.5
        ax.text(0.97, 0.06, "RMSE: {:.3f}".format(rmse_val),
                size=15, ha='right', va='bottom', transform=ax.transAxes)

        mae_val = (auc_vals1[use_mtypes] - auc_vals2[use_mtypes]).abs().mean()
        ax.text(0.97, 0.02, "MAE: {:.3f}".format(mae_val),
                size=15, ha='right', va='bottom', transform=ax.transAxes)

    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    for ax in axarr:
        ax.grid(linewidth=0.83, alpha=0.41)

        ax.plot(plt_lims, [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims,
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    coh_lbl1 = get_cohort_label(args.cohorts[0])
    coh_lbl2 = get_cohort_label(args.cohorts[1])

    fig.text(0.5, 0, "AUC in training cohort {}".format(coh_lbl1),
             fontsize=21, weight='semibold', ha='center', va='top')
    fig.text(-1 / 53, 0.5, "AUC in training cohort\n{}".format(coh_lbl2),
             fontsize=21, weight='semibold', rotation=90,
             ha='center', va='center')

    fig.tight_layout(w_pad=2.3)
    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "metrics-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_transfer_aucs(trnsf_dicts, auc_dfs, pheno_dicts, args):
    fig, axarr = plt.subplots(figsize=(14, 14), nrows=2, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if not isinstance(mtype, RandomType)}
    plt_min = 0.83

    for i, (base_coh, trnsf_coh) in enumerate(permutations(args.cohorts)):
        trnsf_aucs = pd.concat([
            trnsf_data[trnsf_coh]['AUC']['mean']
            for trnsf_data in trnsf_dicts[base_coh].values()
            ])

        axarr[i, 0].set_ylabel(
            "AUC when transferred\nto {}".format(get_cohort_label(trnsf_coh)),
            size=23, weight='semibold'
            )

        for j, train_coh in enumerate(args.cohorts):
            if i == 1:
                axarr[i, j].set_xlabel(
                    "AUC when trained\non {}".format(
                        get_cohort_label(train_coh)),
                    size=23, weight='semibold'
                    )

            for mtype in use_mtypes:
                use_gene = tuple(mtype.label_iter())[0]
                gene_clr = choose_label_colour(use_gene)

                train_auc = auc_dfs[train_coh].loc[mtype, 'mean']
                trnsf_auc = trnsf_aucs[mtype]
                plt_min = min(plt_min,
                              train_auc - 0.01, trnsf_auc - 0.01)

                mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                            * np.mean(pheno_dicts[args.cohorts[1]][mtype]))
                mtype_sz **= 0.5

                axarr[i, j].scatter(train_auc, trnsf_auc,
                                    c=[gene_clr], s=397 * mtype_sz,
                                    alpha=0.19, edgecolor='none')

    for ax in axarr.flatten():
        plt_lims = plt_min, 1 + (1 - plt_min) / 181
        ax.grid(linewidth=0.83, alpha=0.41)

        ax.plot(plt_lims, [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims,
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    fig.tight_layout(w_pad=2.1, h_pad=4.1)
    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "transfer-aucs_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_silent_comparison(auc_dfs, cdata_dict, args):
    fig, (freq_ax, sil_ax) = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)

    use_lvls = 'Gene', 'Scale', 'Consequence', 'HGVSp'
    sil_k = 'synonymous_variant'
    for coh in args.cohorts:
        cdata_dict[coh].add_mut_lvls(use_lvls)

    n_samps1 = len(cdata_dict[args.cohorts[0]].get_samples())
    n_samps2 = len(cdata_dict[args.cohorts[1]].get_samples())
    muts_dict1 = dict(cdata_dict[args.cohorts[0]].mtrees[use_lvls])
    muts_dict2 = dict(cdata_dict[args.cohorts[1]].mtrees[use_lvls])

    plt_genes = {
        gene for gene in muts_dict1.keys() | muts_dict2.keys()
        if ((gene in muts_dict1 and 'Point' in dict(muts_dict1[gene])
             and len(dict(muts_dict1[gene])['Point'].get_samples()) >= 20)
            or (gene in muts_dict2 and 'Point' in dict(muts_dict2[gene])
                and len(dict(muts_dict2[gene])['Point'].get_samples()) >= 20))
        }

    plot_dicts = {'Freq': dict(), 'Sil': dict()}
    line_dicts = {'Freq': dict(), 'Sil': dict()}
    freq_max = 0.27
    sil_max = 0.19

    for plt_gene in plt_genes:
        if (plt_gene not in muts_dict1 or
                'Point' not in dict(muts_dict1[plt_gene])):
            freq1 = 0
            sil1 = 0

        else:
            n_muts1 = len(muts_dict1[plt_gene]['Point'].get_samples())
            freq1 = n_muts1 / n_samps1

            if sil_k not in dict(muts_dict1[plt_gene]['Point']):
                sil1 = 0
            else:
                sil1 = len(muts_dict1[plt_gene][
                    'Point'][sil_k].get_samples()) / n_muts1

        if (plt_gene not in muts_dict2 or
                'Point' not in dict(muts_dict2[plt_gene])):
            freq2 = 0
            sil2 = 0

        else:
            n_muts2 = len(muts_dict2[plt_gene]['Point'].get_samples())
            freq2 = len(muts_dict2[plt_gene]['Point'].get_samples()) / n_samps2

            if sil_k not in dict(muts_dict2[plt_gene]['Point']):
                sil2 = 0
            else:
                sil2 = len(muts_dict2[plt_gene][
                    'Point'][sil_k].get_samples()) / n_muts2

        freq_max = max(freq_max, freq1 + 0.03, freq2 + 0.03)
        sil_max = max(sil_max, sil1 + 0.03, sil2 + 0.03)
        plt_clr = choose_label_colour(plt_gene)

        freq_ax.scatter(freq1, freq2,
                        c=[plt_clr], s=29, alpha=0.29, edgecolor='none')
        sil_ax.scatter(sil1, sil2,
                       c=[plt_clr], s=29, alpha=0.29, edgecolor='none')

        line_dicts['Freq'][freq1, freq2] = dict(c=plt_clr)
        line_dicts['Sil'][sil1, sil2] = dict(c=plt_clr)

        if abs(freq1 - freq2) > 0.03 or (freq1 + freq2) > 0.3:
            plot_dicts['Freq'][freq1, freq2] = 0.005, (plt_gene, '')
        else:
            plot_dicts['Freq'][freq1, freq2] = 0.005, ('', '')

        if abs(sil1 - sil2) > 0.1:
            plot_dicts['Sil'][sil1, sil2] = 0.005, (plt_gene, '')
        else:
            plot_dicts['Sil'][sil1, sil2] = 0.005, ('', '')

    freq_max = min(freq_max, 1.005)
    sil_max = min(sil_max, 1.005)

    for ax, pnt_lbl, plt_max in zip([freq_ax, sil_ax],
                                    ['Freq', 'Sil'], [freq_max, sil_max]):
        plt_lims = -plt_max / 73, plt_max
        ax.grid(linewidth=0.83, alpha=0.41)

        ax.plot([0, plt_max - 0.01], [0, plt_max - 0.01],
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        ax.plot([0, 0], [0, plt_max],
                color='black', linewidth=0.83, alpha=0.83)
        ax.plot([1, 1], [0, plt_max],
                color='black', linewidth=0.83, alpha=0.83)
        ax.plot([0, plt_max], [0, 0],
                color='black', linewidth=0.83, alpha=0.83)
        ax.plot([0, plt_max], [1, 1],
                color='black', linewidth=0.83, alpha=0.83)

        if plot_dicts[pnt_lbl]:
            lbl_pos = place_scatter_labels(
                plot_dicts[pnt_lbl], ax,
                plt_lims=[[plt_max / 31, plt_max], [plt_max / 31, plt_max]],
                line_dict=line_dicts[pnt_lbl], linewidth=0.91, alpha=0.43
                )

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    coh_lbl1 = get_cohort_label(args.cohorts[0])
    coh_lbl2 = get_cohort_label(args.cohorts[1])

    freq_ax.set_xlabel(
        "% of {} samples with\ngene's point mutation".format(coh_lbl1),
        size=21, weight='semibold'
        )
    freq_ax.set_ylabel(
        "% of {} samples with\ngene's point mutation".format(coh_lbl2),
        size=21, weight='semibold'
        )

    sil_ax.set_xlabel(
        "% of gene's point mutants with"
        "\nsilent mutation in {}".format(coh_lbl1),
        size=21, weight='semibold'
        )
    sil_ax.set_ylabel(
        "% of gene's point mutants with"
        "\nsilent mutation in {}".format(coh_lbl2),
        size=21, weight='semibold'
        )

    fig.tight_layout(w_pad=2.9)
    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "silent-comp.svg"),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_subtype_comparison(auc_dfs, pheno_dicts, conf_dfs, plt_gene, args):
    fig, (auc_ax, conf_ax) = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty()
                      and tuple(mtype.label_iter())[0] == plt_gene)}

    base_mtype = MuType({('Gene', plt_gene): pnt_mtype})
    auc_min = -0.005
    auc_max = 0.05
    conf_min = 0.43

    for mtype in use_mtypes:
        conf_sc1 = calc_conf(conf_dfs[args.cohorts[0]][mtype],
                             conf_dfs[args.cohorts[0]][base_mtype])
        conf_sc2 = calc_conf(conf_dfs[args.cohorts[1]][mtype],
                             conf_dfs[args.cohorts[1]][base_mtype])

        if conf_sc1 > 0.5 or conf_sc2 > 0.5:
            mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                        * np.mean(pheno_dicts[args.cohorts[1]][mtype])) ** 0.5

            auc_diff1 = auc_dfs[args.cohorts[0]].loc[mtype, 'mean']
            auc_diff1 -= auc_dfs[args.cohorts[0]].loc[base_mtype, 'mean']
            auc_diff2 = auc_dfs[args.cohorts[1]].loc[mtype, 'mean']
            auc_diff2 -= auc_dfs[args.cohorts[1]].loc[base_mtype, 'mean']

            auc_min = min(auc_min, auc_diff1 - 0.007, auc_diff2 - 0.007)
            auc_max = max(auc_max, auc_diff1 + 0.007, auc_diff2 + 0.007)
            conf_min = min(conf_min, conf_sc1 - 0.02, conf_sc2 - 0.02)

            auc_ax.scatter(auc_diff1, auc_diff2,
                           s=1401 * mtype_sz, alpha=0.29, edgecolor='none')
            conf_ax.scatter(conf_sc1, conf_sc2,
                            s=1401 * mtype_sz, alpha=0.29, edgecolor='none')

    auc_ax.plot([auc_min, auc_max], [0, 0],
                color='black', linewidth=1.7, alpha=0.71, linestyle=':')
    auc_ax.plot([0, 0], [auc_min, auc_max],
                color='black', linewidth=1.7, alpha=0.71, linestyle=':')

    conf_ax.plot([0.5, 0.5], [conf_min, 1.001],
                 color='black', linewidth=1.7, alpha=0.71, linestyle=':')
    conf_ax.plot([conf_min, 1.001], [0.5, 0.5],
                 color='black', linewidth=1.7, alpha=0.71, linestyle=':')

    auc_ax.set_xlim([auc_min, auc_max])
    auc_ax.set_ylim([auc_min, auc_max])
    conf_ax.set_xlim([conf_min, 1 + (1 - conf_min) / 61])
    conf_ax.set_ylim([conf_min, 1 + (1 - conf_min) / 61])

    fig.tight_layout(w_pad=2.3)
    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "{}_subtype-comparison_{}.svg".format(
                         plt_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_subtype_transfer(auc_dfs, trnsf_dicts, pheno_dicts, conf_dfs,
                          plt_gene, args):
    fig, axarr = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)

    base_mtype = MuType({('Gene', plt_gene): pnt_mtype})
    plt_min = 0.89

    for ax, (train_coh, other_coh) in zip(axarr, permutations(args.cohorts)):
        trnsf_aucs = pd.concat([
            trnsf_data[other_coh]['AUC']['mean']
            for trnsf_data in trnsf_dicts[train_coh].values()
            ])

        use_mtypes = {mtype for mtype in trnsf_aucs.index
                      if ((tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty()
                          and tuple(mtype.label_iter())[0] == plt_gene)}

        for mtype in use_mtypes:
            train_auc = auc_dfs[train_coh].loc[mtype, 'mean']
            mtype_sz = np.mean(pheno_dicts[train_coh][mtype])
            plt_min = min(plt_min, train_auc - 0.01, trnsf_aucs[mtype] - 0.01)

            if mtype == base_mtype:
                use_mrk = 'X'
                use_sz = 5305 * mtype_sz
                use_alpha = 0.47

            else:
                use_mrk = 'o'
                use_sz = 1507 * mtype_sz
                use_alpha = 0.23

                conf_sc = calc_conf(conf_dfs[train_coh][mtype],
                                    conf_dfs[train_coh][base_mtype])

            ax.scatter(train_auc, trnsf_aucs[mtype],
                       marker=use_mrk, s=use_sz, alpha=use_alpha,
                       edgecolor='none')

        x_lbl = "AUC in training cohort\n{}".format(
            get_cohort_label(train_coh))
        y_lbl = "AUC when transferred\nto {}".format(
            get_cohort_label(other_coh))

        ax.set_xlabel(x_lbl, size=23, weight='semibold')
        ax.set_ylabel(y_lbl, size=23, weight='semibold')

    for ax in axarr:
        ax.grid(linewidth=0.83, alpha=0.41)
        plt_lims = plt_min, 1 + (1 - plt_min) / 181

        ax.plot(plt_lims, [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims,
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    fig.tight_layout(w_pad=2.3)
    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "{}__subtype-transfer_{}.svg".format(
                         plt_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_subtype_response(auc_dicts, ccle_dfs, resp_vals,
                          conf_dicts, pheno_dicts, plt_gene, args):
    fig, ax = plt.subplots(figsize=(11, 11))
    base_mtype = MuType({('Gene', plt_gene): pnt_mtype})

    auc_df = pd.DataFrame({
        coh: auc_data.loc[
            [mtype for mtype in auc_data.index
             if (not isinstance(mtype, RandomType)
                 and tuple(mtype.label_iter())[0] == plt_gene)],
            'mean'
            ]
        for coh, auc_data in auc_dicts.items()
        })

    auc_df = auc_df.loc[~auc_df.isnull().any(axis=1)
                        & (auc_df > 0.7).all(axis=1)]

    if auc_df.shape[0] == 0:
        return None

    use_resp = resp_vals[~resp_vals.isna()]

    corr_df = pd.DataFrame({
        coh: {
            mtype: -spearmanr(
                ccle_df.loc[mtype,
                            set(use_resp.index) & set(ccle_df.columns)],
                use_resp[set(use_resp.index) & set(ccle_df.columns)]
                ).correlation
            for mtype in auc_df.index if mtype in ccle_df.index
            }
        for coh, ccle_df in ccle_dfs.items()
        })

    plot_dict = dict()
    clr_dict = dict()
    plt_lims = np.percentile(corr_df.quantile(q=[0, 1]).values, q=[0, 100])
    plt_lims += [-0.03, 0.03]

    for mtype, (corr_x, corr_y) in corr_df.iterrows():
        plt_sz = np.prod([np.mean(phn_dict[mtype])
                          for phn_dict in pheno_dicts.values()])

        mtype_lbl = get_fancy_label(tuple(mtype.subtype_iter())[0][1],
                                    pnt_link='\n', phrase_link=' ')
        plot_dict[corr_x, corr_y] = (0.07 * plt_sz ** 0.5, ('', mtype_lbl))
        clr_dict[corr_x, corr_y] = choose_label_colour(mtype_lbl)

        ax.scatter(corr_x, corr_y, c=[clr_dict[corr_x, corr_y]],
                   s=plt_sz * 5031, alpha=0.37, edgecolors='none')

    drug_lbl = "correlation with {} response\nusing ".format(resp_vals.name)
    ax.set_xlabel(''.join([drug_lbl, get_cohort_label(args.cohorts[0])]),
                  size=25, weight='semibold')
    ax.set_ylabel(''.join([drug_lbl, get_cohort_label(args.cohorts[1])]),
                  size=25, weight='semibold')

    ax.plot(plt_lims, [0, 0],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0, 0], plt_lims, color='black', linewidth=1.3,
            linestyle=':', alpha=0.71)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax, font_size=17)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "{}_{}__subtype-response_{}.svg".format(
                         plt_gene, resp_vals.name, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_cohorts',
        description="Plots comparisons across and between a pair of cohorts."
        )

    parser.add_argument('cohorts', nargs=2, help="which cohorts to use")
    parser.add_argument('classif', help='a mutation classifier')

    parser.add_argument('--genes', '-g', nargs='+',
                        help='a list of mutated genes', type=str)
    parser.add_argument('--drugs', '-d', nargs='+',
                        help='a list of drug treatments', type=str)
    args = parser.parse_args()

    cdata_dict = {coh: None for coh in args.cohorts}
    phn_dicts = {coh: dict() for coh in args.cohorts}
    auc_dict = {coh: dict() for coh in args.cohorts}
    conf_dict = {coh: dict() for coh in args.cohorts}
    trnsf_dicts = {coh: dict() for coh in args.cohorts}
    ccle_dict = {coh: dict() for coh in args.cohorts}

    for coh in args.cohorts:
        use_src = choose_source(coh)

        out_datas = [
            out_file.parts[-2:]
            for out_file in Path(base_dir).glob(os.path.join(
                "{}__{}__samps-*".format(use_src, coh),
                "out-trnsf__*__{}.p.gz".format(args.classif)
                ))
            ]

        out_list = pd.DataFrame([
            {'Samps': int(out_data[0].split('__samps-')[1]),
             'Levels': '__'.join(out_data[1].split(
                 'out-trnsf__')[1].split('__')[:-1])}
            for out_data in out_datas
            ])

        if out_list.shape[0] == 0:
            raise ValueError("No experiment output found for "
                             "cohort `{}` !".format(coh))

        out_use = out_list.groupby('Levels')['Samps'].min()
        if 'Consequence__Exon' not in out_use.index:
            raise ValueError(
                "Cannot compare AUCs until this experiment is run with "
                "mutation levels `Consequence__Exon` which tests "
                "genes' base mutations!"
                )

        for lvls, ctf in out_use.iteritems():
            out_tag = "{}__{}__samps-{}".format(use_src, coh, ctf)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "cohort-data__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[coh] is None:
                cdata_dict[coh] = new_cdata
            else:
                cdata_dict[coh].merge(new_cdata)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-pheno__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                phn_dicts[coh].update(pickle.load(f))

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-aucs__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                auc_dict[coh][lvls] = pickle.load(f)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-conf__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                conf_dict[coh][lvls] = pickle.load(f)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-trnsf__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                trnsf_dicts[coh][lvls] = pickle.load(f)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "trnsf-preds__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                ccle_mat = pickle.load(f)['CCLE']

                ccle_dict[coh][lvls] = pd.DataFrame(
                    np.vstack(ccle_mat.values), index=ccle_mat.index,
                    columns=trnsf_dicts[coh][lvls]['CCLE']['Samps']
                    )

    resp_df = load_response_data()
    auc_dfs = {coh: pd.concat(auc_dict[coh].values()) for coh in args.cohorts}

    conf_dfs = {coh: pd.concat(conf_dict[coh].values())
                for coh in args.cohorts}
    ccle_dfs = {coh: pd.concat(ccle_dict[coh].values())
                for coh in args.cohorts}

    # creates directory where plots will be saved
    os.makedirs(os.path.join(plot_dir, '__'.join(args.cohorts)),
                exist_ok=True)

    plot_auc_comparison(auc_dfs, phn_dicts, args)
    plot_coh_comparison(auc_dfs, conf_dfs, phn_dicts, args)
    plot_sub_comparison(auc_dfs, trnsf_dicts, phn_dicts, conf_dfs, args)

    plot_cross_sub_comparison(auc_dfs, phn_dicts, conf_dfs, args)
    plot_metrics_comparison(auc_dfs, phn_dicts, args)
    plot_transfer_aucs(trnsf_dicts, auc_dfs, phn_dicts, args)
    plot_silent_comparison(auc_dfs, cdata_dict, args)

    if args.genes is not None:
        for gene in args.genes:
            plot_subtype_comparison(auc_dfs, phn_dicts, conf_dfs, gene, args)
            plot_subtype_transfer(auc_dfs, trnsf_dicts, phn_dicts, conf_dfs,
                                  gene, args)

        if args.drugs is not None:
            for drug in args.drugs:
                plot_subtype_response(auc_dfs, ccle_dfs, resp_df[drug],
                                      conf_dfs, phn_dicts, gene, args)


if __name__ == '__main__':
    main()

