
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'cohorts')

from HetMan.experiments.subvariant_test import pnt_mtype, copy_mtype
from HetMan.experiments.subvariant_test.utils import get_fancy_label
from HetMan.experiments.subvariant_tour.utils import RandomType
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_test.plot_aucs import (
    choose_gene_colour, place_labels)
from HetMan.experiments.subvariant_test.plot_gene import get_cohort_label
from HetMan.experiments.subvariant_infer.setup_infer import choose_source
from HetMan.experiments.subvariant_infer.utils import get_mtype_gene

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from itertools import permutations

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_auc_comparison(auc_dfs, pheno_dicts, args):
    fig, ax = plt.subplots(figsize=(12, 12))

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if not isinstance(mtype, RandomType)}

    plt_min = 0.83
    for mtype in use_mtypes:
        auc_val1 = auc_dfs[args.cohorts[0]].loc[mtype, 'mean']
        auc_val2 = auc_dfs[args.cohorts[1]].loc[mtype, 'mean']

        use_gene = mtype.get_labels()[0]
        gene_clr = choose_gene_colour(use_gene)
        plt_min = min(plt_min, auc_val1 - 0.02, auc_val2 - 0.02)
        mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                    * np.mean(pheno_dicts[args.cohorts[1]][mtype])) ** 0.5

        ax.scatter(auc_val1, auc_val2, c=[gene_clr],
                   s=899 * mtype_sz, alpha=0.17, edgecolor='none')

    ax.plot([plt_min, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [plt_min, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([plt_min + 0.01, 0.997], [plt_min + 0.01, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim([plt_min, 1 + (1 - plt_min) / 71])
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 71])

    lbl_base = "AUC in training cohort\n{}"
    xlbl = lbl_base.format(get_cohort_label(args.cohorts[0]))
    ylbl = lbl_base.format(get_cohort_label(args.cohorts[1]))
    ax.set_xlabel(xlbl, size=23, weight='semibold')
    ax.set_ylabel(ylbl, size=23, weight='semibold')

    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "auc-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_sub_comparison(auc_dfs, trnsf_dicts, pheno_dicts, conf_dfs, args):
    fig, axarr = plt.subplots(figsize=(15, 14), nrows=2, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (mtype.subtype_list()[0][1]
                           & copy_mtype).is_empty())}

    auc_dicts = {
        train_coh: {
            'Wthn': auc_dfs[train_coh]['mean'], 'Trnsf': pd.concat([
                trnsf_data[train_coh]['AUC']['mean']
                for trnsf_data in trnsf_dicts[other_coh].values()
                ])
            }
        for train_coh, other_coh in permutations(args.cohorts)
        }

    for i, auc_lbl in enumerate(['Wthn', 'Trnsf']):
        for j, coh in enumerate(args.cohorts):
            pnt_dict = dict()
            clr_dict = dict()

            for gene, auc_vec in auc_dicts[coh][auc_lbl][use_mtypes].groupby(
                    lambda mtype: mtype.get_labels()[0]):

                if len(auc_vec) > 1:
                    base_mtype = MuType({('Gene', gene): pnt_mtype})
                    base_indx = auc_vec.index.get_loc(base_mtype)

                    best_subtype = auc_vec[:base_indx].append(
                        auc_vec[(base_indx + 1):]).idxmax()
                    best_indx = auc_vec.index.get_loc(best_subtype)

                    if auc_vec[best_indx] > 0.6:
                        clr_dict[gene] = choose_gene_colour(gene)

                        base_size = np.mean(pheno_dicts[coh][base_mtype])
                        best_prop = np.mean(pheno_dicts[coh][best_subtype])
                        best_prop /= base_size

                        conf_sc = np.greater.outer(
                            conf_dfs[coh].loc[best_subtype, 'mean'],
                            conf_dfs[coh].loc[base_mtype, 'mean']
                            ).mean()

                        if conf_sc > 0.9:
                            mtype_lbl = '\n'.join(
                                get_fancy_label(best_subtype).split('\n')[1:])

                            pnt_dict[auc_vec[base_indx],
                                     auc_vec[best_indx]] = (base_size ** 0.53,
                                                            (gene, mtype_lbl))

                        else:
                            pnt_dict[auc_vec[base_indx],
                                     auc_vec[best_indx]] = (base_size ** 0.53,
                                                            (gene, ''))

                        pie_ax = inset_axes(
                            axarr[i, j],
                            width=base_size ** 0.5, height=base_size ** 0.5,
                            bbox_to_anchor=(auc_vec[base_indx],
                                            auc_vec[best_indx]),
                            bbox_transform=axarr[i, j].transData, loc=10,
                            axes_kwargs=dict(aspect='equal'), borderpad=0
                            )

                        pie_ax.pie(
                            x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                            colors=[clr_dict[gene] + (0.77, ),
                                    clr_dict[gene] + (0.29, )]
                            )

            lbl_pos = place_labels(pnt_dict)
            for (pnt_x, pnt_y), pos in lbl_pos.items():
                axarr[i, j].text(pos[0][0], pos[0][1] + 700 ** -1,
                                 pnt_dict[pnt_x, pnt_y][1][0],
                                 size=13, ha=pos[1], va='bottom')
                axarr[i, j].text(pos[0][0], pos[0][1] - 700 ** -1,
                                 pnt_dict[pnt_x, pnt_y][1][1],
                                 size=9, ha=pos[1], va='top')

                x_delta = pnt_x - pos[0][0]
                y_delta = pnt_y - pos[0][1]
                ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

                if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
                    use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
                    pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
                    lbl_gap = 0.006 / ln_lngth

                    axarr[i, j].plot([pnt_x - pnt_gap * x_delta,
                                      pos[0][0] + lbl_gap * x_delta],
                                     [pnt_y - pnt_gap * y_delta,
                                      pos[0][1] + lbl_gap * y_delta
                                      + 0.008 + 0.004 * np.sign(y_delta)],
                                     c=use_clr, linewidth=2.3, alpha=0.27)

            axarr[i, j].plot([0.48, 1], [0.5, 0.5], color='black',
                             linewidth=1.3, linestyle=':', alpha=0.71)
            axarr[i, j].plot([0.5, 0.5], [0.48, 1], color='black',
                             linewidth=1.3, linestyle=':', alpha=0.71)

            axarr[i, j].plot([0.48, 1.0005], [1, 1], color='black',
                             linewidth=1.9, alpha=0.89)
            axarr[i, j].plot([1, 1], [0.48, 1.0005], color='black',
                             linewidth=1.9, alpha=0.89)
            axarr[i, j].plot([0.49, 0.997], [0.49, 0.997], color='#550000',
                             linewidth=2.1, linestyle='--', alpha=0.41)

            axarr[i, j].set_xlim([0.48, 1.01])
            axarr[i, j].set_ylim([0.48, 1.01])
            axarr[i, j].set_xlabel("AUC using all point mutations",
                                   size=23, weight='semibold')

            if j == 1:
                axarr[i, j].set_ylabel("AUC of best found subgrouping",
                                   size=23, weight='semibold')

    fig.tight_layout(w_pad=2.9, h_pad=0.9)
    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "sub-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_cross_sub_comparison(auc_dfs, pheno_dicts, conf_dfs, args):
    fig, axarr = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (mtype.subtype_list()[0][1]
                           & copy_mtype).is_empty())}

    for i, (base_coh, other_coh) in enumerate(permutations(args.cohorts)):
        pnt_dict = dict()
        clr_dict = dict()

        for gene, auc_vec in auc_dfs[base_coh].loc[
                use_mtypes, 'mean'].groupby(
                    lambda mtype: mtype.get_labels()[0]):

            if len(auc_vec) > 1:
                base_mtype = MuType({('Gene', gene): pnt_mtype})
                base_indx = auc_vec.index.get_loc(base_mtype)

                best_subtype = auc_vec[:base_indx].append(
                    auc_vec[(base_indx + 1):]).idxmax()
                best_indx = auc_vec.index.get_loc(best_subtype)

                if auc_vec[best_indx] > 0.6:
                    clr_dict[gene] = choose_gene_colour(gene)
                    plt_x = auc_dfs[other_coh].loc[base_mtype, 'mean']
                    plt_y = auc_dfs[other_coh].loc[best_subtype, 'mean']

                    base_size = np.mean(pheno_dicts[other_coh][base_mtype])
                    best_prop = np.mean(pheno_dicts[other_coh][best_subtype])
                    best_prop /= base_size

                    conf_sc = np.greater.outer(
                        conf_dfs[other_coh].loc[best_subtype, 'mean'],
                        conf_dfs[other_coh].loc[base_mtype, 'mean']
                        ).mean()

                    if conf_sc > 0.8:
                        mtype_lbl = '\n'.join(
                            get_fancy_label(best_subtype).split('\n')[1:])

                        pnt_dict[plt_x, plt_y] = (base_size ** 0.53,
                                                  (gene, mtype_lbl))

                    else:
                        pnt_dict[plt_x, plt_y] = (base_size ** 0.53,
                                                  (gene, ''))

                    pie_ax = inset_axes(axarr[i],
                                        width=base_size ** 0.5,
                                        height=base_size ** 0.5,
                                        bbox_to_anchor=(plt_x, plt_y),
                                        bbox_transform=axarr[i].transData,
                                        loc=10, borderpad=0,
                                        axes_kwargs=dict(aspect='equal'))

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.29, 0],
                               colors=[clr_dict[gene] + (0.77, ),
                                       clr_dict[gene] + (0.29, )])

        # figure out where to place the labels for each point, and plot them
        lbl_pos = place_labels(pnt_dict)
        for (pnt_x, pnt_y), pos in lbl_pos.items():
            axarr[i].text(pos[0][0], pos[0][1] + 700 ** -1,
                          pnt_dict[pnt_x, pnt_y][1][0],
                          size=13, ha=pos[1], va='bottom')
            axarr[i].text(pos[0][0], pos[0][1] - 700 ** -1,
                          pnt_dict[pnt_x, pnt_y][1][1],
                          size=9, ha=pos[1], va='top')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]
            ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

            # if the label is sufficiently far away from its point...
            if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
                use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
                pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
                lbl_gap = 0.006 / ln_lngth

                axarr[i].plot([pnt_x - pnt_gap * x_delta,
                               pos[0][0] + lbl_gap * x_delta],
                              [pnt_y - pnt_gap * y_delta,
                               pos[0][1] + lbl_gap * y_delta
                               + 0.008 + 0.004 * np.sign(y_delta)],
                              c=use_clr, linewidth=2.3, alpha=0.27)

        axarr[i].plot([0.48, 1], [0.5, 0.5], color='black',
                      linewidth=1.3, linestyle=':', alpha=0.71)
        axarr[i].plot([0.5, 0.5], [0.48, 1], color='black',
                      linewidth=1.3, linestyle=':', alpha=0.71)

        axarr[i].plot([0.48, 1.0005], [1, 1], color='black',
                      linewidth=1.9, alpha=0.89)
        axarr[i].plot([1, 1], [0.48, 1.0005], color='black',
                      linewidth=1.9, alpha=0.89)
        axarr[i].plot([0.49, 0.997], [0.49, 0.997], color='#550000',
                      linewidth=2.1, linestyle='--', alpha=0.41)

        xlbl = "AUC in {} using\nall point mutations".format(
            get_cohort_label(other_coh))
        ylbl = "AUC in {} of best\nfound subgrouping in {}".format(
            get_cohort_label(other_coh), get_cohort_label(base_coh))

        axarr[i].set_xlabel(xlbl, size=19, weight='semibold')
        axarr[i].set_ylabel(ylbl, size=19, weight='semibold')
        axarr[i].set_xlim([0.48, 1.01])
        axarr[i].set_ylim([0.48, 1.01])

    fig.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "cross-sub-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_metrics_comparison(auc_dfs, pheno_dicts, args):
    fig, axarr = plt.subplots(figsize=(19, 6), nrows=1, ncols=3)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if not isinstance(mtype, RandomType)}

    plt_min = 0.83
    for ax, auc_lbl in zip(axarr, ['mean', 'all', 'CV']):
        ax.set_title(auc_lbl, size=25, weight='semibold')

        if auc_lbl == 'CV':
            auc_vals1 = auc_dfs[args.cohorts[0]]['CV'].apply(
                np.quantile, q=0.25)
            auc_vals2 = auc_dfs[args.cohorts[1]]['CV'].apply(
                np.quantile, q=0.25)

        else:
            auc_vals1 = auc_dfs[args.cohorts[0]][auc_lbl]
            auc_vals2 = auc_dfs[args.cohorts[1]][auc_lbl]

        for mtype in use_mtypes:
            use_gene = mtype.get_labels()[0]
            gene_clr = choose_gene_colour(use_gene)

            plt_min = min(plt_min,
                          auc_vals1[mtype] - 0.02, auc_vals2[mtype] - 0.02)
            mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                        * np.mean(pheno_dicts[args.cohorts[1]][mtype])) ** 0.5

            ax.scatter(auc_vals1[mtype], auc_vals2[mtype], c=[gene_clr],
                       s=408 * mtype_sz, alpha=0.17, edgecolor='none')

        rmse_val = ((auc_vals1[use_mtypes]
                     - auc_vals2[use_mtypes]) ** 2).mean() ** 0.5
        ax.text(0.96, 0.08, "RMSE: {:.3f}".format(rmse_val),
                size=17, ha='right', va='bottom', transform=ax.transAxes)

        mae_val = (auc_vals1[use_mtypes] - auc_vals2[use_mtypes]).abs().mean()
        ax.text(0.96, 0.03, "MAE: {:.3f}".format(mae_val),
                size=17, ha='right', va='bottom', transform=ax.transAxes)

    for ax in axarr:
        ax.plot([plt_min, 1], [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], [plt_min, 1],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot([plt_min, 1.0005], [1, 1],
                color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], [plt_min, 1.0005],
                color='black', linewidth=1.9, alpha=0.89)
        ax.plot([plt_min + 0.01, 0.997], [plt_min + 0.01, 0.997],
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        ax.set_xlim([plt_min, 1 + (1 - plt_min) / 53])
        ax.set_ylim([plt_min, 1 + (1 - plt_min) / 53])

    fig.text(0.5, 0, "AUC in training cohort\n{}".format(args.cohorts[0]),
             fontsize=23, weight='semibold', ha='center', va='top')
    fig.text(0.03, 0.5, "AUC in training cohort\n{}".format(args.cohorts[1]),
             fontsize=23, weight='semibold', rotation=90,
             ha='right', va='center')

    fig.tight_layout(w_pad=0.3)
    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "metrics-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_transfer_aucs(trnsf_dicts, auc_dfs, pheno_dicts, args):
    fig, axarr = plt.subplots(figsize=(15, 14), nrows=2, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if not isinstance(mtype, RandomType)}
    plt_min = 0.83

    for j, (train_coh, other_coh) in enumerate(permutations(args.cohorts)):
        trnsf_aucs = pd.concat([
            trnsf_data[other_coh]['AUC']['mean']
            for trnsf_data in trnsf_dicts[train_coh].values()
            ])

        other_aucs = pd.concat([
            trnsf_data[train_coh]['AUC']['mean']
            for trnsf_data in trnsf_dicts[other_coh].values()
            ])

        for mtype in use_mtypes:
            use_gene = mtype.get_labels()[0]
            gene_clr = choose_gene_colour(use_gene)

            train_auc = auc_dfs[train_coh].loc[mtype, 'mean']
            trnsf_auc = trnsf_aucs[mtype]
            other_auc = other_aucs[mtype]
            plt_min = min(
                plt_min, train_auc - 0.01, trnsf_auc - 0.01, other_auc - 0.01)

            mtype_sz = (np.mean(pheno_dicts[args.cohorts[0]][mtype])
                        * np.mean(pheno_dicts[args.cohorts[1]][mtype])) ** 0.5

            axarr[0, j].scatter(train_auc, trnsf_auc,
                                c=[gene_clr], s=397 * mtype_sz,
                                alpha=0.17, edgecolor='none')
            axarr[1, j].scatter(train_auc, other_auc,
                                c=[gene_clr], s=397 * mtype_sz,
                                alpha=0.17, edgecolor='none')

        for i in range(2):
            axarr[i, j].plot([plt_min, 1], [0.5, 0.5],
                             color='black', linewidth=1.3, linestyle=':',
                             alpha=0.71)
            axarr[i, j].plot([0.5, 0.5], [plt_min, 1],
                             color='black', linewidth=1.3, linestyle=':',
                             alpha=0.71)

            axarr[i, j].plot([plt_min, 1.0005], [1, 1],
                             color='black', linewidth=1.9, alpha=0.89)
            axarr[i, j].plot([1, 1], [plt_min, 1.0005],
                             color='black', linewidth=1.9, alpha=0.89)
            axarr[i, j].plot([plt_min + 0.01, 0.997], [plt_min + 0.01, 0.997],
                             color='#550000', linewidth=2.1, linestyle='--',
                             alpha=0.41)

            axarr[i, j].set_xlim([plt_min, 1 + (1 - plt_min) / 53])
            axarr[i, j].set_ylim([plt_min, 1 + (1 - plt_min) / 53])

        axarr[1, j].set_xlabel("AUC within cohort\n{}".format(train_coh),
                               size=23, weight='semibold')

        axarr[0, j].set_ylabel(
            "AUC when transferred to cohort\n{}".format(other_coh),
            size=23, weight='semibold'
            )
        axarr[1, j].set_ylabel(
            "AUC when transferred to cohort\n{}".format(train_coh),
            size=23, weight='semibold'
            )

    fig.tight_layout(w_pad=2.9, h_pad=0.9)
    plt.savefig(os.path.join(plot_dir, '__'.join(args.cohorts),
                             "transfer-aucs_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_subtype_comparison(auc_dfs, pheno_dicts, conf_dfs, plt_gene, args):
    fig, (auc_ax, conf_ax) = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)

    use_mtypes = {mtype for mtype in (auc_dfs[args.cohorts[0]].index
                                      & auc_dfs[args.cohorts[1]].index)
                  if (not isinstance(mtype, RandomType)
                      and (mtype.subtype_list()[0][1]
                           & copy_mtype).is_empty()
                      and mtype.get_labels()[0] == plt_gene)}

    base_mtype = MuType({('Gene', plt_gene): pnt_mtype})
    auc_min = -0.005
    auc_max = 0.05
    conf_min = 0.43

    for mtype in use_mtypes:
        conf_sc1 = np.greater.outer(
            conf_dfs[args.cohorts[0]].loc[mtype, 'mean'],
            conf_dfs[args.cohorts[0]].loc[base_mtype, 'mean']
            ).mean()

        conf_sc2 = np.greater.outer(
            conf_dfs[args.cohorts[1]].loc[mtype, 'mean'],
            conf_dfs[args.cohorts[1]].loc[base_mtype, 'mean']
            ).mean()

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
                      if ((mtype.subtype_list()[0][1] & copy_mtype).is_empty()
                          and mtype.get_labels()[0] == plt_gene)}

        for mtype in use_mtypes:
            train_auc = auc_dfs[train_coh].loc[mtype, 'mean']
            plt_min = min(plt_min, train_auc - 0.01, trnsf_aucs[mtype] - 0.01)

            conf_sc = np.greater.outer(
                conf_dfs[train_coh].loc[mtype, 'mean'],
                conf_dfs[train_coh].loc[base_mtype, 'mean']
                ).mean()

            if mtype == base_mtype:
                use_mrk = 'X'
                use_sz = 5305 * np.mean(pheno_dicts[train_coh][mtype])
                use_alpha = 0.47

            else:
                use_mrk = 'o'
                use_sz = 1507 * np.mean(pheno_dicts[train_coh][mtype])
                use_alpha = 0.23

            ax.scatter(train_auc, trnsf_aucs[mtype],
                       marker=use_mrk, s=use_sz, alpha=use_alpha,
                       edgecolor='none')

        x_lbl = "AUC in training cohort\n{}".format(
            get_cohort_label(train_coh))
        y_lbl = "AUC in transfer cohort\n{}".format(
            get_cohort_label(other_coh))

        ax.set_xlabel(x_lbl, size=23, weight='semibold')
        ax.set_ylabel(y_lbl, size=23, weight='semibold')

    for ax in axarr:
        ax.plot([plt_min, 1], [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], [plt_min, 1],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot([plt_min, 1.0005], [1, 1],
                color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], [plt_min, 1.0005],
                color='black', linewidth=1.9, alpha=0.89)
        ax.plot([plt_min, 0.999], [plt_min, 0.999],
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        ax.set_xlim([plt_min, 1 + (1 - plt_min) / 77])
        ax.set_ylim([plt_min, 1 + (1 - plt_min) / 77])

    fig.tight_layout(w_pad=2.3)
    plt.savefig(
        os.path.join(plot_dir, '__'.join(args.cohorts),
                     "{}__subtype-transfer_{}.svg".format(
                         plt_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance of a model in predicting the presence of "
        "enumerated mutations across and between a pair of tumour cohorts."
        )

    parser.add_argument('cohorts', nargs=2, help="which TCGA cohorts to use")
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('--genes', '-g', nargs='*',
                        help='a list of mutated genes', type=str)
    args = parser.parse_args()

    phn_dicts = {coh: dict() for coh in args.cohorts}
    auc_dict = {coh: dict() for coh in args.cohorts}
    conf_dict = {coh: dict() for coh in args.cohorts}
    trnsf_dicts = {coh: dict() for coh in args.cohorts}

    for coh in args.cohorts:
        use_src = choose_source(coh)

        out_datas = [
            out_file.parts[-2:]
            for out_file in Path(base_dir).glob(os.path.join(
                "{}__{}__samps-*".format(use_src, coh),
                "trnsf-vals__*__{}.p.gz".format(args.classif)
                ))
            ]

        out_list = pd.DataFrame([
            {'Samps': int(out_data[0].split('__samps-')[1]),
             'Levels': '__'.join(out_data[1].split(
                 'trnsf-vals__')[1].split('__')[:-1])}
            for out_data in out_datas
            ])

        if out_list.shape[0] == 0:
            raise ValueError("No experiment output found for "
                             "cohort `{}` !".format(coh))

        out_use = out_list.groupby('Levels')['Samps'].min()
        if 'Exon__Location__Protein' not in out_use.index:
            raise ValueError(
                "Cannot compare AUCs until this experiment is run with "
                "mutation levels `Exon__Location__Protein` which tests "
                "genes' base mutations!"
                )

        for lvls, ctf in out_use.iteritems():
            out_tag = "{}__{}__samps-{}".format(use_src, coh, ctf)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-pheno__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                phn_dicts[coh].update(pickle.load(f))

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-aucs__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                auc_dict[coh][lvls] = pd.DataFrame.from_dict(pickle.load(f))

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-conf__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                conf_dict[coh][lvls] = pd.DataFrame.from_dict(pickle.load(f))

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-trnsf__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                trnsf_dicts[coh][lvls] = pickle.load(f)

    # creates directory where plots will be saved
    os.makedirs(os.path.join(plot_dir, '__'.join(args.cohorts)),
                exist_ok=True)

    auc_dfs = {coh: pd.concat(auc_dict[coh].values()) for coh in args.cohorts}
    conf_dfs = {coh: pd.concat(conf_dict[coh].values())
                for coh in args.cohorts}

    plot_auc_comparison(auc_dfs, phn_dicts, args)
    plot_sub_comparison(auc_dfs, trnsf_dicts, phn_dicts, conf_dfs, args)
    plot_cross_sub_comparison(auc_dfs, phn_dicts, conf_dfs, args)
    plot_metrics_comparison(auc_dfs, phn_dicts, args)
    plot_transfer_aucs(trnsf_dicts, auc_dfs, phn_dicts, args)

    if args.genes is not None:
        for gene in args.genes:
            plot_subtype_comparison(auc_dfs, phn_dicts, conf_dfs, gene, args)
            plot_subtype_transfer(auc_dfs, trnsf_dicts, phn_dicts, conf_dfs,
                                  gene, args)


if __name__ == '__main__':
    main()

