
from ..utilities.mutations import (pnt_mtype, shal_mtype, deep_mtype,
                                   copy_mtype, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir
from ..subgrouping_test import train_cohorts
from .utils import siml_fxs, remove_pheno_dups, get_mut_ex, get_mcomb_lbl
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.colour_maps import simil_cmap, variant_clrs
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

from functools import reduce
from operator import add
from itertools import combinations as combn
from itertools import permutations as permt
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'point')


def plot_divergent_types(pred_dfs, pheno_dicts, auc_lists,
                         cdata_dict, args, siml_metric):
    fig, ax = plt.subplots(figsize=(12, 7))

    divg_dfs = dict()
    gn_dict = dict()
    rst_dict = dict()
    plot_dict = dict()
    clr_dict = dict()
    line_dict = dict()
    annt_dict = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined subsets of single point mutations...
    for (src, coh), auc_list in auc_lists.items():
        use_combs = remove_pheno_dups({
            mut for mut, auc_val in auc_list.iteritems()
            if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                and len(mut.mtypes) == 1 and get_mut_ex(mut) == args.ex_lbl
                and all(pnt_mtype != get_subtype(mtype)
                        for mtype in mut.mtypes)
                and all(pnt_mtype.is_supertype(get_subtype(mtype))
                        for mtype in mut.mtypes))
            }, pheno_dicts[src, coh])

        if not use_combs:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {get_label(mcomb) for mcomb in use_combs}
        train_samps = cdata_dict[src, coh].get_train_samples()
        siml_vals = dict()

        all_mtypes = {
            gene: MuType({('Gene', gene): base_mtree[gene].allkey()})
            for gene in use_genes
            }

        if args.ex_lbl == 'IsoShal':
            for gene in use_genes:
                all_mtypes[gene] -= MuType({('Gene', gene): shal_mtype})

        all_phns = {
            gene: np.array(cdata_dict[src, coh].train_pheno(all_mtype))
            for gene, all_mtype in all_mtypes.items()
            }

        for mcomb in use_combs:
            use_mtype = tuple(mcomb.mtypes)[0]
            cur_gene = get_label(mcomb)
            use_preds = pred_dfs[src, coh].loc[mcomb, train_samps]

            if (src, coh, cur_gene) not in gn_dict:
                gn_dict[src, coh, cur_gene] = np.array(
                    cdata_dict[src, coh].train_pheno(
                        MuType({('Gene', cur_gene): pnt_mtype}))
                    )

            rst_dict[src, coh, mcomb] = np.array(
                cdata_dict[src, coh].train_pheno(
                    ExMcomb(mcomb.all_mtype,
                            (mcomb.all_mtype & pnt_mtype) - use_mtype)
                    )
                )

            assert not (pheno_dicts[src, coh][mcomb]
                        & rst_dict[src, coh, mcomb]).any()

            if rst_dict[src, coh, mcomb].sum() >= 10:
                siml_vals[mcomb] = siml_fxs[siml_metric](
                    use_preds.loc[~all_phns[cur_gene]],
                    use_preds.loc[pheno_dicts[src, coh][mcomb]],
                    use_preds.loc[rst_dict[src, coh, mcomb]]
                    )

        divg_df = pd.DataFrame({'Divg': siml_vals})
        divg_df['AUC'] = auc_list[divg_df.index]
        divg_df = divg_df.sort_values(by='Divg')
        divg_mcombs = set()
        coh_lbl = get_cohort_label(coh)

        for cur_gene, divg_vals in divg_df.groupby(get_label):
            clr_dict[cur_gene] = None
            plt_size = np.mean(gn_dict[src, coh, cur_gene]) ** 0.5
            divg_stat = (divg_vals.Divg < 0.5).any()

            for i, (mcomb, (divg_val, auc_val)) in enumerate(
                    divg_vals.iterrows()):
                line_dict[auc_val, divg_val] = cur_gene

                if not (divg_vals[:i].AUC > auc_val).any():
                    divg_mcombs |= {mcomb}

                    if divg_val < 0.5:
                        plot_dict[auc_val, divg_val] = [
                            plt_size, ("{} in {}".format(cur_gene, coh_lbl),
                                       get_mcomb_lbl(mcomb))
                            ]

                    else:
                        plot_dict[auc_val, divg_val] = [plt_size, ('', '')]

                else:
                    plot_dict[auc_val, divg_val] = [plt_size, ('', '')]

            if not divg_stat:
                line_dict[divg_vals.AUC[0], divg_vals.Divg[0]] = cur_gene

                plot_dict[divg_vals.AUC[0], divg_vals.Divg[0]] = [
                    plt_size, ("{} in {}".format(cur_gene, coh_lbl), '')]

        divg_dfs[src, coh] = divg_df.loc[divg_mcombs]

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    #TODO: make this scale better for smaller number of points?
    size_mult = 4.7 * sum(divg_df.shape[0]
                          for divg_df in divg_dfs.values()) ** -0.47
    for k in plot_dict:
        plot_dict[k][0] *= size_mult

    for k in line_dict:
        line_dict[k] = {'c': clr_dict[line_dict[k]]}

    adj_trans = lambda x: ax.transData.inverted().transform(x)
    xadj, yadj = adj_trans([1, 1]) - adj_trans([0, 0])
    xy_adj = xadj / yadj

    for (src, coh), divg_df in divg_dfs.items():
        for mcomb, (divg_val, auc_val) in divg_df.iterrows():
            cur_gene = get_label(mcomb)
            plt_size = plot_dict[auc_val, divg_val][0] / 3.1

            plt_prop = np.mean(pheno_dicts[src, coh][mcomb])
            plt_prop /= np.mean(gn_dict[src, coh, cur_gene])
            rst_prop = np.mean(rst_dict[src, coh, mcomb])
            rst_prop /= np.mean(gn_dict[src, coh, cur_gene])

            pie_bbox = (auc_val - plt_size * xy_adj / 2,
                        divg_val - plt_size / 2, plt_size * xy_adj, plt_size)

            pie_ax = inset_axes(ax, width='100%', height='100%',
                                bbox_to_anchor=pie_bbox,
                                bbox_transform=ax.transData,
                                axes_kwargs=dict(aspect='equal'), borderpad=0)

            pie_ax.pie(x=[plt_prop, round(1 - plt_prop - rst_prop, 3),
                          rst_prop],
                       colors=[clr_dict[cur_gene] + (0.67,),
                               clr_dict[cur_gene] + (0,),
                               clr_dict[cur_gene] + (0.11,)],
                       explode=[0.29, 0, 0], startangle=270, normalize=True)

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]

    ymin = min(divg_df.Divg.min() for divg_df in divg_dfs.values())
    ymax = max(divg_df.Divg.max() for divg_df in divg_dfs.values())
    yrng = ymax - ymin
    ylims = [ymin - yrng / 9.1, ymax + yrng / 29]

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot(xlims, [0, 0],
            color='black', linewidth=1.11, linestyle='--', alpha=0.67)
    ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

    ax.set_xlabel("Isolated Classification Accuracy", size=21, weight='bold')
    ax.set_ylabel("Inferred Similarity to\nRemaining Point Mutations",
                  size=21, weight='bold')

    for k in np.linspace(args.auc_cutoff, 0.99, 200):
        if (k, 0) not in plot_dict:
            plot_dict[k, 0] = [1 / 11, ('', '')]

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax, plt_lims=[xlims, ylims],
                                       font_size=9, line_dict=line_dict,
                                       linewidth=1.23, alpha=0.37)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    plt.savefig(os.path.join(plot_dir,
                             "{}_{}-divergent-types_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()
    return annt_dict


def plot_divergent_pairs(pred_dfs, pheno_dicts, auc_lists,
                         cdata_dict, args, siml_metric):
    fig, ax = plt.subplots(figsize=(12, 7))

    divg_lists = dict()
    plot_dict = dict()
    clr_dict = dict()
    line_dict = dict()
    size_dict = dict()
    annt_dict = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined and subsets of point mutations...
    for (src, coh), auc_list in auc_lists.items():
        use_combs = remove_pheno_dups({
            mut for mut, auc_val in auc_list.iteritems()
            if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                and len(mut.mtypes) == 1 and get_mut_ex(mut) == args.ex_lbl
                and all(pnt_mtype.is_supertype(get_subtype(mtype))
                        for mtype in mut.mtypes))
            }, pheno_dicts[src, coh])

        # find all pairs of subgroupings from the same gene that are disjoint
        # either by construction or by phenotype
        use_pairs = {
            tuple(sorted([mcomb1, mcomb2]))
            for mcomb1, mcomb2 in combn(use_combs, 2)
            if (get_label(mcomb1) == get_label(mcomb2)
                and (all((mtp1 & mtp2).is_empty()
                         for mtp1, mtp2 in product(mcomb1.mtypes,
                                                   mcomb2.mtypes))
                     or not (pheno_dicts[src, coh][mcomb1]
                             & pheno_dicts[src, coh][mcomb2]).any()))
            }

        # skip this dataset for plotting if we cannot find any such pairs
        if not use_pairs:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {get_label(mcomb)
                     for comb_pair in use_pairs for mcomb in comb_pair}

        all_mtypes = {
            gene: MuType({('Gene', gene): base_mtree[gene].allkey()})
            for gene in use_genes
            }

        if args.ex_lbl == 'IsoShal':
            for gene in use_genes:
                all_mtypes[gene] -= MuType({('Gene', gene): shal_mtype})

        all_phns = {
            gene: np.array(cdata_dict[src, coh].train_pheno(all_mtype))
            for gene, all_mtype in all_mtypes.items()
            }

        train_samps = cdata_dict[src, coh].get_train_samples()
        pair_combs = set(reduce(add, use_pairs))
        use_preds = pred_dfs[src, coh].loc[pair_combs, train_samps]
        map_args = []

        wt_vals = {mcomb: use_preds.loc[mcomb][~all_phns[get_label(mcomb)]]
                   for mcomb in pair_combs}
        mut_vals = {mcomb: use_preds.loc[mcomb, pheno_dicts[src, coh][mcomb]]
                    for mcomb in pair_combs}

        if siml_metric == 'mean':
            wt_means = {mcomb: vals.mean() for mcomb, vals in wt_vals.items()}
            mut_means = {mcomb: vals.mean()
                         for mcomb, vals in mut_vals.items()}

            map_args += [
                (wt_vals[mcomb1], mut_vals[mcomb1],
                 use_preds.loc[mcomb1, pheno_dicts[src, coh][mcomb2]],
                 wt_means[mcomb1], mut_means[mcomb1], None)
                for mcombs in use_pairs for mcomb1, mcomb2 in permt(mcombs)
                ]

        elif siml_metric == 'ks':
            base_dists = {
                mcomb: ks_2samp(wt_vals[mcomb], mut_vals[mcomb],
                                alternative='greater').statistic
                for mcomb in pair_combs
                }

            map_args += [
                (wt_vals[mcomb1], mut_vals[mcomb1],
                 use_preds.loc[mcomb1, pheno_dicts[src, coh][mcomb2]],
                 base_dists[mcomb1])
                for mcombs in use_pairs for mcomb1, mcomb2 in permt(mcombs)
                ]

        if siml_metric == 'mean':
            chunk_size = int(len(map_args) / (11 * args.cores)) + 1
        elif siml_metric == 'ks':
            chunk_size = int(len(map_args) / (17 * args.cores)) + 1

        pool = mp.Pool(args.cores)
        siml_list = pool.starmap(siml_fxs[siml_metric], map_args, chunk_size)
        pool.close()
        siml_df = pd.DataFrame(dict(
            zip(use_pairs, zip(siml_list[::2], siml_list[1::2])))).transpose()

        divg_list = siml_df.max(axis=1).sort_values()
        divg_indx = {mcomb: None for mcomb in pair_combs}
        divg_pairs = set()

        for mcomb1, mcomb2 in divg_list.index:
            if divg_indx[mcomb1] is None and divg_indx[mcomb2] is None:
                divg_indx[mcomb1] = mcomb2
                divg_indx[mcomb2] = mcomb1
                divg_pairs |= {(mcomb1, mcomb2)}

            if not any(v is None for v in divg_indx.values()):
                break

        for mcomb1, mcomb2 in divg_pairs:
            pair_k = src, coh, mcomb1, mcomb2

            size_dict[pair_k] = (np.mean(pheno_dicts[src, coh][mcomb1])
                                 * np.mean(pheno_dicts[src, coh][mcomb2]))
            size_dict[pair_k] **= 0.5

            plot_dict[auc_list[[mcomb1, mcomb2]].min(),
                      divg_list[mcomb1, mcomb2]] = [3.1 * size_dict[pair_k],
                                                    ('', '')]

        pair_df = pd.DataFrame({
            'Divg': divg_list.loc[divg_pairs],
            'AUC': [auc_list[list(mcombs)].min() for mcombs in divg_pairs]
            })
        pair_df['Score'] = pair_df.Divg * (1 - pair_df.AUC ** 2)
        pair_df = pair_df.sort_values(by='Score')

        divg_lists[src, coh] = divg_list.loc[divg_pairs]
        annt_dict[src, coh] = set()

        for gene, pair_vals in pair_df.groupby(
                lambda mcombs: get_label(mcombs[0])):
            clr_dict[gene] = None

            lbl_pair = pair_vals.index[0]
            annt_dict[src, coh] |= set(pair_vals.index[:5])
            pair_lbl = '\nvs.\n'.join([get_mcomb_lbl(mcomb)
                                       for mcomb in lbl_pair])

            plt_tupl = auc_list[list(lbl_pair)].min(), divg_list[lbl_pair]
            line_dict[plt_tupl] = src, coh, gene

            plot_dict[plt_tupl] = [3.1 * size_dict[(src, coh, *lbl_pair)],
                                   ("{} in {}".format(gene,
                                                      get_cohort_label(coh)),
                                    pair_lbl)]

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    size_mult = 6107 * sum(
        len(divg_list) for divg_list in divg_lists.values()) ** -0.31

    for k in line_dict:
        line_dict[k] = {'c': clr_dict[line_dict[k][-1]]}
    for k in size_dict:
        size_dict[k] *= size_mult

    for (src, coh), divg_list in divg_lists.items():
        for (mcomb1, mcomb2), divg_val in divg_list.iteritems():
            cur_gene = get_label(mcomb1)

            ax.scatter(auc_lists[src, coh][[mcomb1, mcomb2]].min(), divg_val,
                       s=size_dict[src, coh, mcomb1, mcomb2],
                       c=[clr_dict[cur_gene]],
                       alpha=0.31, edgecolor='none')

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]

    ymin = min(divg_list.min() for divg_list in divg_lists.values())
    ymax = max(divg_list.max() for divg_list in divg_lists.values())
    yrng = ymax - ymin
    ylims = [ymin - yrng / 11, ymax + yrng / 41]

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot(xlims, [0, 0],
            color='black', linewidth=1.11, linestyle='--', alpha=0.67)
    ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

    ax.set_xlabel("Minimum Classification Accuracy", size=21, weight='bold')
    ax.set_ylabel("Maximum Inferred Similarity", size=21, weight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[xlims, ylims],
                                       font_size=10, line_dict=line_dict,
                                       linewidth=1.7, alpha=0.41)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    plt.savefig(os.path.join(plot_dir,
                             "{}_{}-divergent-pairs_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()
    return annt_dict


def plot_orthogonal_scores(plt_mcomb1, plt_mcomb2, pred_df, auc_vals,
                           pheno_dict, cdata, data_tag, args, siml_metric):
    fig, ((mcomb2_ax, sctr_ax), (crnr_ax, mcomb1_ax)) = plt.subplots(
        figsize=(13, 12), nrows=2, ncols=2,
        gridspec_kw=dict(height_ratios=[4, 1], width_ratios=[1, 4])
        )

    use_gene = {get_label(mtype) for mcomb in [plt_mcomb1, plt_mcomb2]
                for mtype in mcomb.mtypes}
    assert len(use_gene) == 1
    use_gene = tuple(use_gene)[0]

    base_mtree = tuple(cdata.mtrees.values())[0]
    all_mtype = MuType({('Gene', use_gene): base_mtree[use_gene].allkey()})
    if args.ex_lbl == 'IsoShal':
        all_mtype -= MuType({('Gene', use_gene): shal_mtype})

    use_preds = pred_df.loc[[plt_mcomb1, plt_mcomb2],
                            cdata.get_train_samples()].T
    use_preds.columns = ['Subg1', 'Subg2']

    x_min, y_min = use_preds.min()
    x_max, y_max = use_preds.max()
    x_rng, y_rng = x_max - x_min, y_max - y_min
    xlims = x_min - x_rng / 31, x_max + x_rng / 31
    ylims = y_min - y_rng / 31, y_max + y_rng / 31

    all_phn = np.array(cdata.train_pheno(all_mtype))
    use_preds['Phn'] = np.array(['Other' if phn else 'WT' for phn in all_phn])
    use_preds.loc[pheno_dict[plt_mcomb1], 'Phn'] = 'Subg1'
    use_preds.loc[pheno_dict[plt_mcomb2], 'Phn'] = 'Subg2'

    for ax in sctr_ax, mcomb1_ax, mcomb2_ax:
        ax.grid(alpha=0.47, linewidth=0.9)

    use_clrs = {'WT': variant_clrs['WT'],
                'Subg1': '#080097', 'Subg2': '#00847F', 'Other': 'black'}
    mcomb_lbls = [get_mcomb_lbl(mcomb) for mcomb in [plt_mcomb1, plt_mcomb2]]

    sctr_ax.plot(use_preds.loc[use_preds.Phn == 'WT', 'Subg1'],
                 use_preds.loc[use_preds.Phn == 'WT', 'Subg2'],
                 marker='o', markersize=6, linewidth=0, alpha=0.19,
                 mfc=use_clrs['WT'], mec='none')

    sctr_ax.plot(use_preds.loc[use_preds.Phn == 'Subg1', 'Subg1'],
                 use_preds.loc[use_preds.Phn == 'Subg1', 'Subg2'],
                 marker='o', markersize=9, linewidth=0, alpha=0.23,
                 mfc=use_clrs['Subg1'], mec='none')

    sctr_ax.plot(use_preds.loc[use_preds.Phn == 'Subg2', 'Subg1'],
                 use_preds.loc[use_preds.Phn == 'Subg2', 'Subg2'],
                 marker='o', markersize=9, linewidth=0, alpha=0.23,
                 mfc=use_clrs['Subg2'], mec='none')

    sctr_ax.plot(use_preds.loc[use_preds.Phn == 'Other', 'Subg1'],
                 use_preds.loc[use_preds.Phn == 'Other', 'Subg2'],
                 marker='o', markersize=8, linewidth=0, alpha=0.31,
                 mfc='none', mec=use_clrs['Other'])

    sctr_ax.text(0.98, 0.03,
                 "Subgrouping 1:\n{} mutants\n{}".format(use_gene,
                                                         mcomb_lbls[0]),
                 size=15, c=use_clrs['Subg1'], ha='right', va='bottom',
                 transform=sctr_ax.transAxes)
    sctr_ax.text(0.03, 0.98,
                 "Subgrouping 2:\n{} mutants\n{}".format(use_gene,
                                                         mcomb_lbls[1]),
                 size=15, c=use_clrs['Subg2'], ha='left', va='top',
                 transform=sctr_ax.transAxes)

    sns.violinplot(data=use_preds, y='Phn', x='Subg1', ax=mcomb1_ax,
                   order=['WT', 'Subg1', 'Subg2', 'Other'],
                   palette=use_clrs, orient='h', linewidth=0, cut=0)

    sns.violinplot(data=use_preds, x='Phn', y='Subg2', ax=mcomb2_ax,
                   order=['WT', 'Subg2', 'Subg1', 'Other'],
                   palette=use_clrs, orient='v', linewidth=0, cut=0)

    for mcomb_ax in mcomb1_ax, mcomb2_ax:
        for i in range(3):
            mcomb_ax.get_children()[i * 2].set_alpha(0.61)
            mcomb_ax.get_children()[i * 2].set_linewidth(0)

        mcomb_ax.get_children()[6].set_edgecolor('black')
        mcomb_ax.get_children()[6].set_facecolor('white')
        mcomb_ax.get_children()[6].set_linewidth(1.3)
        mcomb_ax.get_children()[6].set_alpha(0.61)

    sctr_ax.set_xticklabels([])
    sctr_ax.set_yticklabels([])

    mcomb1_ax.set_xlabel("Subgrouping Task 1\nPredicted Scores",
                         size=23, weight='semibold')
    mcomb1_ax.yaxis.label.set_visible(False)
    mcomb1_ax.set_yticklabels([
        "Wild-Type", "Subg1", "Subg2", "Other\n{}\nMuts".format(use_gene)])

    mcomb2_ax.set_ylabel("Subgrouping Task 2\nPredicted Scores",
                         size=23, weight='semibold')
    mcomb2_ax.xaxis.label.set_visible(False)
    mcomb2_ax.set_xticklabels(mcomb2_ax.get_xticklabels(),
                              rotation=31, ha='right')

    mcomb1_ax.text(1, 0.83, "n={}".format((use_preds.Phn == 'WT').sum()),
                   size=13, ha='left', transform=mcomb1_ax.transAxes,
                   clip_on=False)
    mcomb1_ax.text(1, 0.58, "n={}".format((use_preds.Phn == 'Subg1').sum()),
                   size=13, ha='left', transform=mcomb1_ax.transAxes,
                   clip_on=False)
    mcomb1_ax.text(1, 0.33, "n={}".format((use_preds.Phn == 'Subg2').sum()),
                   size=13, ha='left', transform=mcomb1_ax.transAxes,
                   clip_on=False)
    mcomb1_ax.text(1, 0.08, "n={}".format((use_preds.Phn == 'Other').sum()),
                   size=13, ha='left', transform=mcomb1_ax.transAxes,
                   clip_on=False)

    mcomb2_ax.text(1 / 8, 1, "n={}".format((use_preds.Phn == 'WT').sum()),
                   size=13, rotation=31, ha='left',
                   transform=mcomb2_ax.transAxes, clip_on=False)
    mcomb2_ax.text(3 / 8, 1, "n={}".format((use_preds.Phn == 'Subg2').sum()),
                   size=13, rotation=31, ha='left',
                   transform=mcomb2_ax.transAxes, clip_on=False)
    mcomb2_ax.text(5 / 8, 1, "n={}".format((use_preds.Phn == 'Subg1').sum()),
                   size=13, rotation=31, ha='left',
                   transform=mcomb2_ax.transAxes, clip_on=False)
    mcomb2_ax.text(7 / 8, 1, "n={}".format((use_preds.Phn == 'Other').sum()),
                   size=13, rotation=31, ha='left',
                   transform=mcomb2_ax.transAxes, clip_on=False)

    crnr_ax.text(0.95, 0.59, "(AUC1: {:.3f})".format(auc_vals[plt_mcomb1]),
                 size=11, ha='right', transform=crnr_ax.transAxes,
                 clip_on=False)
    crnr_ax.text(0, 0.55, "(AUC2: {:.3f})".format(auc_vals[plt_mcomb2]),
                 size=11, rotation=31, ha='right',
                 transform=crnr_ax.transAxes, clip_on=False)

    wt_vals = use_preds.loc[use_preds.Phn == 'WT', ['Subg1', 'Subg2']]
    mut_simls = {
        subg: siml_fxs[siml_metric](
            wt_vals[subg], use_preds.loc[use_preds.Phn == subg, subg],
            use_preds.loc[use_preds.Phn == oth_subg, subg]
            )
        for subg, oth_subg in permt(['Subg1', 'Subg2'])
        }

    oth_simls = {
        subg: siml_fxs[siml_metric](
            wt_vals[subg], use_preds.loc[use_preds.Phn == subg, subg],
            use_preds.loc[use_preds.Phn == 'Other', subg]
            )
        for subg in ['Subg1', 'Subg2']
        }

    crnr_ax.text(0.95, 0.34, "(Siml1: {:.3f})".format(mut_simls['Subg1']),
                 size=11, ha='right', transform=crnr_ax.transAxes,
                 clip_on=False)
    crnr_ax.text(0.95, 0.09, "(Siml1: {:.3f})".format(oth_simls['Subg1']),
                 size=11, ha='right', transform=crnr_ax.transAxes,
                 clip_on=False)

    crnr_ax.text(0.25, 0.55, "(Siml2: {:.3f})".format(mut_simls['Subg2']),
                 size=11, rotation=31, ha='right',
                 transform=crnr_ax.transAxes, clip_on=False)
    crnr_ax.text(0.5, 0.55, "(Siml2: {:.3f})".format(oth_simls['Subg2']),
                 size=11, rotation=31, ha='right',
                 transform=crnr_ax.transAxes, clip_on=False)

    crnr_ax.axis('off')

    sctr_ax.set_xlim(xlims)
    sctr_ax.set_ylim(ylims)
    mcomb1_ax.set_xlim(xlims)
    mcomb2_ax.set_ylim(ylims)

    file_lbl = '__'.join(['__'.join([mtype.get_filelabel()[:30]
                                     for mtype in mcomb.mtypes])
                          for mcomb in [plt_mcomb1, plt_mcomb2]])

    fig.tight_layout(w_pad=-1.3, h_pad=-1.3)
    plt.savefig(os.path.join(
        plot_dir, data_tag, "{}_{}-ortho-scores_{}__{}.svg".format(
            args.ex_lbl, siml_metric, args.classif, file_lbl)
        ), bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_point',
        description="Compares point mutation subgroupings with a cohort."
        )

    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('ex_lbl', help="a classification mode",
                        choices={'Iso', 'IsoShal'})

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.8)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    # parse command line arguments, find completed runs for this classifier,
    # create directory where plots will be saved
    args = parser.parse_args()
    out_datas = tuple(Path(base_dir).glob(
        os.path.join("*", "out-aucs__*__*__{}.p.gz".format(args.classif))))
    os.makedirs(plot_dir, exist_ok=True)

    # get info on completed runs and group them by tumor cohort, filtering out
    # cohorts where the `base` set of mutation levels has not been tested
    out_list = pd.DataFrame(
        [{'Source': '__'.join(out_data.parts[-2].split('__')[:-1]),
          'Cohort': out_data.parts[-2].split('__')[-1],
          'Levels': '__'.join(out_data.parts[-1].split('__')[1:-2]),
          'File': out_data}
         for out_data in out_datas]
        ).groupby('Cohort').filter(
            lambda outs: 'Consequence__Exon' in set(outs.Levels))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_list = out_list[out_list.Cohort.isin(train_cohorts)]
    use_iter = out_list.groupby(['Source', 'Cohort', 'Levels'])['File']

    out_dirs = {(src, coh): Path(base_dir, '__'.join([src, coh]))
                for src, coh, _ in use_iter.groups}
    out_tags = {fl: '__'.join(fl.parts[-1].split('__')[1:])
                for fl in out_list.File}
    pred_tag = "out-pred_{}".format(args.ex_lbl)

    phn_dicts = {(src, coh): dict() for src, coh, _ in use_iter.groups}
    cdata_dict = {(src, coh): None for src, coh, _ in use_iter.groups}

    auc_lists = {(src, coh): pd.Series(dtype='float')
                 for src, coh, _ in use_iter.groups}
    pred_dfs = {(src, coh): pd.DataFrame() for src, coh, _ in use_iter.groups}

    for (src, coh, lvls), out_files in use_iter:
        out_aucs = list()
        out_preds = list()

        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["out-pheno",
                                             out_tags[out_file]])),
                             'r') as f:
                phn_dicts[src, coh].update(pickle.load(f))

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["out-aucs",
                                             out_tags[out_file]])),
                             'r') as f:
                out_aucs += [pickle.load(f)[args.ex_lbl]['mean']]

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join([pred_tag, out_tags[out_file]])),
                             'r') as f:
                pred_vals = pickle.load(f)

            out_preds += [pred_vals.applymap(np.mean)]

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["cohort-data",
                                             out_tags[out_file]])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[src, coh] is None:
                cdata_dict[src, coh] = new_cdata
            else:
                cdata_dict[src, coh].merge(new_cdata)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals.index) for auc_vals in out_aucs]] * 2))
        super_comp = np.apply_along_axis(all, 1, mtypes_comp)

        # if there is not a subgrouping set that contains all the others,
        # concatenate the output of all sets...
        if not super_comp.any():
            auc_lists[src, coh] = auc_lists[src, coh].append(
                pd.concat(out_aucs, sort=False))
            pred_dfs[src, coh] = pd.concat(
                [pred_dfs[src, coh], *out_preds], sort=False)

        # ...otherwise, use the "superset"
        else:
            super_indx = super_comp.argmax()

            auc_lists[src, coh] = auc_lists[src, coh].append(
                out_aucs[super_indx])
            pred_dfs[src, coh] = pd.concat(
                [pred_dfs[src, coh], out_preds[super_indx]], sort=False)

    # filter out duplicate subgroupings due to overlapping search criteria
    for src, coh, _ in use_iter.groups:
        auc_lists[src, coh].sort_index(inplace=True)
        pred_dfs[src, coh].sort_index(inplace=True)
        assert (auc_lists[src, coh].index == pred_dfs[src, coh].index).all()

        auc_lists[src, coh] = auc_lists[src, coh].loc[
            ~auc_lists[src, coh].index.duplicated()]
        pred_dfs[src, coh] = pred_dfs[src, coh].loc[
            ~pred_dfs[src, coh].index.duplicated()]

    # create the plots
    for siml_metric in args.siml_metrics:
        if args.auc_cutoff < 1:
            annt_types = plot_divergent_types(pred_dfs, phn_dicts, auc_lists,
                                              cdata_dict, args, siml_metric)

            annt_pairs = plot_divergent_pairs(pred_dfs, phn_dicts, auc_lists,
                                              cdata_dict, args, siml_metric)

            for (src, coh), pair_list in annt_pairs.items():
                if pair_list:
                    os.makedirs(os.path.join(plot_dir, '__'.join([src, coh])),
                                exist_ok=True)

                for mcomb1, mcomb2 in pair_list:
                    plot_orthogonal_scores(
                        mcomb1, mcomb2, pred_dfs[src, coh],
                        auc_lists[src, coh], phn_dicts[src, coh],
                        cdata_dict[src, coh], '__'.join([src, coh]),
                        args, siml_metric
                        )


if __name__ == '__main__':
    main()

