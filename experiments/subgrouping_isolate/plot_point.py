
from ..utilities.mutations import (
    pnt_mtype, shal_mtype, gains_mtype, dels_mtype, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import (load_cohorts_data, siml_fxs, remove_pheno_dups,
                    get_mut_ex, get_mcomb_lbl, choose_subtype_colour)
from ..utilities.labels import get_cohort_label
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.colour_maps import simil_cmap, variant_clrs
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
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
    fig, ax = plt.subplots(figsize=(12, 8))

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

        # for each subgrouping, find the subset of point mutations that
        # defines it, the gene it's associated with, and its task predictions
        for mcomb in use_combs:
            use_mtype = tuple(mcomb.mtypes)[0]
            cur_gene = get_label(mcomb)
            use_preds = pred_dfs[src, coh].loc[mcomb, train_samps]

            # get the samples that carry any point mutation of this gene
            if (src, coh, cur_gene) not in gn_dict:
                gn_dict[src, coh, cur_gene] = np.array(
                    cdata_dict[src, coh].train_pheno(
                        MuType({('Gene', cur_gene): pnt_mtype}))
                    )

            # get samples that have any point mutation of this gene that is
            # not the subgrouping's mutation and no other mutation of the gene
            rst_dict[src, coh, mcomb] = np.array(
                cdata_dict[src, coh].train_pheno(
                    ExMcomb(mcomb.all_mtype,
                            (mcomb.all_mtype & pnt_mtype) - use_mtype)
                    )
                )

            assert not (pheno_dicts[src, coh][mcomb]
                        & rst_dict[src, coh, mcomb]).any()

            # find the similarity of the remaining exclusively defined point
            # mutations to samples carrying only this subgrouping's mutation
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
        coh_lbl = get_cohort_label(coh).replace("TCGA-", '')

        for cur_gene, divg_vals in divg_df.groupby(get_label):
            if cur_gene not in clr_dict:
                clr_dict[cur_gene] = choose_label_colour(cur_gene)

            plt_size = np.mean(gn_dict[src, coh, cur_gene]) ** 0.5
            divg_stat = (divg_vals.Divg < 0.5).any()
            divg_tupl = divg_vals.AUC[0], divg_vals.Divg[0]
            line_dict[divg_tupl] = cur_gene

            for i, (mcomb, (divg_val, auc_val)) in enumerate(
                    divg_vals.iterrows()):
                if not (divg_vals[:i].AUC > auc_val).any():
                    divg_mcombs |= {mcomb}
                    plot_dict[auc_val, divg_val] = [plt_size, ('', '')]

            if divg_stat:
                plot_dict[divg_tupl] = [plt_size,
                                        ("{} in {}".format(cur_gene, coh_lbl),
                                         get_mcomb_lbl(mcomb))]

            else:
                plot_dict[divg_tupl] = [
                    plt_size, ("{} in {}".format(cur_gene, coh_lbl), '')]

        divg_dfs[src, coh] = divg_df.loc[divg_mcombs]

    # TODO: make this scale better for smaller number of points?
    size_mult = sum(df.shape[0] for df in divg_dfs.values()) ** -0.23
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
            plt_size = 0.83 * size_mult * plot_dict[auc_val, divg_val][0]
            plot_dict[auc_val, divg_val][0] *= 0.23 * size_mult

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
    ylims = [ymin - yrng / 5.3, ymax + yrng / 5.3]

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

    for siml_val in [0, 1]:
        ax.plot(xlims, [siml_val, siml_val],
                color='black', linewidth=0.83, linestyle=':', alpha=0.67)

    ax.set_xlabel("Isolated Classification Accuracy", size=21, weight='bold')
    ax.set_ylabel("Inferred Similarity to Same Gene's"
                  "\nRemaining Point Mutations", size=21, weight='bold')

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax, plt_lims=[xlims, ylims],
                                       font_size=11, line_dict=line_dict,
                                       linewidth=0.91, alpha=0.37)

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
    fig, ax = plt.subplots(figsize=(12, 8))

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
        coh_lbl = get_cohort_label(coh).replace("TCGA-", '')

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
            if gene not in clr_dict:
                clr_dict[gene] = choose_label_colour(gene)

            lbl_pair = pair_vals.index[0]
            annt_dict[src, coh] |= set(pair_vals.index[:5])
            pair_lbl = '\nvs.\n'.join([get_mcomb_lbl(mcomb)
                                       for mcomb in lbl_pair])

            plt_tupl = auc_list[list(lbl_pair)].min(), divg_list[lbl_pair]
            line_dict[plt_tupl] = src, coh, gene

            plot_dict[plt_tupl] = [size_dict[(src, coh, *lbl_pair)],
                                   ("{} in {}".format(gene, coh_lbl),
                                    pair_lbl)]

    size_mult = sum(len(divg_list)
                    for divg_list in divg_lists.values()) ** -0.23

    for k in line_dict:
        line_dict[k] = {'c': clr_dict[line_dict[k][-1]]}
    for k in size_dict:
        size_dict[k] *= 6037 * size_mult

    for (src, coh), divg_list in divg_lists.items():
        for (mcomb1, mcomb2), divg_val in divg_list.iteritems():
            cur_gene = get_label(mcomb1)
            plt_tupl = auc_lists[src, coh][[mcomb1, mcomb2]].min(), divg_val
            plot_dict[plt_tupl][0] *= 0.23 * size_mult

            ax.scatter(*plt_tupl, s=size_dict[src, coh, mcomb1, mcomb2],
                       c=[clr_dict[cur_gene]], alpha=0.31, edgecolor='none')

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]

    ymin = min(divg_list.min() for divg_list in divg_lists.values())
    ymax = max(divg_list.max() for divg_list in divg_lists.values())
    yrng = ymax - ymin
    ylims = [ymin - yrng / 4.1, ymax + yrng / 4.1]

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

    for siml_val in [0, 1]:
        ax.plot(xlims, [siml_val, siml_val],
                color='black', linewidth=0.83, linestyle=':', alpha=0.67)

    ax.set_xlabel("Minimum Classification Accuracy\nAcross Subgrouping Pair",
                  size=21, weight='bold')
    ax.set_ylabel("Maximum of Inferred Similarities"
                  "\nBetween Subgrouping Pair", size=21, weight='bold')

    for k in np.linspace(args.auc_cutoff, 0.99, 100):
        for j in [0, 1]:
            if (k, j) not in plot_dict:
                plot_dict[k, j] = [1 / 503, ('', '')]

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[xlims, ylims],
                                       font_size=11, line_dict=line_dict,
                                       linewidth=0.91, alpha=0.37)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    plt.savefig(os.path.join(plot_dir,
                             "{}_{}-divergent-pairs_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()
    return annt_dict


def plot_orthogonal_scores(pred_vals, auc_vals, pheno_dict, cdata,
                           data_tag, args, siml_metric):
    fig, ((mcomb2_ax, sctr_ax), (crnr_ax, mcomb1_ax)) = plt.subplots(
        figsize=(13, 12), nrows=2, ncols=2,
        gridspec_kw=dict(height_ratios=[3.5, 1], width_ratios=[1, 3.5])
        )

    plt_mcomb1, plt_mcomb2 = pred_vals.index
    assert len(plt_mcomb1.mtypes) == len(plt_mcomb2.mtypes) == 1
    mcomb1_vals, mcomb2_vals = pred_vals[cdata.get_train_samples()].values

    use_gene = {get_label(mtype) for mcomb in [plt_mcomb1, plt_mcomb2]
                for mtype in mcomb.mtypes}
    assert len(use_gene) == 1
    use_gene = tuple(use_gene)[0]

    base_mtree = tuple(cdata.mtrees.values())[0]
    all_mtype = MuType({('Gene', use_gene): base_mtree[use_gene].allkey()})
    if args.ex_lbl == 'IsoShal':
        all_mtype -= MuType({('Gene', use_gene): shal_mtype})

    x_min, x_max = np.percentile(mcomb1_vals, q=[0, 100])
    y_min, y_max = np.percentile(mcomb2_vals, q=[0, 100])
    x_rng, y_rng = x_max - x_min, y_max - y_min
    xlims = x_min - x_rng / 31, x_max + x_rng / 11
    ylims = y_min - y_rng / 31, y_max + y_rng / 11

    use_phns = {
        'WT': ~np.array(cdata.train_pheno(all_mtype)),
        'Mut1': pheno_dict[plt_mcomb1], 'Mut2': pheno_dict[plt_mcomb2],
        'Other': (
            np.array(cdata.train_pheno(
                ExMcomb(all_mtype, MuType({('Gene', use_gene): pnt_mtype}))))
            & ~pheno_dict[plt_mcomb1] & ~pheno_dict[plt_mcomb2]
            ),
        'Gain': np.array(cdata.train_pheno(
            ExMcomb(all_mtype, MuType({('Gene', use_gene): gains_mtype})))),
        'Del': np.array(cdata.train_pheno(
            ExMcomb(all_mtype, MuType({('Gene', use_gene): dels_mtype}))))
        }

    use_fclrs = {'WT': variant_clrs['WT'],
                 'Mut1': '#080097', 'Mut2': '#00847F',
                 'Other': 'none', 'Gain': 'none', 'Del': 'none'}

    use_eclrs = {'WT': 'none', 'Mut1': 'none', 'Mut2': 'none',
                 'Other': choose_subtype_colour(pnt_mtype),
                 'Gain': choose_subtype_colour(gains_mtype),
                 'Del': choose_subtype_colour(dels_mtype)}

    use_sizes = {'WT': 5, 'Mut1': 7, 'Mut2': 7,
                 'Other': 5, 'Gain': 5, 'Del': 5}
    use_alphas = {'WT': 0.21, 'Mut1': 0.31, 'Mut2': 0.31,
                  'Other': 0.35, 'Gain': 0.35, 'Del': 0.35}

    for lbl, phn in use_phns.items():
        sctr_ax.plot(mcomb1_vals[phn], mcomb2_vals[phn],
                     marker='o', linewidth=0,
                     markersize=use_sizes[lbl], alpha=use_alphas[lbl],
                     mfc=use_fclrs[lbl], mec=use_eclrs[lbl], mew=1.9)

    mcomb_lbls = [get_mcomb_lbl(mcomb) for mcomb in [plt_mcomb1, plt_mcomb2]]
    subg_lbls = ["only {} mutation\nis {}".format(use_gene, mcomb_lbl)
                 for mcomb_lbl in mcomb_lbls]

    sctr_ax.text(0.98, 0.03, subg_lbls[0], size=17, c=use_fclrs['Mut1'],
                 ha='right', va='bottom', transform=sctr_ax.transAxes)
    sctr_ax.text(0.03, 0.98, subg_lbls[1], size=17, c=use_fclrs['Mut2'],
                 ha='left', va='top', transform=sctr_ax.transAxes)

    sctr_ax.text(0.97, 0.98, get_cohort_label(data_tag.split('__')[1]),
                 size=21, style='italic', ha='right', va='top',
                 transform=sctr_ax.transAxes)

    sctr_ax.set_xticklabels([])
    sctr_ax.set_yticklabels([])

    for ax in sctr_ax, mcomb1_ax, mcomb2_ax:
        ax.grid(alpha=0.47, linewidth=0.9)

    use_preds = pd.DataFrame({
        'Mut1': mcomb1_vals, 'Mut2': mcomb2_vals, 'Phn': 'WT'})
    for lbl in ['Mut1', 'Mut2', 'Other', 'Gain', 'Del']:
        use_preds.loc[use_phns[lbl], 'Phn'] = lbl

    sns.violinplot(data=use_preds, y='Phn', x='Mut1', ax=mcomb1_ax,
                   order=['WT', 'Mut1', 'Mut2', 'Other', 'Gain', 'Del'],
                   palette=use_fclrs, orient='h', linewidth=0, cut=0)

    sns.violinplot(data=use_preds, x='Phn', y='Mut2', ax=mcomb2_ax,
                   order=['WT', 'Mut2', 'Mut1', 'Other', 'Gain', 'Del'],
                   palette=use_fclrs, orient='v', linewidth=0, cut=0)

    for mcomb_ax in mcomb1_ax, mcomb2_ax:
        for i in range(3):
            mcomb_ax.get_children()[i * 2].set_alpha(0.61)
            mcomb_ax.get_children()[i * 2].set_linewidth(0)

        i = 0
        for lbl in ['Other', 'Gain', 'Del']:
            if use_phns[lbl].sum() > 1:
                i += 1

                mcomb_ax.get_children()[4 + i * 2].set_edgecolor(
                    use_eclrs[lbl])
                mcomb_ax.get_children()[4 + i * 2].set_facecolor('white')
                mcomb_ax.get_children()[4 + i * 2].set_linewidth(2.9)
                mcomb_ax.get_children()[4 + i * 2].set_alpha(0.71)

    mcomb1_ax.set_xlabel("Subgrouping Isolation Task 1\nPredicted Scores",
                         size=23, weight='semibold')
    mcomb2_ax.set_ylabel("Subgrouping Isolation Task 2\nPredicted Scores",
                         size=23, weight='semibold')

    mcomb2_ax.xaxis.label.set_visible(False)
    mcomb1_ax.yaxis.label.set_visible(False)

    for i, lbl in enumerate(['WT', 'Mut1', 'Mut2', 'Other', 'Gain', 'Del']):
        mcomb1_ax.text(1, 0.9 - i / 6, "n={}".format(use_phns[lbl].sum()),
                       size=13, ha='left', transform=mcomb1_ax.transAxes,
                       clip_on=False)

    for i, lbl in enumerate(['WT', 'Mut2', 'Mut1', 'Other', 'Gain', 'Del']):
        mcomb2_ax.text(0.04 + i / 6, 1, "n={}".format(use_phns[lbl].sum()),
                       size=13, rotation=31, ha='left',
                       transform=mcomb2_ax.transAxes, clip_on=False)

    wt_vals = use_preds.loc[use_preds.Phn == 'WT', ['Mut1', 'Mut2']]
    siml_vals = {
        'mut': {
            subg: format(siml_fxs[siml_metric](
                wt_vals[subg], use_preds.loc[use_preds.Phn == subg, subg],
                use_preds.loc[use_preds.Phn == oth_subg, subg]
            ), '.3f')
            for subg, oth_subg in permt(['Mut1', 'Mut2'])
            }
        }

    for lbl in ['Other', 'Gain', 'Del']:
        if use_phns[lbl].sum() >= 10:
            siml_vals[lbl] = {
                subg: format(siml_fxs[siml_metric](
                    wt_vals[subg], use_preds.loc[use_preds.Phn == subg, subg],
                    use_preds.loc[use_preds.Phn == lbl, subg]
                ), '.3f')
                for subg in ['Mut1', 'Mut2']
            }

        else:
            siml_vals[lbl] = {'Mut1': "    n/a", 'Mut2': "    n/a"}

    mcomb1_ax.set_yticklabels(['WT', 'Subg1', 'Subg2',
                               'Other {}\nPoint Muts'.format(use_gene),
                               'Gains', 'Dels'], size=12)

    mcomb2_ax.set_xticklabels(['WT', 'Subg2', 'Subg1',
                               'Other', 'Gains', 'Dels'],
                              rotation=31, size=12, ha='right')

    mcomb1_ax.text(-0.17, 0.72, "(AUC1: {:.3f})".format(auc_vals[plt_mcomb1]),
                   size=11, ha='right', transform=mcomb1_ax.transAxes,
                   clip_on=False)
    mcomb2_ax.text(-0.04, -0.06,
                   "(AUC2: {:.3f})".format(auc_vals[plt_mcomb2]),
                   size=11, rotation=31, ha='right', va='top',
                   transform=mcomb2_ax.transAxes, clip_on=False)

    for i, lbl in enumerate(['mut', 'Other', 'Gain', 'Del']):
        mcomb1_ax.text(-0.17, 0.61 - i / 6,
                       "(Siml1: {})".format(siml_vals[lbl]['Mut1']),
                       size=11, ha='right', va='top',
                       transform=mcomb1_ax.transAxes, clip_on=False)

        mcomb2_ax.text(0.14 + i / 6, -0.06,
                       "(Siml2: {})".format(siml_vals[lbl]['Mut2']),
                       size=11, rotation=31, ha='right', va='top',
                       transform=mcomb2_ax.transAxes, clip_on=False)

    crnr_ax.axis('off')
    sctr_ax.set_xlim(xlims)
    sctr_ax.set_ylim(ylims)
    mcomb1_ax.set_xlim(xlims)
    mcomb2_ax.set_ylim(ylims)

    file_lbl = '__'.join(['__'.join([mtype.get_filelabel()[:30]
                                     for mtype in mcomb.mtypes])
                          for mcomb in [plt_mcomb1, plt_mcomb2]])

    fig.tight_layout(w_pad=-10, h_pad=-2)
    plt.savefig(os.path.join(
        plot_dir, data_tag, "{}_{}-ortho-scores_{}__{}.svg".format(
            args.ex_lbl, siml_metric, args.classif, file_lbl)
        ), bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_point',
        description="Compares point mutation subgroupings across all cohorts."
        )

    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('ex_lbl', help="a classification mode",
                        choices={'Iso', 'IsoShal'})

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.8)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data_cache')
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
    pred_dfs, phn_dicts, auc_lists, cdata_dict = load_cohorts_data(
        out_list, args.ex_lbl, args.data_cache)

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
                        pred_dfs[src, coh].loc[[mcomb1, mcomb2]],
                        auc_lists[src, coh], phn_dicts[src, coh],
                        cdata_dict[src, coh], '__'.join([src, coh]),
                        args, siml_metric
                        )


if __name__ == '__main__':
    main()

