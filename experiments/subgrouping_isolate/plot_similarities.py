
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, shal_mtype, deep_mtype,
    dup_mtype, gains_mtype, loss_mtype, dels_mtype, ExMcomb
    )
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir
from .utils import remove_pheno_dups
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.misc import choose_label_colour
from ..utilities.colour_maps import simil_cmap
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

import warnings
from ..utilities.misc import warning_on_one_line
warnings.formatwarning = warning_on_one_line

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'similarities')

SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}
cna_mtypes = {'Gain': gains_mtype, 'Loss': dels_mtype}


def plot_subpoint_divergences(pred_df, pheno_dict, auc_vals, cdata, args,
                              siml_metric, add_lgnd=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    use_combs = remove_pheno_dups({
        mcomb for mcomb in auc_vals.index
        if (isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
            and not (mcomb.all_mtype & shal_mtype).is_empty())
        }, pheno_dict)

    pnt_aucs = auc_vals[[
        mcomb for mcomb in use_combs
        if (auc_vals[mcomb] > 0.6
            and pnt_mtype != tuple(tuple(
                mcomb.mtypes)[0].subtype_iter())[0][1]
            and pnt_mtype.is_supertype(
                tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1]))
        ]]

    plt_gby = pnt_aucs.groupby(lambda mtype: tuple(mtype.label_iter())[0])
    clr_dict = {gene: choose_label_colour(gene)
                for gene in plt_gby.groups.keys()}
    lbl_pos = {gene: list() for gene in plt_gby.groups.keys()}

    train_samps = cdata.get_train_samples()
    plot_dict = dict()
    font_dict = dict()
    line_dict = dict()
    ylim = 1.03

    # TODO: differentiate between genes without CNAs and those
    #  with too much overlap between CNAs and point mutations?
    auc_list: pd.Series
    for cur_gene, auc_list in plt_gby:
        use_preds = pred_df.loc[
            set(auc_list.index), train_samps].applymap(np.mean)

        use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]
        all_mtype = MuType({('Gene', cur_gene): use_mtree.allkey()})
        all_phn = np.array(cdata.train_pheno(all_mtype))

        gn_mtype = MuType({('Gene', cur_gene): {
            ('Scale', 'Point'): use_mtree['Point'].allkey()}})
        gn_phn = np.array(cdata.train_pheno(gn_mtype))

        for mcomb, auc_val in auc_list.iteritems():
            use_mtype = tuple(mcomb.mtypes)[0]
            rst_phn = gn_phn & ~np.array(cdata.train_pheno(use_mtype))

            siml_val = SIML_FXS[siml_metric](
                use_preds.loc[mcomb][~all_phn],
                use_preds.loc[mcomb][pheno_dict[mcomb]],
                use_preds.loc[mcomb][rst_phn]
                )

            base_size = np.mean(pheno_dict[mcomb])
            ylim = max(ylim, abs(siml_val) + 0.13)

            pos_tupl = auc_val, siml_val
            lbl_pos[cur_gene] += [pos_tupl]
            plot_dict[pos_tupl] = [1.41 * base_size, ('', '')]

            ax.scatter(auc_val, siml_val, c=[clr_dict[cur_gene]],
                       s=601 * base_size, alpha=0.29, edgecolor='none')

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    ax.grid(alpha=0.43, linewidth=0.61)
    ax.tick_params(labelsize=11)

    for siml_val in [-1, 0, 1, 2]:
        if -ylim <= siml_val <= ylim:
            for k in np.linspace(0.63, 0.97, 200):
                plot_dict[k, siml_val] = [0.1, ('', '')]

    plt_counts = pd.Series({gn: len(pos) for gn, pos in lbl_pos.items()})
    plt_counts = plt_counts[plt_counts.values > 1].sort_values()

    for gene in plt_counts.index:
        pos_med = tuple(pd.DataFrame(lbl_pos[gene]).mean().tolist())

        font_dict[pos_med] = dict(c=clr_dict[gene], weight='bold')
        line_dict[pos_med] = dict(c=clr_dict[gene])
        plot_dict[pos_med] = [0, (gene, '')]

    _ = place_scatter_labels(
        plot_dict, ax, plt_lims=[[0.59, 1], [-ylim, ylim]],
        plc_lims=[[0.63, 0.97], [-ylim * 0.83, ylim * 0.83]],
        plt_type='scatter', font_size=13, font_dict=font_dict,
        line_dict=line_dict, linewidth=2.3, alpha=0.23
        )

    ax.plot([1, 1], [-ylim, ylim],
            color='black', linewidth=1.1, alpha=0.89)
    ax.plot([0.6, 1], [0, 0],
            color='black', linewidth=1.7, linestyle=':', alpha=0.61)

    for siml_val in [-1, 1, 2]:
        ax.plot([0.6, 1], [siml_val] * 2,
                color=simil_cmap(clr_norm(siml_val)),
                linewidth=3.7, linestyle=':', alpha=0.41)

    ax.set_xlabel("Accuracy of Isolated Subgrouping Classifier",
                  size=21, weight='bold')
    ax.set_ylabel("Remaining Point Mutations'\nSimilarity to Subgrouping",
                  size=21, weight='bold')

    ax.set_xlim(0.59, 1.005)
    ax.set_ylim(-ylim, ylim)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}-subPoint-divergences_{}.svg".format(
                         siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_subcopy_adjacencies(pred_df, pheno_dict, auc_vals, cdata, args,
                             cna_lbl, siml_metric, add_lgnd=False):
    fig, axarr = plt.subplots(figsize=(10, 9), nrows=2, ncols=1)
    cna_mtype = cna_mtypes[cna_lbl]

    use_combs = remove_pheno_dups({
        mcomb for mcomb in auc_vals.index
        if (isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
            and not (mcomb.all_mtype & shal_mtype).is_empty())
        }, pheno_dict)

    pnt_aucs = auc_vals[[
        mcomb for mcomb in use_combs
        if (auc_vals[mcomb] > 0.65
            and pnt_mtype.is_supertype(
                tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1]))
        ]]

    plt_gby = pnt_aucs.groupby(lambda mtype: tuple(mtype.label_iter())[0])
    clr_dict = {gene: choose_label_colour(gene)
                for gene in plt_gby.groups.keys()}
    lbl_pos = [{gene: list() for gene in plt_gby.groups.keys()}
               for _ in range(2)]

    train_samps = cdata.get_train_samples()
    plot_dicts = [dict(), dict()]
    font_dicts = [dict(), dict()]
    line_dicts = [dict(), dict()]
    ylim = 1.03

    # TODO: differentiate between genes without CNAs and those
    #  with too much overlap between CNAs and point mutations?
    auc_list: pd.Series
    for cur_gene, auc_list in plt_gby:
        gene_cna = MuType({('Gene', cur_gene): cna_mtype})
        plt_combs = {mcomb for mcomb in use_combs
                     if tuple(mcomb.mtypes)[0] == gene_cna}

        if len(plt_combs) > 1:
            raise ValueError("Too many exclusive {} CNAs associated with "
                             "`{}`!".format(cna_lbl, cur_gene))

        if len(plt_combs) == 1:
            plt_comb = tuple(plt_combs)[0]
            use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]

            all_mtype = MuType({('Gene', cur_gene): use_mtree.allkey()})
            all_phn = np.array(cdata.train_pheno(all_mtype))
            use_preds = pred_df.loc[set(auc_list.index) | {plt_comb},
                                    train_samps].applymap(np.mean)

            for mcomb, auc_val in auc_list.iteritems():
                copy_simls = [
                    SIML_FXS[siml_metric](
                        use_preds.loc[mcomb][~all_phn],
                        use_preds.loc[mcomb][pheno_dict[mcomb]],
                        use_preds.loc[mcomb][pheno_dict[plt_comb]]
                        )
                    ]

                if auc_vals[plt_comb] > 0.6:
                    copy_simls += [
                        SIML_FXS[siml_metric](
                            use_preds.loc[plt_comb][~all_phn],
                            use_preds.loc[plt_comb][pheno_dict[plt_comb]],
                            use_preds.loc[plt_comb][pheno_dict[mcomb]]
                            )
                        ]

                base_size = np.mean(pheno_dict[mcomb])
                ylim = max(ylim, np.max(np.abs(np.array(copy_simls))) + 0.13)

                for i, (ax, siml_val) in enumerate(
                        zip(axarr[:len(copy_simls)], copy_simls)):
                    pos_tupl = auc_val, siml_val
                    lbl_pos[i][cur_gene] += [pos_tupl]
                    plot_dicts[i][pos_tupl] = [1.41 * base_size, ('', '')]

                    ax.scatter(auc_val, siml_val,
                               c=[clr_dict[cur_gene]], s=601 * base_size,
                               alpha=0.29, edgecolor='none')

        else:
            for i in range(2):
                del(lbl_pos[i][cur_gene])

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    for i, ax in enumerate(axarr):
        ax.grid(alpha=0.43, linewidth=0.61)
        ax.tick_params(labelsize=11)

        for siml_val in [-1, 0, 1, 2]:
            if -ylim <= siml_val <= ylim:
                for k in np.linspace(0.68, 0.97, 200):
                    plot_dicts[i][k, siml_val] = [0.1, ('', '')]

        for gene, pos_list in lbl_pos[i].items():
            if pos_list:
                pos_med = tuple(pd.DataFrame(pos_list).mean().tolist())
                font_dicts[i][pos_med] = dict(c=clr_dict[gene], weight='bold')
                line_dicts[i][pos_med] = dict(c=clr_dict[gene])
                plot_dicts[i][pos_med] = [0, (gene, '')]

        _ = place_scatter_labels(
            plot_dicts[i], ax, plt_lims=[[0.64, 1], [-ylim, ylim]],
            plc_lims=[[0.68, 0.97], [-ylim * 0.83, ylim * 0.83]],
            plt_type='scatter', font_size=13, font_dict=font_dicts[i],
            line_dict=line_dicts[i], linewidth=2.3, alpha=0.23
            )

        ax.plot([1, 1], [-ylim, ylim],
                color='black', linewidth=1.1, alpha=0.89)
        ax.plot([0.65, 1], [0, 0],
                color='black', linewidth=1.7, linestyle=':', alpha=0.61)

        for siml_val in [-1, 1, 2]:
            ax.plot([0.65, 1], [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=3.7, linestyle=':', alpha=0.41)

    axarr[0].set_ylabel("{} Similarity\nto Subgrouping".format(cna_lbl),
                        size=21, weight='bold')
    axarr[1].set_ylabel(
        "Subgrouping Similarity\nto All {} Alterations".format(cna_lbl),
        size=21, weight='bold'
        )

    axarr[1].set_xlabel("Accuracy of Isolated Subgrouping Classifier",
                        size=21, weight='bold')

    for ax in axarr:
        ax.set_xlim(0.64, 1.005)
        ax.set_ylim(-ylim, ylim)

    plt.tight_layout(pad=0, h_pad=1.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}-sub{}-adjacencies_{}.svg".format(
                         siml_metric, cna_lbl, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_copy_interaction(pred_df, pheno_dict, auc_vals,
                          cdata, args, siml_metric, add_lgnd=False):
    fig, (gain_ax, loss_ax) = plt.subplots(figsize=(10, 9), nrows=2, ncols=1)

    use_mcombs = {mut for mut in auc_vals.index
                  if (isinstance(mut, ExMcomb) and len(mut.mtypes) == 1
                      and not (mut.all_mtype & shal_mtype).is_empty())}

    copy_mcombs = {
        cna_lbl: {
            mcomb for mcomb in use_mcombs
            if (auc_vals[mcomb] > 0.6
                and cna_type.is_supertype(
                    tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1]))
            }
        for cna_lbl, cna_type in cna_mtypes.items()
        }

    train_samps = cdata.get_train_samples()
    plot_dicts = {'Gain': dict(), 'Loss': dict()}
    line_dicts = {'Gain': dict(), 'Loss': dict()}
    plt_lims = [0.1, 0.9]

    for cna_lbl, ax in zip(['Gain', 'Loss'], [gain_ax, loss_ax]):
        for copy_comb in copy_mcombs[cna_lbl]:
            cur_gene, copy_subt = tuple(tuple(
                copy_comb.mtypes)[0].subtype_iter())[0]

            pnt_combs = {mcomb for mcomb in use_mcombs
                         if (tuple(mcomb.mtypes)[0]
                             == MuType({('Gene', cur_gene): pnt_mtype}))}

            assert len(pnt_combs) <= 1, (
                "Too many exclusive gene-wide mutations found "
                "for `{}`!".format(cur_gene)
                )

            if len(pnt_combs) == 1:
                pnt_mcomb = tuple(pnt_combs)[0]
                use_clr = choose_label_colour(cur_gene)

                use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]
                all_mtype = MuType({('Gene', cur_gene): use_mtree.allkey()})
                all_phn = np.array(cdata.train_pheno(all_mtype))
                use_preds = pred_df.loc[copy_comb, train_samps].apply(np.mean)

                if siml_metric == 'mean':
                    copy_siml = calculate_mean_siml(
                        use_preds[~all_phn], use_preds[pheno_dict[copy_comb]],
                        use_preds[pheno_dict[pnt_mcomb]]
                        )

                elif siml_metric == 'ks':
                    copy_siml = calculate_ks_siml(
                        use_preds[~all_phn], use_preds[pheno_dict[copy_comb]],
                        use_preds[pheno_dict[pnt_mcomb]]
                        )

                if (copy_subt & deep_mtype).is_empty():
                    subt_lbl = get_fancy_label(copy_subt)
                else:
                    subt_lbl = ''

                dyad_size = np.mean(pheno_dict[copy_comb]
                                    | pheno_dict[pnt_mcomb])

                plot_dicts[cna_lbl][auc_vals[copy_comb], copy_siml] = (
                    dyad_size ** 0.91, (cur_gene, subt_lbl))
                line_dicts[cna_lbl][auc_vals[copy_comb], copy_siml] = dict(
                    c=use_clr)

                ax.scatter(auc_vals[copy_comb], copy_siml, s=dyad_size * 2039,
                           c=[use_clr], alpha=0.31, edgecolor='none')

                plt_lims[0] = min(plt_lims[0], copy_siml - 0.11)
                plt_lims[1] = max(plt_lims[1], copy_siml + 0.11)

    gain_ax.set_ylabel("Similarity to\nAll Gain Alterations",
                       size=23, weight='bold')
    loss_ax.set_ylabel("Similarity to\nAll Loss Alterations",
                       size=23, weight='bold')
    loss_ax.set_xlabel("Accuracy of Isolated Classifier",
                       size=23, weight='bold')

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    y_rng = plt_lims[1] - plt_lims[0]

    for cna_lbl, ax in zip(['Gain', 'Loss'], [gain_ax, loss_ax]):
        ax.grid(alpha=0.43, linewidth=0.7)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.1, alpha=0.89)
        ax.plot([0.6, 1], [0, 0],
                color='black', linewidth=1.3, linestyle=':', alpha=0.53)

        for siml_val in [-1, 1, 2]:
            ax.plot([0.6, 1], [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.53)

        ax.set_xlim(0.59, 1.005)
        ax.set_ylim(*plt_lims)

        lbl_pos = place_scatter_labels(
            plot_dicts[cna_lbl], ax,
            plt_type='scatter', font_size=11, seed=args.seed,
            line_dict=line_dicts[cna_lbl], linewidth=0.71, alpha=0.61
            )

    plt.tight_layout(pad=0, h_pad=1.9)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}-copy-interaction_{}.svg".format(
                         siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_score_symmetry(pred_dfs, pheno_dict, auc_dfs,
                        cdata, args, siml_metric):
    assert sorted(auc_dfs['Iso'].index) == sorted(auc_dfs['IsoShal'].index)
    fig, (iso_ax, ish_ax) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2)

    iso_combs = remove_pheno_dups({
        mut for mut, auc_val in auc_dfs['Iso']['mean'].iteritems()
        if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
            and not (mut.all_mtype & shal_mtype).is_empty())
        }, pheno_dict)

    ish_combs = remove_pheno_dups({
        mut for mut, auc_val in auc_dfs['IsoShal']['mean'].iteritems()
        if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
            and (mut.all_mtype & shal_mtype).is_empty()
            and all((mtp & shal_mtype).is_empty() for mtp in mut.mtypes))
        }, pheno_dict)

    pairs_dict = {
        ex_lbl: [
            (mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
            if ((tuple(mcomb1.label_iter())[0]
                 == tuple(mcomb2.label_iter())[0])
                and (all((mtp1 & mtp2).is_empty()
                         for mtp1, mtp2 in product(mcomb1.mtypes,
                                                   mcomb2.mtypes))
                     or not (pheno_dict[mcomb1] & pheno_dict[mcomb2]).any()))
            ]
        for ex_lbl, use_combs in [('Iso', iso_combs), ('IsoShal', ish_combs)]
        }

    if args.verbose:
        for ex_lbl, use_pairs in pairs_dict.items():
            for cur_gene, gene_pairs in pd.Series(use_pairs).reindex(
                use_pairs).groupby(
                    lambda mcombs: tuple(mcombs[0].label_iter())[0]):
                gene_combs = set(reduce(add, gene_pairs.index))

                print('\n'.join([
                    '\n##########',
                    "{}({})  {} pairs from {} types".format(
                        cur_gene, ex_lbl,
                        len(gene_pairs), len(gene_combs)
                        ),
                    '----------'
                    ] + ['\txxxxx\t'.join([str(mcomb) for mcomb in pair])
                         for pair in tuple(gene_pairs.index)[
                             ::(len(gene_pairs)
                                // (args.verbose * 3) + 1)
                            ]]
                    ))

    combs_dict = {ex_lbl: set(reduce(add, use_pairs))
                  for ex_lbl, use_pairs in pairs_dict.items() if use_pairs}

    if not combs_dict:
        return None

    use_genes = {tuple(mcomb.label_iter())[0]
                 for pair_combs in combs_dict.values()
                 for mcomb in pair_combs}
    base_mtree = tuple(cdata.mtrees.values())[0]

    all_mtypes = {
        'Iso': {gene: MuType({('Gene', gene): base_mtree[gene].allkey()})
                for gene in use_genes}
        }

    all_mtypes['IsoShal'] = {
        gene: all_mtype - MuType({('Gene', gene): shal_mtype})
        for gene, all_mtype in all_mtypes['Iso'].items()
        }

    all_phns = {ex_lbl: {gene: np.array(cdata.train_pheno(all_mtype))
                         for gene, all_mtype in all_dict.items()}
                for ex_lbl, all_dict in all_mtypes.items()}

    train_samps = cdata.get_train_samples()
    map_args = []
    ex_indx = []

    for ex_lbl, pair_combs in combs_dict.items():
        ex_indx += [(ex_lbl, mcombs) for mcombs in pairs_dict[ex_lbl]]
        use_preds = pred_dfs[ex_lbl].loc[pair_combs, train_samps].applymap(
            np.mean)

        wt_vals = {
            mcomb: use_preds.loc[mcomb][~all_phns[ex_lbl][
                tuple(mcomb.label_iter())[0]]]
            for mcomb in pair_combs
            }

        mut_vals = {mcomb: use_preds.loc[mcomb, pheno_dict[mcomb]]
                    for mcomb in pair_combs}

        if siml_metric == 'mean':
            wt_means = {mcomb: vals.mean() for mcomb, vals in wt_vals.items()}
            mut_means = {mcomb: vals.mean()
                         for mcomb, vals in mut_vals.items()}

            map_args += [(wt_vals[mcomb1], mut_vals[mcomb1],
                          use_preds.loc[mcomb1, pheno_dict[mcomb2]],
                          wt_means[mcomb1], mut_means[mcomb1], None)
                         for mcombs in pairs_dict[ex_lbl]
                         for mcomb1, mcomb2 in permt(mcombs)]

        elif siml_metric == 'ks':
            base_dists = {
                mcomb: ks_2samp(wt_vals[mcomb], mut_vals[mcomb],
                                alternative='greater').statistic
                for mcomb in pair_combs
                }

            map_args += [(wt_vals[mcomb1], mut_vals[mcomb1],
                          use_preds.loc[mcomb1, pheno_dict[mcomb2]],
                          base_dists[mcomb1])
                         for mcombs in pairs_dict[ex_lbl]
                         for mcomb1, mcomb2 in permt(mcombs)]

    if siml_metric == 'mean':
        chunk_size = int(len(map_args) / args.cores) + 1
    elif siml_metric == 'ks':
        chunk_size = int(len(map_args) / (31 * args.cores)) + 1

    pool = mp.Pool(args.cores)
    siml_list = pool.starmap(SIML_FXS[siml_metric], map_args, chunk_size)
    pool.close()
    siml_vals = dict(zip(ex_indx, zip(siml_list[::2], siml_list[1::2])))

    clr_dict = {gene: choose_label_colour(gene) for gene in use_genes}
    plt_lims = min(siml_list) - 0.19, max(siml_list) + 0.19
    size_mult = 23 * len(map_args) ** 0.23
    clr_norm = colors.Normalize(vmin=-1, vmax=2)

    for ax, ex_lbl in zip([iso_ax, ish_ax], ['Iso', 'IsoShal']):
        ax.grid(alpha=0.47, linewidth=0.9)

        ax.plot(plt_lims, [0, 0],
                color='black', linewidth=1.3, linestyle=':', alpha=0.53)
        ax.plot([0, 0], plt_lims,
                color='black', linewidth=1.3, linestyle=':', alpha=0.53)

        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=1.7, linestyle='--', alpha=0.41)

        for siml_val in [-1, 1, 2]:
            ax.plot(plt_lims, [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.53)
            ax.plot([siml_val] * 2, plt_lims,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.53)

        plt_lctr = plt.MaxNLocator(7, steps=[1, 2, 5])
        ax.xaxis.set_major_locator(plt_lctr)
        ax.yaxis.set_major_locator(plt_lctr)

        for mcomb1, mcomb2 in pairs_dict[ex_lbl]:
            cur_gene = tuple(mcomb1.label_iter())[0]
            plt_sz = size_mult * (np.mean(pheno_dict[mcomb1])
                                  * np.mean(pheno_dict[mcomb2])) ** 0.5

            ax.scatter(*siml_vals[ex_lbl, (mcomb1, mcomb2)],
                       s=plt_sz, c=[clr_dict[cur_gene]], alpha=0.13,
                       edgecolor='none')

        ax.set_xlim(*plt_lims)
        ax.set_ylim(*plt_lims)

        if ex_lbl == 'IsoShal':
            ax.text(1, 0, "AUC >= {:.2f}".format(args.auc_cutoff),
                    size=19, ha='right', va='bottom',
                    transform=ax.transAxes, fontstyle='italic')

    iso_ax.set_title(
        "Similarities Computed Treating\nShallow CNAs as Mutant\n",
        size=23, weight='bold'
        )
    ish_ax.set_title(
        "Similarities Computed Treating\nShallow CNAs as Wild-Type\n",
        size=23, weight='bold'
        )

    plt.tight_layout(w_pad=3.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}-siml-symmetry_{}.svg".format(
                         siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_similarities',
        description="Compares pairs of genes' subgroupings with a cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.7)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(out_dir.glob(
        "out-aucs__*__*__{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort])), exist_ok=True)

    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    out_iter = out_use.groupby('Levels')['File']
    out_aucs = {lvls: list() for lvls in out_iter.groups}
    out_preds = {lvls: list() for lvls in out_iter.groups}

    phn_dict = dict()
    cdata = None

    auc_dfs = {ex_lbl: pd.DataFrame([])
               for ex_lbl in ['All', 'Iso', 'IsoShal']}
    pred_dfs = {ex_lbl: pd.DataFrame([])
                for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for lvls, out_files in out_iter:
        for out_file in out_files:
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                out_aucs[lvls] += [pickle.load(f)]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pred", out_tag])),
                             'r') as f:
                out_preds[lvls] += [pickle.load(f)]

            with bz2.BZ2File(Path(out_dir,
                                  '__'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata is None:
                cdata = new_cdata
            else:
                cdata.merge(new_cdata)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[ex_lbl] = pd.concat([
                    auc_dfs[ex_lbl],
                    pd.DataFrame(out_aucs[lvls][super_indx][ex_lbl])
                    ], sort=False)

                pred_dfs[ex_lbl] = pd.concat([
                    pred_dfs[ex_lbl], out_preds[lvls][super_indx][ex_lbl]],
                    sort=False
                    )

    auc_dfs = {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
               for ex_lbl, auc_df in auc_dfs.items()}

    if 'Consequence__Exon' in out_iter.groups.keys():
        for siml_metric in args.siml_metrics:
            plot_subpoint_divergences(pred_dfs['Iso'], phn_dict,
                                      auc_dfs['Iso']['mean'],
                                      cdata, args, siml_metric)

            for cna_lbl in cna_mtypes:
                plot_subcopy_adjacencies(
                    pred_dfs['Iso'], phn_dict, auc_dfs['Iso']['mean'],
                    cdata, args, cna_lbl, siml_metric
                    )

                plot_copy_interaction(pred_dfs['Iso'], phn_dict,
                                      auc_dfs['Iso']['mean'],
                                      cdata, args, siml_metric)

    else:
        warnings.warn("Cannot analyze the similarities between CNAs and "
                      "point mutation types until this experiment has been "
                      "run with the `Conseqeuence__Exon` mutation level "
                      "combination on this cohort!")

    if args.auc_cutoff < 1:
        for siml_metric in args.siml_metrics:
            plot_score_symmetry(pred_dfs, phn_dict, auc_dfs,
                                cdata, args, siml_metric)


if __name__ == '__main__':
    main()

