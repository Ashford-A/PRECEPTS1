"""
Creates assorted plots for the output related to one particular
mutated gene across all tested cohorts.
"""

from ..utilities.mutations import (
    pnt_mtype, copy_mtype, shal_mtype,
    dup_mtype, loss_mtype, gains_mtype, dels_mtype, Mcomb, ExMcomb
    )
from dryadic.features.mutations import MuType

from .utils import remove_pheno_dups
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..subvariant_test import variant_clrs
from ..subvariant_isolate import mcomb_clrs
from ..utilities.colour_maps import simil_cmap

from ..utilities.labels import get_fancy_label
from ..subvariant_test.utils import get_cohort_label
from ..utilities.label_placement import place_scatterpie_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from itertools import combinations as combn
from itertools import permutations as permt
from itertools import product
from functools import reduce
from operator import or_, add

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Wedge

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'gene')


SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}


def choose_subtype_colour(mut):
    if (copy_mtype & mut).is_empty():
        mut_clr = variant_clrs['Point']

    elif gains_mtype.is_supertype(mut):
        mut_clr = variant_clrs['Gain']
    elif dels_mtype.is_supertype(mut):
        mut_clr = variant_clrs['Loss']

    elif not (gains_mtype & mut).is_empty():
        mut_clr = mcomb_clrs['Point+Gain']
    elif not (dels_mtype & mut).is_empty():
        mut_clr = mcomb_clrs['Point+Loss']

    return mut_clr


def plot_size_comparisons(auc_vals, pheno_dict, conf_vals,
                          use_coh, args, add_lgnd=False):
    fig, ax = plt.subplots(figsize=(13, 8))

    pnt_dict = dict()
    clr_dict = dict()

    plt_df = pd.DataFrame({
        mut: {'Size': np.sum(pheno_dict[mut]), 'AUC': auc_val}
        for mut, auc_val in auc_vals.iteritems()
        }).transpose().astype({'Size': int})

    #TODO: differentiate between deep- and shal-exclusive mutations?
    for mut, (size_val, auc_val) in plt_df.iterrows():
        if isinstance(mut, MuType):
            sub_mut = tuple(mut.subtype_iter())[0][1]
            plt_mrk = 'o'
            plt_clr = choose_subtype_colour(sub_mut)

            if sub_mut.is_supertype(pnt_mtype):
                plt_sz = 413
                lbl_gap = 0.31

                if sub_mut == pnt_mtype:
                    plt_lbl = "Any Point"

                elif sub_mut.is_supertype(dup_mtype):
                    plt_lbl = "Any Point + Any Gain"
                elif sub_mut.is_supertype(loss_mtype):
                    plt_lbl = "Any Point + Any Loss"

                elif sub_mut.is_supertype(dup_mtype):
                    plt_lbl = "Any Point + Deep Gains"
                elif sub_mut.is_supertype(loss_mtype):
                    plt_lbl = "Any Point + Deep Losses"

            else:
                plt_lbl = ''
                plt_sz = 31
                lbl_gap = 0.13

        elif len(mut.mtypes) == 1:
            iso_mtype = tuple(tuple(mut.mtypes)[0].subtype_iter())[0][1]
            plt_mrk = 'D'
            plt_clr = choose_subtype_colour(iso_mtype)

            if iso_mtype.is_supertype(pnt_mtype):
                plt_sz = 413
                lbl_gap = 0.31

                if iso_mtype == pnt_mtype:
                    plt_lbl = "Only: Any Point"

                elif iso_mtype.is_supertype(gains_mtype):
                    plt_lbl = "Only: Any Point + Any Gain"
                elif iso_mtype.is_supertype(dels_mtype):
                    plt_lbl = "Only: Any Point + Any Loss"

                elif iso_mtype.is_supertype(dup_mtype):
                    plt_lbl = "Only: Any Point + Deep Gains"
                elif iso_mtype.is_supertype(loss_mtype):
                    plt_lbl = "Only: Any Point + Deep Losses"

            else:
                plt_lbl = ''
                plt_sz = 37
                lbl_gap = 0.19

        else:
            plt_mrk = 'D'

            if (size_val, auc_val) not in clr_dict:
                plt_clr = 'black'

            plt_lbl = ''
            plt_sz = 47
            lbl_gap = 0.29

        if (plt_lbl is not None
                and not ((size_val, auc_val) in pnt_dict
                         and pnt_dict[size_val, auc_val][0] > lbl_gap)):
            clr_dict[size_val, auc_val] = plt_clr
            pnt_dict[size_val, auc_val] = lbl_gap, (plt_lbl, '')

        ax.scatter(size_val, auc_val, marker=plt_mrk,
                   c=[plt_clr], s=plt_sz, alpha=0.19, edgecolor='none')

    size_min, size_max = plt_df.Size.quantile(q=[0, 1])
    auc_min, auc_max = plt_df.AUC.quantile(q=[0, 1])

    x_min = max(size_min - (size_max - size_min) / 29, 0)
    x_max = size_max + (size_max - size_min) / 29
    y_min, y_max = auc_min - (1 - auc_min) / 17, 1 + (1 - auc_min) / 113
    x_rng, y_rng = x_max - x_min, y_max - y_min

    ax.plot([x_min, x_max], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([x_min, x_max], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0, 0], [y_min, y_max], color='black', linewidth=1.9, alpha=0.89)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    coh_lbl = get_cohort_label(use_coh)
    ax.set_xlabel("# of Mutated Samples in {}".format(coh_lbl),
                  size=25, weight='semibold')
    ax.set_ylabel("Classification Task\nAccuracy in {}".format(coh_lbl),
                  size=25, weight='semibold')

    if pnt_dict:
        lbl_pos = place_scatterpie_labels(pnt_dict, fig, ax, seed=args.seed)

        for (pnt_x, pnt_y), pos in lbl_pos.items():
            ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                    pnt_dict[pnt_x, pnt_y][1][0],
                    size=11, ha=pos[1], va='bottom')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]

            if abs(x_delta) > x_rng / 23 or abs(y_delta) > y_rng / 23:
                end_x = pos[0][0] + np.sign(x_delta) * x_rng / 203
                end_y = pos[0][1] + np.heaviside(y_delta, 0) * y_rng / 29

                ln_x, ln_y = (pnt_x - end_x) / x_rng, (pnt_y - end_y) / y_rng
                ln_mag = (ln_x ** 2 + ln_y ** 2) ** 0.5
                ln_cos, ln_sin = ln_x / ln_mag, ln_y / ln_mag

                ax.plot([pnt_x - ln_cos * x_rng / 11
                         * pnt_dict[pnt_x, pnt_y][0], end_x],
                        [pnt_y - ln_sin * y_rng / 11
                         * pnt_dict[pnt_x, pnt_y][0], end_y],
                        c=clr_dict[pnt_x, pnt_y], linewidth=1.7, alpha=0.31)

    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__size-comparison_{}_{}.svg".format(
                         use_coh, args.classif, args.expr_source)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_comparisons(auc_dfs, pheno_dict, use_coh, args):
    fig, axarr = plt.subplots(figsize=(15, 15), nrows=3, ncols=3)

    base_aucs = {
        ex_lbl: auc_vals[[not isinstance(mtype, (Mcomb, ExMcomb))
                          for mtype in auc_vals.index]]
        for ex_lbl, auc_vals in auc_dfs.items()
        }

    base_mtypes = {tuple(sorted(auc_vals.index))
                   for auc_vals in base_aucs.values()}
    assert len(base_mtypes) == 1, ("Mismatching mutation types across "
                                   "isolation testing holdout modes!")

    base_mtypes = tuple(base_mtypes)[0]
    iso_aucs = {'All': base_aucs['All']}

    iso_aucs['Iso'] = auc_dfs['Iso'][[
        isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
        and tuple(mcomb.mtypes)[0] in base_mtypes
        and not (mcomb.all_mtype & shal_mtype).is_empty()
        for mcomb in auc_dfs['Iso'].index
        ]]

    iso_aucs['IsoShal'] = auc_dfs['IsoShal'][[
        isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
        and tuple(mcomb.mtypes)[0] in base_mtypes
        and (mcomb.all_mtype & shal_mtype).is_empty()
        for mcomb in auc_dfs['IsoShal'].index
        ]]

    assert not set(iso_aucs['Iso'].index & iso_aucs['IsoShal'].index)
    for ex_lbl in ('Iso', 'IsoShal'):
        iso_aucs[ex_lbl].index = [tuple(mcomb.mtypes)[0]
                                  for mcomb in iso_aucs[ex_lbl].index]

    plt_min = 0.83
    for (i, ex_lbl1), (j, ex_lbl2) in combn(enumerate(base_aucs.keys()), 2):
        for mtype, auc_val1 in base_aucs[ex_lbl1].iteritems():
            plt_min = min(plt_min, auc_val1 - 0.013,
                          base_aucs[ex_lbl2][mtype] - 0.013)

            mtype_sz = 503 * np.mean(pheno_dict[mtype])
            plt_clr = choose_subtype_colour(tuple(mtype.subtype_iter())[0][1])

            axarr[i, j].scatter(base_aucs[ex_lbl2][mtype], auc_val1,
                                c=[plt_clr], s=mtype_sz,
                                alpha=0.19, edgecolor='none')

        for mtype in set(iso_aucs[ex_lbl1].index & iso_aucs[ex_lbl2].index):
            plt_x = iso_aucs[ex_lbl1][mtype]
            plt_y = iso_aucs[ex_lbl2][mtype]

            plt_min = min(plt_min, plt_x - 0.013, plt_y - 0.013)
            mtype_sz = 503 * np.mean(pheno_dict[mtype])
            plt_clr = choose_subtype_colour(tuple(mtype.subtype_iter())[0][1])

            axarr[j, i].scatter(plt_x, plt_y, c=[plt_clr],
                                s=mtype_sz, alpha=0.19, edgecolor='none')

    for i, j in permt(range(3), r=2):
        axarr[i, j].grid(alpha=0.53, linewidth=0.7)
        axarr[j, i].grid(alpha=0.53, linewidth=0.7)

        if j - i != 1 and i < 2:
            axarr[i, j].xaxis.set_major_formatter(plt.NullFormatter())
        else:
            axarr[i, j].xaxis.set_major_locator(
                plt.MaxNLocator(7, steps=[1, 2, 4]))

        if j - i != 1 and j > 0:
            axarr[i, j].yaxis.set_major_formatter(plt.NullFormatter())
        else:
            axarr[i, j].yaxis.set_major_locator(
                plt.MaxNLocator(7, steps=[1, 2, 4]))

        axarr[i, j].plot([plt_min, 1], [0.5, 0.5], color='black',
                         linewidth=1.3, linestyle=':', alpha=0.71)
        axarr[i, j].plot([0.5, 0.5], [plt_min, 1], color='black',
                         linewidth=1.3, linestyle=':', alpha=0.71)

        axarr[i, j].plot([plt_min, 1], [1, 1],
                         color='black', linewidth=1.7, alpha=0.89)
        axarr[i, j].plot([1, 1], [plt_min, 1],
                         color='black', linewidth=1.7, alpha=0.89)

        axarr[i, j].plot([plt_min, 0.997], [plt_min, 0.997], color='#550000',
                         linewidth=2.1, linestyle='--', alpha=0.41)

        axarr[i, j].set_xlim([plt_min, 1 + (1 - plt_min) / 113])
        axarr[i, j].set_ylim([plt_min, 1 + (1 - plt_min) / 113])

    for i, (ex_lbl, auc_vals) in enumerate(base_aucs.items()):
        axarr[i, i].axis('off')
        axarr[i, i].text(0.5, 0.5, ex_lbl,
                         size=37, fontweight='bold', ha='center', va='center')

    plt.tight_layout(w_pad=1.7, h_pad=1.7)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__iso-comparisons_{}_{}.svg".format(
                         use_coh, args.classif, args.expr_source)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_dyad_comparisons(auc_vals, pheno_dict, conf_vals, use_coh, args):
    fig, (gain_ax, loss_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    pnt_aucs = auc_vals[[
        not isinstance(mtype, (Mcomb, ExMcomb))
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    plot_df = pd.DataFrame(
        index=pnt_aucs.index,
        columns=pd.MultiIndex.from_product([['gain', 'loss'],
                                            ['all', 'deep']]),
        dtype=float
        )

    for pnt_type, (copy_indx, copy_type) in product(
            pnt_aucs.index,
            zip(plot_df.columns, [gains_mtype, dup_mtype,
                                  dels_mtype, loss_mtype])
            ):
        dyad_type = MuType({('Gene', args.gene): copy_type}) | pnt_type

        if dyad_type in auc_vals.index:
            plot_df.loc[pnt_type, copy_indx] = auc_vals[dyad_type]

    plt_min = 0.83
    for ax, copy_lbl in zip([gain_ax, loss_ax], ['gain', 'loss']):
        for dpth_lbl in ['all', 'deep']:
            copy_aucs = plot_df[copy_lbl, dpth_lbl]
            copy_aucs = copy_aucs[~copy_aucs.isnull()]

            for pnt_type, copy_auc in copy_aucs.iteritems():
                plt_min = min(plt_min,
                              pnt_aucs[pnt_type] - 0.03, copy_auc - 0.03)

                mtype_sz = 1003 * np.mean(pheno_dict[pnt_type])
                plt_clr = choose_subtype_colour(
                    tuple(pnt_type.subtype_iter())[0][1])

                if dpth_lbl == 'all':
                    dpth_clr = plt_clr
                    edg_lw = 0
                else:
                    dpth_clr = 'none'
                    edg_lw = mtype_sz ** 0.5 / 4.7

                ax.scatter(pnt_aucs[pnt_type], copy_auc,
                           facecolor=dpth_clr, s=mtype_sz, alpha=0.21,
                           edgecolor=plt_clr, linewidths=edg_lw)

    for copy_lbl, copy_type, copy_ax, copy_lw in zip(
            ['All Gains', 'Deep Gains', 'All Losses', 'Deep Losses'],
            [gains_mtype, dup_mtype, dels_mtype, loss_mtype],
            [gain_ax, gain_ax, loss_ax, loss_ax],
            [3.1, 4.3, 3.1, 4.3]
        ):
        gene_copy = MuType({('Gene', args.gene): copy_type})

        if gene_copy in auc_vals.index:
            copy_auc = auc_vals[gene_copy]
            copy_clr = choose_subtype_colour(copy_type)
            use_lbl = ' '.join([copy_lbl.split(' ')[0], args.gene,
                                copy_lbl.split(' ')[1]])

            copy_ax.text(max(plt_min, 0.51), copy_auc + (1 - copy_auc) / 173,
                         use_lbl, c=copy_clr, size=13, ha='left', va='bottom')
            copy_ax.plot([plt_min, 1], [copy_auc, copy_auc], color=copy_clr,
                         linewidth=copy_lw, linestyle=':', alpha=0.83)

    plt_lims = plt_min, 1 + (1 - plt_min) / 91
    for ax in (gain_ax, loss_ax):
        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

        ax.plot(plt_lims, [0.5, 0.5],
                color='black', linewidth=1.1, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims,
                color='black', linewidth=1.1, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.7, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.7, alpha=0.89)
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=1.9, linestyle='--', alpha=0.41)

        ax.set_xlabel("Accuracy of Subgrouping Classifier",
                      size=23, weight='semibold')

    gain_ax.set_ylabel("Accuracy of\n(Subgrouping or CNAs) Classifier",
                       size=23, weight='semibold')
    gain_ax.set_title("Gain CNAs", size=27, weight='semibold')
    loss_ax.set_title("Loss CNAs", size=27, weight='semibold')

    plt.tight_layout(w_pad=3.1)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__dyad-comparisons_{}_{}.svg".format(
                         use_coh, args.classif, args.expr_source)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_score_symmetry(pred_dfs, pheno_dict, auc_dfs, cdata,
                        args, use_coh, siml_metric):
    fig, (iso_ax, ish_ax) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2)

    use_mtree = tuple(cdata.mtrees.values())[0][args.gene]
    all_mtypes = {
        'Iso': MuType({('Gene', args.gene): use_mtree.allkey()})}
    all_mtypes['IsoShal'] = all_mtypes['Iso'] - MuType({
        ('Gene', args.gene): shal_mtype})

    all_phns = {ex_lbl: np.array(cdata.train_pheno(all_mtype))
                for ex_lbl, all_mtype in all_mtypes.items()}
    train_samps = cdata.get_train_samples()

    iso_combs = remove_pheno_dups({
        mut for mut, auc_val in auc_dfs['Iso'].iteritems()
        if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
            and not (mut.all_mtype & shal_mtype).is_empty())
        }, pheno_dict)

    ish_combs = remove_pheno_dups({
        mut for mut, auc_val in auc_dfs['IsoShal'].iteritems()
        if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
            and (mut.all_mtype & shal_mtype).is_empty()
            and all((mtp & shal_mtype).is_empty() for mtp in mut.mtypes))
        }, pheno_dict)

    pairs_dict = {
        ex_lbl: [
            (mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
            if (all((mtp1 & mtp2).is_empty()
                    for mtp1, mtp2 in product(mcomb1.mtypes, mcomb2.mtypes))
                or not (pheno_dict[mcomb1] & pheno_dict[mcomb2]).any())
            ]
        for ex_lbl, use_combs in [('Iso', iso_combs), ('IsoShal', ish_combs)]
        }

    if args.verbose:
        for ex_lbl, use_combs in zip(['Iso', 'IsoShal'],
                                     [iso_combs, ish_combs]):
            pair_strs = [
                "###########"
                "{}: {}({})  {} pairs from {} types".format(
                    use_coh, args.gene, ex_lbl,
                    len(pairs_dict[ex_lbl]), len(use_combs)
                    )
                ]

            if pairs_dict[ex_lbl]:
                pair_strs += ['----------']

                pair_strs += [
                    '\txxxxx\t'.join([str(mcomb) for mcomb in pair])
                    for pair in pairs_dict[ex_lbl][
                        ::(len(pairs_dict[ex_lbl]) // (args.verbose * 7) + 1)]
                    ]

            print('\n'.join(pair_strs))

    combs_dict = {ex_lbl: set(reduce(add, use_pairs))
                  for ex_lbl, use_pairs in pairs_dict.items() if use_pairs}

    if not combs_dict:
        return None

    map_args = []
    ex_indx = []

    for ex_lbl, pair_combs in combs_dict.items():
        ex_indx += [(ex_lbl, mcombs) for mcombs in pairs_dict[ex_lbl]]
        use_preds = pred_dfs[ex_lbl].loc[pair_combs, train_samps].applymap(
            np.mean)

        wt_vals = {mcomb: use_preds.loc[mcomb, ~all_phns[ex_lbl]]
                   for mcomb in pair_combs}
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
            base_dists = {mcomb: ks_2samp(wt_vals[mcomb], mut_vals[mcomb],
                                          alternative='greater').statistic
                          for mcomb in pair_combs}

            map_args += [(wt_vals[mcomb1], mut_vals[mcomb1],
                          use_preds.loc[mcomb1, pheno_dict[mcomb2]],
                          base_dists[mcomb1])
                         for mcombs in pairs_dict[ex_lbl]
                         for mcomb1, mcomb2 in permt(mcombs)]

    if siml_metric == 'mean':
        chunk_size = int(0.43 * len(map_args) / args.cores) + 1
    elif siml_metric == 'ks':
        chunk_size = int(0.07 * len(map_args) / args.cores) + 1

    pool = mp.Pool(args.cores)
    siml_list = pool.starmap(SIML_FXS[siml_metric], map_args, chunk_size)
    pool.close()
    siml_vals = dict(zip(ex_indx, zip(siml_list[::2], siml_list[1::2])))

    #TODO: scale by plot ranges or leave as is and thus make sizes
    # relative to "true" plotting area?
    plt_lims = min(siml_list) - 0.19, max(siml_list) + 0.19
    size_mult = (plt_lims[1] - plt_lims[0]) / (113 * len(map_args) ** 0.23)
    gap_adj = (plt_lims[1] - plt_lims[0]) / 9173 * np.array([1, -1])
    clr_norm = colors.Normalize(vmin=-1, vmax=2)

    for ax, ex_lbl in zip([iso_ax, ish_ax], ['Iso', 'IsoShal']):
        ax.grid(alpha=0.47, linewidth=0.9)
        plt_wdgs = []

        ax.plot(plt_lims, [0, 0],
                color='black', linewidth=1.37, linestyle=':', alpha=0.53)
        ax.plot([0, 0], plt_lims,
                color='black', linewidth=1.37, linestyle=':', alpha=0.53)

        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=1.43, linestyle='--', alpha=0.41)

        for siml_val in [-1, 1, 2]:
            ax.plot(plt_lims, [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.37)
            ax.plot([siml_val] * 2, plt_lims,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.37)

        plt_lctr = plt.MaxNLocator(7, steps=[1, 2, 5])
        ax.xaxis.set_major_locator(plt_lctr)
        ax.yaxis.set_major_locator(plt_lctr)

        for mcomb1, mcomb2 in pairs_dict[ex_lbl]:
            plt_sz = size_mult * (np.mean(pheno_dict[mcomb1])
                                  * np.mean(pheno_dict[mcomb2])) ** 0.5

            plt_clrs = [
                choose_subtype_colour(
                    tuple(reduce(or_, mcomb.mtypes).subtype_iter())[0][1])
                for mcomb in [mcomb1, mcomb2]
                ]

            if plt_clrs[0] == plt_clrs[1]:
                ax.add_patch(Wedge(siml_vals[ex_lbl, (mcomb1, mcomb2)],
                                   plt_sz ** 0.5, 0, 360,
                                   facecolor=plt_clrs[0], alpha=0.19,
                                   edgecolor='none', lw=0,
                                   transform=ax.transData))

            else:
                for i, mcomb in enumerate([mcomb1, mcomb2]):
                    plt_pos = np.array([gap_adj[i], 0])
                    plt_pos += siml_vals[ex_lbl, (mcomb1, mcomb2)]

                    ax.add_patch(Wedge(plt_pos, plt_sz ** 0.5,
                                       90 + i * 180, 90 + (i + 1) * 180,
                                       facecolor=plt_clrs[i], alpha=0.19,
                                       edgecolor='none', lw=0,
                                       transform=ax.transData))

    iso_ax.set_title(
        "Similarities Computed Treating\nShallow CNAs as Mutant\n",
        size=23, weight='semibold'
        )
    ish_ax.set_title(
        "Similarities Computed Treating\nShallow CNAs as Wild-Type\n",
        size=23, weight='semibold'
        )

    for ax in [iso_ax, ish_ax]:
        ax.set_xlim(*plt_lims)
        ax.set_ylim(*plt_lims)

    plt.tight_layout(w_pad=3.1)
    plt.savefig(os.path.join(
        plot_dir, args.gene, "{}__{}-siml-symmetry_{}_{}.svg".format(
            use_coh, siml_metric, args.classif, args.expr_source)
        ), bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_gene',
        description="Plots gene-specific experiment output across cohorts."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('gene', help="a mutated gene")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--cohorts', nargs='+')
    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.7)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="print info about created plots")

    args = parser.parse_args()
    out_list = tuple(Path(base_dir).glob(
        os.path.join("{}__*".format(args.expr_source),
                     "out-conf__*__*__{}.p.gz".format(args.classif))
        ))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_df = pd.DataFrame(
        [{'Cohort': out_file.parts[-2].split('__')[1],
          'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    if args.cohorts:
        out_df = out_df.loc[out_df.Cohort.isin(args.cohorts)]

        if out_df.shape[0] == 0:
            raise ValueError("No completed experiments found for given "
                             "cohort(s) {} !".format(set(args.cohorts)))

    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    out_iter = out_df.groupby(['Cohort', 'Levels'])['File']
    phn_dicts = {coh: dict() for coh in out_df.Cohort.unique()}
 
    out_dirs = {coh: Path(base_dir, '__'.join([args.expr_source, coh]))
                for coh in out_df.Cohort.values}
    out_tags = {fl: '__'.join(fl.parts[-1].split('__')[1:])
                for fl in out_df.File}

    for (coh, lvls), out_files in out_iter:
        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[coh],
                                  '__'.join(["out-pheno",
                                             out_tags[out_file]])),
                             'r') as f:
                phn_vals = pickle.load(f)

            phn_dicts[coh].update({
                mut: phns for mut, phns in phn_vals.items()
                if tuple(mut.label_iter())[0] == args.gene
                })

    use_cohs = {coh for coh, phn_dict in phn_dicts.items() if phn_dict}
    if not use_cohs:
        raise ValueError("No completed experiments found having tested "
                         "mutations of the gene {} for the given "
                         "parameters!".format(args.gene))

    out_use = out_df.loc[out_df.Cohort.isin(use_cohs)]
    use_iter = out_use.groupby(['Cohort', 'Levels'])['File']

    out_aucs = {(coh, lvls): list() for coh, lvls in use_iter.groups}
    out_confs = {(coh, lvls): list() for coh, lvls in use_iter.groups}
    out_preds = {(coh, lvls): list() for coh, lvls in use_iter.groups}
    cdata_dict = {coh: None for coh, _ in use_iter.groups}

    auc_dfs = {coh: {ex_lbl: pd.DataFrame([])
                     for ex_lbl in ['All', 'Iso', 'IsoShal']}
               for coh in use_cohs}
    conf_dfs = {coh: {ex_lbl: pd.DataFrame([])
                      for ex_lbl in ['All', 'Iso', 'IsoShal']}
                for coh in use_cohs}
    pred_dfs = {coh: {ex_lbl: pd.DataFrame([])
                      for ex_lbl in ['All', 'Iso', 'IsoShal']}
                for coh in use_cohs}

    for (coh, lvls), out_files in use_iter:
        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[coh],
                                  '__'.join(["out-aucs",
                                             out_tags[out_file]])),
                             'r') as f:
                auc_vals = pickle.load(f)

            out_aucs[coh, lvls] += [
                {ex_lbl: auc_df.loc[[mut for mut in auc_df.index
                                     if args.gene in mut.label_iter()]]
                 for ex_lbl, auc_df in auc_vals.items()}
                ]

            with bz2.BZ2File(Path(out_dirs[coh],
                                  '__'.join(["out-conf",
                                             out_tags[out_file]])),
                             'r') as f:
                conf_vals = pickle.load(f)

            out_confs[coh, lvls] += [{
                ex_lbl: pd.DataFrame(conf_dict).loc[
                    out_aucs[coh, lvls][-1][ex_lbl].index]
                for ex_lbl, conf_dict in conf_vals.items()
                }]

            with bz2.BZ2File(Path(out_dirs[coh],
                                  '__'.join(["out-pred",
                                             out_tags[out_file]])),
                             'r') as f:
                pred_vals = pickle.load(f)

            out_preds[coh, lvls] += [{
                ex_lbl: pd.DataFrame(pred_dict).loc[
                    out_aucs[coh, lvls][-1][ex_lbl].index]
                for ex_lbl, pred_dict in pred_vals.items()
                }]

            with bz2.BZ2File(Path(out_dirs[coh],
                                  '__'.join(["cohort-data",
                                             out_tags[out_file]])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[coh] is None:
                cdata_dict[coh] = new_cdata
            else:
                cdata_dict[coh].merge(new_cdata, use_genes=[args.gene])

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[coh, lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[coh][ex_lbl] = pd.concat([
                    auc_dfs[coh][ex_lbl],
                    out_aucs[coh, lvls][super_indx][ex_lbl]
                    ], sort=False)

                conf_dfs[coh][ex_lbl] = pd.concat([
                    conf_dfs[coh][ex_lbl],
                    out_confs[coh, lvls][super_indx][ex_lbl]
                    ], sort=False)

                pred_dfs[coh][ex_lbl] = pd.concat([
                    pred_dfs[coh][ex_lbl],
                    out_preds[coh, lvls][super_indx][ex_lbl]
                    ], sort=False)

    for coh, coh_lvls in out_use.groupby('Cohort')['Levels']:
        for ex_lbl in ['All', 'Iso', 'IsoShal']:
            auc_dfs[coh][ex_lbl] = auc_dfs[coh][ex_lbl]['mean'].loc[
                ~auc_dfs[coh][ex_lbl].index.duplicated()]
            conf_dfs[coh][ex_lbl] = conf_dfs[coh][ex_lbl]['mean'].loc[
                ~conf_dfs[coh][ex_lbl].index.duplicated()]

        plot_size_comparisons(auc_dfs[coh]['All'], phn_dicts[coh],
                              conf_dfs[coh]['All'], coh, args)

        plot_iso_comparisons(auc_dfs[coh], phn_dicts[coh], coh, args)
        plot_dyad_comparisons(auc_dfs[coh]['All'], phn_dicts[coh],
                              conf_dfs[coh]['All'], coh, args)

        for siml_metric in args.siml_metrics:
            plot_score_symmetry(pred_dfs[coh], phn_dicts[coh], auc_dfs[coh],
                                cdata_dict[coh], args, coh, siml_metric)

        if 'Consequence__Exon' not in set(coh_lvls.tolist()):
            if args.verbose:
                print("Cannot compare AUCs until this experiment is run "
                      "with mutation levels `Consequence__Exon` "
                      "which tests genes' base mutations!")


if __name__ == '__main__':
    main()

