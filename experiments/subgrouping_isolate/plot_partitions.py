
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, shal_mtype,
    dup_mtype, loss_mtype, gains_mtype, dels_mtype, Mcomb, ExMcomb
    )
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir
from .utils import siml_fxs, choose_subtype_colour, remove_pheno_dups
from ..utilities.colour_maps import simil_cmap, auc_cmap
from ..utilities.labels import get_fancy_label, get_cohort_label

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

import numpy as np
import pandas as pd
import math
from scipy.stats import ks_2samp

from itertools import combinations as combn
from itertools import permutations as permt
from itertools import product
from functools import reduce
from operator import add

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as ptchs

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'partitions')

PLOT_MAX = 25000


def classify_mtype(mtype):
    if (mtype & copy_mtype).is_empty():
        mtype_cls = 'Point'

    elif (mtype & pnt_mtype).is_empty():
        if (mtype & gains_mtype).is_empty():
            mtype_cls = 'Loss'
        elif (mtype & dels_mtype).is_empty():
            mtype_cls = 'Gain'
        else:
            raise ValueError("Cannot classify `{}`!".format(mtype))

    elif (mtype & gains_mtype).is_empty():
        mtype_cls = 'Point + Loss'
    elif (mtype & dels_mtype).is_empty():
        mtype_cls = 'Point + Gain'

    else:
        raise ValueError("Cannot classify `{}`!".format(mtype))

    return mtype_cls


def classify_mcomb(mcomb):
    if len(mcomb.mtypes) == 1:
        mcomb_cls = classify_mtype(tuple(mcomb.mtypes)[0])

    else:
        mcomb_cls = ' & '.join(sorted(classify_mtype(mtype)
                                      for mtype in mcomb.mtypes))

    return mcomb_cls


def plot_symmetry_decomposition(pred_df, pheno_dict, auc_vals,
                                cdata, args, plt_gene, ex_lbl, siml_metric):
    use_mtree = tuple(cdata.mtrees.values())[0][plt_gene]
    use_combs = auc_vals.index.tolist()

    use_pairs = [
        (mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
        if (all((mtp1 & mtp2).is_empty()
                for mtp1, mtp2 in product(mcomb1.mtypes, mcomb2.mtypes))
            or not (pheno_dict[mcomb1] & pheno_dict[mcomb2]).any())
        ]

    if not use_pairs:
        print("no suitable pairs found among {} possible "
              "mutations for: {}({}) !".format(len(use_combs),
                                               plt_gene, ex_lbl))
        return True

    if len(use_pairs) > PLOT_MAX:
        print("found {} suitable pairs for {}({}), only plotting "
              "the top {} by max AUC!".format(len(use_pairs), plt_gene,
                                              ex_lbl, PLOT_MAX))

        use_pairs = pd.Series({
            tuple(mcombs): max(auc_vals[mcomb] for mcomb in mcombs)
            for mcombs in use_pairs
            }).sort_values()[-(PLOT_MAX):].index.tolist()

    mcomb_clx = {mcomb: classify_mcomb(mcomb) for mcomb in use_combs}
    cls_counts = pd.Series(
        reduce(add, [[mcomb_clx[mcomb] for mcomb in use_pair]
                     for use_pair in use_pairs])
        ).value_counts()

    if len(cls_counts) == 1:
        print("only one partition found, cannot plot decomposition "
              "for {}({}) !".format(plt_gene, ex_lbl))

        return True

    fig, axarr = plt.subplots(
        figsize=(1.5 + 3 * len(cls_counts), 1 + 3 * len(cls_counts)),
        nrows=1 + len(cls_counts), ncols=1 + len(cls_counts),
        gridspec_kw=dict(width_ratios=[1] + [2] * len(cls_counts),
                         height_ratios=[7] * len(cls_counts) + [2])
        )

    all_mtype = MuType({('Gene', plt_gene): use_mtree.allkey()})
    if ex_lbl == 'IsoShal':
        all_mtype -= MuType({('Gene', plt_gene): shal_mtype})

    pair_combs = set(reduce(add, use_pairs))
    train_samps = cdata.get_train_samples()
    use_preds = pred_df.loc[pair_combs, train_samps]
    all_phn = np.array(cdata.train_pheno(all_mtype))

    wt_vals = {mcomb: use_preds.loc[mcomb, ~all_phn] for mcomb in pair_combs}
    mut_vals = {mcomb: use_preds.loc[mcomb, pheno_dict[mcomb]]
                for mcomb in pair_combs}

    if siml_metric == 'mean':
        chunk_size = int(0.91 * len(use_pairs) / args.cores) + 1

        wt_means = {mcomb: vals.mean() for mcomb, vals in wt_vals.items()}
        mut_means = {mcomb: vals.mean() for mcomb, vals in mut_vals.items()}

        map_args = [(wt_vals[mcomb1], mut_vals[mcomb1],
                     use_preds.loc[mcomb1, pheno_dict[mcomb2]],
                     wt_means[mcomb1], mut_means[mcomb1], None)
                    for mcombs in use_pairs
                    for mcomb1, mcomb2 in permt(mcombs)]

    elif siml_metric == 'ks':
        chunk_size = int(0.91 * len(use_pairs) / args.cores) + 1

        base_dists = {mcomb: ks_2samp(wt_vals[mcomb], mut_vals[mcomb],
                                      alternative='greater').statistic
                      for mcomb in pair_combs}

        map_args = [(wt_vals[mcomb1], mut_vals[mcomb1],
                     use_preds.loc[mcomb1, pheno_dict[mcomb2]],
                     base_dists[mcomb1])
                    for mcombs in use_pairs
                    for mcomb1, mcomb2 in permt(mcombs)]

    pool = mp.Pool(args.cores)
    siml_list = pool.starmap(siml_fxs[siml_metric], map_args, chunk_size)
    pool.close()
    siml_vals = dict(zip(use_pairs, zip(siml_list[::2], siml_list[1::2])))

    size_mult = max(727 - math.log(len(use_pairs), 1 + 1 / 77), 31)
    PAIR_CLRS = ['#0DAAFF', '#FF8B00']
    acc_norm = colors.Normalize(vmin=args.auc_cutoff, vmax=auc_vals.max())
    acc_cmap = sns.cubehelix_palette(start=1.07, rot=1.31,
                                     gamma=0.83, light=0.19, dark=0.73,
                                     reverse=True, as_cmap=True)

    plt_sizes = {
        (mcomb1, mcomb2): size_mult * (np.mean(pheno_dict[mcomb1])
                                       * np.mean(pheno_dict[mcomb2])) ** 0.5
        for mcomb1, mcomb2 in use_pairs
        }

    for (i, cls1), (j, cls2) in combn(enumerate(cls_counts.index), 2):
        pair_count = len(plt_sizes)

        for (mcomb1, mcomb2), plt_sz in plt_sizes.items():
            if mcomb_clx[mcomb1] == cls2 and mcomb_clx[mcomb2] == cls1:
                use_clr, use_alpha = PAIR_CLRS[0], 1 / 6.1
            elif mcomb_clx[mcomb1] == cls1 and mcomb_clx[mcomb2] == cls2:
                use_clr, use_alpha = PAIR_CLRS[1], 1 / 6.1

            else:
                use_clr, use_alpha = '0.61', 1 / 17
                pair_count -= 1

            axarr[i, j + 1].scatter(*siml_vals[mcomb1, mcomb2],
                                    c=[use_clr], s=plt_sz, alpha=use_alpha,
                                    edgecolor='none')

            if use_clr in PAIR_CLRS:
                axarr[j, i + 1].scatter(
                    *siml_vals[mcomb1, mcomb2],
                    c=[acc_cmap(acc_norm(auc_vals[mcomb1]))], s=plt_sz,
                    alpha=use_alpha, edgecolor='none'
                    )

        if pair_count == 1:
            pair_lbl = "1 pair"
        else:
            pair_lbl = "{} pairs".format(pair_count)

        axarr[j, i + 1].text(0.01, 1, pair_lbl, size=13,
                             ha='left', va='bottom', fontstyle='italic',
                             transform=axarr[j, i + 1].transAxes)
        axarr[i, j + 1].text(0.99, 1, "({})".format(pair_count), size=13,
                             ha='right', va='bottom', fontstyle='italic',
                             transform=axarr[i, j + 1].transAxes)

    plt_lims = min(siml_list) - 0.07, max(siml_list) + 0.07
    plt_gap = (plt_lims[1] - plt_lims[0]) / 53
    cls_counts: pd.Series
    clx_counts = pd.Series(mcomb_clx).value_counts()

    for i, (cls, cls_count) in enumerate(cls_counts.iteritems()):
        axarr[-1, i + 1].text(0.5, 13 / 17, cls, size=23,
                              ha='center', va='top', fontweight='semibold',
                              transform=axarr[-1, i + 1].transAxes)

        if clx_counts[cls] == 1:
            count_lbl = "1 subgrouping"
        else:
            count_lbl = "{} subgroupings".format(clx_counts[cls])

        axarr[-1, i + 1].text(0.5, -1 / 7, count_lbl, size=19,
                              ha='center', va='bottom', fontstyle='italic',
                              transform=axarr[-1, i + 1].transAxes)

        for (mcomb1, mcomb2), plt_sz in plt_sizes.items():
            if mcomb_clx[mcomb1] == cls and mcomb_clx[mcomb2] == cls:
                use_clr, use_alpha = 'black', 0.37
            elif mcomb_clx[mcomb1] == cls:
                use_clr, use_alpha = PAIR_CLRS[0], 0.19
            elif mcomb_clx[mcomb2] == cls:
                use_clr, use_alpha = PAIR_CLRS[1], 0.19

            else:
                use_clr, use_alpha = '0.73', 1 / 6.1

            axarr[i, i + 1].scatter(*siml_vals[mcomb1, mcomb2],
                                    c=[use_clr], s=plt_sz, alpha=use_alpha,
                                    edgecolor='none')

        if cls_count == 1:
            cls_lbl = "1 total pair"
        else:
            cls_lbl = "{} total pairs".format(cls_count)

        axarr[i, i + 1].text(0.99, 1, cls_lbl, size=13,
                             ha='right', va='bottom', fontstyle='italic',
                             transform=axarr[i, i + 1].transAxes)

        axarr[-2, i + 1].add_patch(ptchs.Rectangle(
            (0.02, -0.23), 0.96, 0.061,
            facecolor=PAIR_CLRS[0], alpha=0.61, edgecolor='none',
            transform=axarr[-2, i + 1].transAxes, clip_on=False
            ))

    clr_ax = axarr[-2, 0].inset_axes(bounds=(1 / 3, -3 / 17, 4 / 7, 43 / 23),
                                     clip_on=False, in_layout=False)
    clr_bar = ColorbarBase(ax=clr_ax, cmap=acc_cmap, norm=acc_norm,
                           ticklocation='left')

    clr_ax.set_title("AUC", size=21, fontweight='bold')
    clr_ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 4, 5]))
    tcks_loc = clr_ax.get_yticks().tolist()
    clr_ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(tcks_loc))
    clr_bar.ax.set_yticklabels([format(tick, '.2f').lstrip('0')
                                for tick in tcks_loc],
                               size=15, fontweight='semibold')

    siml_norm = colors.Normalize(vmin=-1, vmax=2)
    plt_lctr = plt.MaxNLocator(5, steps=[1, 2, 5])
    for ax in axarr[:-1, 1:].flatten():
        ax.grid(alpha=0.47, linewidth=0.7)

        ax.plot(plt_lims, [0, 0],
                color='black', linewidth=0.83, linestyle=':', alpha=0.47)
        ax.plot([0, 0], plt_lims,
                color='black', linewidth=0.83, linestyle=':', alpha=0.47)

        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=1.13, linestyle='--', alpha=0.37)

        for siml_val in [-1, 1, 2]:
            ax.plot(plt_lims, [siml_val] * 2,
                    color=simil_cmap(siml_norm(siml_val)),
                    linewidth=2.7, linestyle=':', alpha=0.31)
            ax.plot([siml_val] * 2, plt_lims,
                    color=simil_cmap(siml_norm(siml_val)),
                    linewidth=2.7, linestyle=':', alpha=0.31)

        ax.set_xlim(*plt_lims)
        ax.set_ylim(*plt_lims)
        ax.xaxis.set_major_locator(plt_lctr)
        ax.yaxis.set_major_locator(plt_lctr)

    for i in range(len(cls_counts)):
        for j in range(1, len(cls_counts) + 1):
            if i != (j - 1):
                axarr[i, j].set_xticklabels([])
                axarr[i, j].set_yticklabels([])

            else:
                axarr[i, j].tick_params(labelsize=11)

    for ax in axarr[:, 0].tolist() + axarr[-1, :].tolist():
        ax.axis('off')

    plt.tight_layout(w_pad=2 / 7, h_pad=2 / 7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_{}_{}-symm-decomposition_{}.svg".format(
                         plt_gene, ex_lbl, siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Creates plots describing how the mutations of a particular gene or "
        "genes can be segregated according to similarity with regard to "
        "downstream effects within a particular cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour sample cohort")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--genes', '-g', nargs='+',
                        help="restrict plots drawn to these mutated genes?")
    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.7)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})
    parser.add_argument('--cores', '-c', type=int, default=1)

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(out_dir.glob(
        "out-aucs__*__*__{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    # create directory where plots will be stored
    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort])), exist_ok=True)

    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
    )

    out_iter = out_use.groupby('Levels')['File']
    phn_dict = dict()
    cdata = None

    auc_df = pd.DataFrame([])
    pred_dfs = {ex_lbl: pd.DataFrame([])
                for ex_lbl in ['Iso', 'IsoShal']}

    # load experiment output
    for lvls, out_files in out_iter:
        out_aucs = list()
        out_preds = {ex_lbl: list() for ex_lbl in ['Iso', 'IsoShal']}

        for out_file in out_files:
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                auc_vals = pickle.load(f)

            auc_vals = pd.DataFrame({
                ex_lbl: auc_vals[ex_lbl]['mean'][
                    [mut for mut in auc_vals[ex_lbl]['mean'].index
                     if isinstance(mut, ExMcomb)]
                    ]
                for ex_lbl in ['Iso', 'IsoShal']
                })

            if args.genes:
                auc_vals = auc_vals.loc[[
                    mcomb for mcomb in auc_vals.index
                    if tuple(mcomb.label_iter())[0] in set(args.genes)
                    ]]

            out_aucs += [auc_vals]

            for ex_lbl in ['Iso', 'IsoShal']:
                with bz2.BZ2File(Path(out_dir,
                                      '__'.join(["out-pred_{}".format(ex_lbl),
                                                 out_tag])),
                                 'r') as f:
                    pred_vals = pickle.load(f)

                out_preds[ex_lbl] += [
                    pred_vals.loc[
                        out_aucs[-1][ex_lbl].index].applymap(np.mean)
                    ]

            with bz2.BZ2File(Path(out_dir,
                                  '__'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata is None:
                cdata = new_cdata
            else:
                cdata.merge(new_cdata, use_genes=args.genes)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals.index) for auc_vals in out_aucs]] * 2))
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if not super_list.any():
            for ex_lbl in ['Iso', 'IsoShal']:
                for aucs in out_aucs:
                    auc_dfs[ex_lbl] = pd.concat([auc_dfs[ex_lbl],
                                                 aucs[ex_lbl]], sort=False)

                for preds in out_preds[ex_lbl]:
                    pred_dfs[ex_lbl] = pd.concat([pred_dfs[ex_lbl], preds],
                                                 sort=False)

        else:
            super_indx = super_list.argmax()

            auc_df = pd.concat([auc_df,
                                pd.DataFrame(out_aucs[super_indx])],
                               sort=False)

            for ex_lbl in ['Iso', 'IsoShal']:
                pred_dfs[ex_lbl] = pd.concat([pred_dfs[ex_lbl],
                                              out_preds[ex_lbl][super_indx]],
                                             sort=False)

    auc_df = auc_df.loc[~auc_df.index.duplicated()]
    pred_dfs = {ex_lbl: pred_df.loc[~pred_df.index.duplicated()]
                for ex_lbl, pred_df in pred_dfs.items()}

    if cdata._muts.shape[0] == 0:
        raise ValueError("No mutation calls found in cohort "
                         "`{}` for these genes!".format(args.cohort))

    if not phn_dict:
        raise ValueError("No mutation types passing test search criteria "
                         "found for this combination of parameters!")

    use_mtypes = {
        'Iso': remove_pheno_dups({
            mcomb for mcomb, auc_val in auc_df['Iso'].iteritems()
            if (auc_val >= args.auc_cutoff
                and not (mcomb.all_mtype & shal_mtype).is_empty())
            }, phn_dict),

        'IsoShal': remove_pheno_dups({
            mcomb for mcomb, auc_val in auc_df['IsoShal'].iteritems()
            if (auc_val >= args.auc_cutoff
                and (mcomb.all_mtype & shal_mtype).is_empty()
                and all((mtp & shal_mtype).is_empty()
                        for mtp in mcomb.mtypes))
            }, phn_dict)
        }

    assert not (use_mtypes['Iso'] & use_mtypes['IsoShal'])
    for ex_lbl in ['Iso', 'IsoShal']:
        for gene, auc_vals in auc_df.loc[use_mtypes[ex_lbl], ex_lbl].groupby(
                lambda mcomb: tuple(mcomb.label_iter())[0]):

            pair_stat = None
            for siml_metric in args.siml_metrics:
                if not pair_stat:
                    pair_stat = plot_symmetry_decomposition(
                        pred_dfs[ex_lbl], phn_dict, auc_vals,
                        cdata, args, gene, ex_lbl, siml_metric
                        )


if __name__ == '__main__':
    main()

