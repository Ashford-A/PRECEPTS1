
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, shal_mtype, deep_mtype,
    dup_mtype, gains_mtype, loss_mtype, dels_mtype, ExMcomb
    )
from dryadic.features.mutations import MuType

from .utils import remove_pheno_dups
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.labels import get_fancy_label
from ..utilities.misc import choose_label_colour
from ..subvariant_test.utils import get_cohort_label
from ..utilities.colour_maps import simil_cmap
from ..utilities.label_placement import place_scatterpie_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

import random
from functools import reduce
from operator import itemgetter, add
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


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'similarities')

SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}
cna_mtypes = {'Gain': gains_mtype, 'Loss': dels_mtype}


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
        if (auc_vals[mcomb] > 0.6
            and pnt_mtype.is_supertype(
                tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1]))
        ]]

    plt_gby = pnt_aucs.groupby(lambda mtype: tuple(mtype.label_iter())[0])
    clr_dict = {gene: choose_label_colour(gene)
                for gene in plt_gby.groups.keys()}
    lbl_pos = {gene: None for gene in plt_gby.groups.keys()}

    ylim = 1.03
    train_samps = cdata.get_train_samples()

    # TODO: differentiate between genes without CNAs and those
    #  with too much overlap between CNAs and point mutations?
    auc_list: pd.Series
    for cur_gene, auc_list in plt_gby:
        gene_cna = MuType({('Gene', cur_gene): cna_mtype})
        plt_types = {mcomb for mcomb in use_combs
                     if tuple(mcomb.mtypes)[0] == gene_cna}

        if len(plt_types) > 1:
            raise ValueError("Too many exclusive {} CNAs associated with "
                             "`{}`!".format(cna_lbl, cur_gene))

        if len(plt_types) == 1:
            plt_type = tuple(plt_types)[0]
            use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]

            all_mtype = MuType({('Gene', cur_gene): use_mtree.allkey()})
            all_phn = np.array(cdata.train_pheno(all_mtype))
            use_preds = pred_df.loc[set(auc_list.index) | {plt_type},
                                    train_samps].applymap(np.mean)

            for mcomb, auc_val in auc_list.iteritems():
                copy_siml1 = SIML_FXS[siml_metric](
                    use_preds.loc[mcomb][~all_phn],
                    use_preds.loc[mcomb][pheno_dict[mcomb]],
                    use_preds.loc[mcomb][pheno_dict[plt_type]]
                    )

                copy_siml2 = SIML_FXS[siml_metric](
                    use_preds.loc[plt_type][~all_phn],
                    use_preds.loc[plt_type][pheno_dict[plt_type]],
                    use_preds.loc[plt_type][pheno_dict[mcomb]]
                    )

                ylim = max(ylim,
                           abs(copy_siml1) + 0.13, abs(copy_siml2) + 0.13)

                for ax, siml_val in zip(axarr, [copy_siml1, copy_siml2]):
                    ax.scatter(auc_val, siml_val,
                               s=np.mean(pheno_dict[mcomb]) * 601,
                               c=[clr_dict[cur_gene]],
                               alpha=0.29, edgecolor='none')

        else:
            del(lbl_pos[cur_gene])

    auc_clip = sorted(
        [(gene, sorted(np.clip(np.clip(auc_list.values,
                                       *auc_list.quantile(q=[0.13, 0.87])),
                               0.59, 0.98)))
         for gene, auc_list in plt_gby if gene in lbl_pos],
        key=itemgetter(0)
        )

    random.seed(args.seed)
    random.shuffle(auc_clip)
    i = -1
    while i < 1000 and any(pos is None for pos in lbl_pos.values()):
        i += 1

        for gene, auc_list in auc_clip:
            if lbl_pos[gene] is None:
                collided = False

                new_x = random.choice(auc_list)
                new_x += random.gauss(0, 0.42 * i / 10000)
                new_x = np.clip(new_x, 0.61, 0.97)

                for oth_gene, oth_pos in lbl_pos.items():
                    if oth_gene != gene and oth_pos is not None:
                        if (abs(oth_pos[0] - new_x)
                                < ((len(gene) + len(oth_gene)) / 193)):
                            collided = True
                            break

                if not collided:
                    lbl_pos[gene] = new_x, 0

    for gene, pos in lbl_pos.items():
        if pos is not None:
            if clr_dict[gene] is None:
                use_clr = '0.87'
            else:
                use_clr = clr_dict[gene]

            axarr[0].text(pos[0], ylim * 1.03, gene,
                          size=17, color=use_clr, alpha=0.67,
                          fontweight='bold', ha='center', va='bottom')

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    for ax in axarr:
        ax.grid(alpha=0.43, linewidth=0.7)

        ax.plot([1, 1], [-ylim, ylim],
                color='black', linewidth=1.7, alpha=0.89)
        ax.plot([0.6, 1], [0, 0],
                color='black', linewidth=1.9, linestyle=':', alpha=0.61)

        for siml_val in [-1, 1, 2]:
            ax.plot([0.6, 1], [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.41)

    axarr[0].set_ylabel("{} Similarity\nto Subgrouping".format(cna_lbl),
                        size=23, weight='bold')
    axarr[1].set_ylabel(
        "Subgrouping Similarity\nto All {} Alterations".format(cna_lbl),
        size=23, weight='bold'
        )

    axarr[1].set_xlabel("Accuracy of Isolated\nSubgrouping Classifier",
                        size=23, weight='bold')

    for ax in axarr:
        ax.set_xlim(0.59, 1.005)
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
    cna_mtypes = {'Gain': gains_mtype, 'Loss': dels_mtype}

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

    plot_dicts = {'Gain': dict(), 'Loss': dict()}
    clr_dict = dict()
    plt_lims = [0.1, 0.9]
    train_samps = cdata.get_train_samples()

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
                if cur_gene not in clr_dict:
                    clr_dict[cur_gene] = choose_label_colour(cur_gene)

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

                ax.scatter(auc_vals[copy_comb], copy_siml,
                           s=dyad_size * 2039, c=[clr_dict[cur_gene]],
                           alpha=0.31, edgecolor='none')

                plt_lims[0] = min(plt_lims[0], copy_siml - 0.11)
                plt_lims[1] = max(plt_lims[1], copy_siml + 0.11)

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    for ax in gain_ax, loss_ax:
        ax.grid(alpha=0.43, linewidth=0.7)

        ax.plot([1, 1], plt_lims, color='black', linewidth=1.1, alpha=0.89)
        ax.plot([0.6, 1], [0, 0],
                color='black', linewidth=1.3, linestyle=':', alpha=0.53)

        for siml_val in [-1, 1, 2]:
            ax.plot([0.6, 1], [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.53)

    gain_ax.set_ylabel("Similarity to\nAll Gain Alterations",
                       size=23, weight='bold')
    loss_ax.set_ylabel("Similarity to\nAll Loss Alterations",
                       size=23, weight='bold')
    loss_ax.set_xlabel("Accuracy of Isolated Classifier",
                       size=23, weight='bold')

    y_rng = plt_lims[1] - plt_lims[0]
    for cna_lbl, ax in zip(['Gain', 'Loss'], [gain_ax, loss_ax]):
        ax.set_xlim(0.59, 1.005)
        ax.set_ylim(*plt_lims)

        lbl_pos = place_scatterpie_labels(plot_dicts[cna_lbl], fig, ax,
                                          font_adj=5 / 7)

        for (pnt_x, pnt_y), pos in lbl_pos.items():
            ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                    plot_dicts[cna_lbl][pnt_x, pnt_y][1][0],
                    size=13, ha=pos[1], va='bottom')
            ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                    plot_dicts[cna_lbl][pnt_x, pnt_y][1][1],
                    size=9, ha=pos[1], va='top')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]

            if abs(x_delta) > 0.013 or abs(y_delta) > y_rng / 23:
                end_x = pos[0][0] + np.sign(x_delta) * 0.013 / 203
                end_y = pos[0][1] + np.heaviside(y_delta, 0) * y_rng / 29

                ln_x, ln_y = (pnt_x - end_x) / 0.41, (pnt_y - end_y) / y_rng
                ln_mag = (ln_x ** 2 + ln_y ** 2) ** 0.5
                ln_cos, ln_sin = ln_x / ln_mag, ln_y / ln_mag

                ax.plot([pnt_x - ln_cos * 0.041
                         * plot_dicts[cna_lbl][pnt_x, pnt_y][0], end_x],
                        [pnt_y - ln_sin * y_rng / 11
                         * plot_dicts[cna_lbl][pnt_x, pnt_y][0], end_y],
                        c=clr_dict[plot_dicts[cna_lbl][pnt_x, pnt_y][1][0]],
                        linewidth=1.7, alpha=0.31)

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

