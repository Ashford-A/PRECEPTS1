
from ..utilities.mutations import (pnt_mtype, copy_mtype, shal_mtype,
                                   dup_mtype, gains_mtype, loss_mtype,
                                   dels_mtype, ExMcomb)
from dryadic.features.mutations import MuType

from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.misc import choose_label_colour
from ..subvariant_test.utils import get_cohort_label
from ..utilities.colour_maps import simil_cmap

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import random
from functools import reduce
from operator import itemgetter, or_, add
from itertools import combinations as combn
from itertools import product

import warnings
from ..utilities.misc import warning_on_one_line
warnings.formatwarning = warning_on_one_line

import numpy as np
import pandas as pd
from math import ceil
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


def plot_copy_adjacencies(pred_df, pheno_dict, auc_vals,
                          cdata, args, add_lgnd=False):
    fig, (gain_ax, loss_ax) = plt.subplots(figsize=(10, 9), nrows=2, ncols=1)

    use_combs = {mcomb for mcomb in auc_vals.index
                 if (isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
                     and not (mcomb.all_mtype & shal_mtype).is_empty())}

    pnt_aucs = auc_vals[[
        mcomb for mcomb in use_combs
        if (auc_vals[mcomb] > 0.6
            and pnt_mtype.is_supertype(
                tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1]))
        ]]

    cna_mtypes = {'Gain': gains_mtype, 'Loss': dels_mtype}
    plt_gby = pnt_aucs.groupby(lambda mtype: tuple(mtype.label_iter())[0])

    clr_dict = {gene: None for gene in plt_gby.groups.keys()}
    plt_lims = [0.1, 0.9]
    train_samps = cdata.get_train_samples()

    auc_list: pd.Series
    for cur_gene, auc_list in plt_gby:
        plt_types = {
            cna_lbl: {mcomb for mcomb in use_combs
                      if (tuple(mcomb.mtypes)[0]
                          == MuType({('Gene', cur_gene): cna_type}))}
            for cna_lbl, cna_type in cna_mtypes.items()
            }

        if plt_types['Gain'] or plt_types['Loss']:
            if clr_dict[cur_gene] is None:
                clr_dict[cur_gene] = choose_label_colour(cur_gene)

            use_preds = pred_df.loc[
                auc_list.index, train_samps].applymap(np.mean)

            use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]
            all_mtype = MuType({(
                'Gene', cur_gene): use_mtree.allkey()})
            all_phn = np.array(cdata.train_pheno(all_mtype))

        for cna_lbl, ax in zip(['Gain', 'Loss'], [gain_ax, loss_ax]):
            # TODO: differentiate between genes without CNAs and those
            #  with too much overlap between CNAs and point mutations?
            if len(plt_types[cna_lbl]) > 1:
                raise ValueError("Too many exclusive {} CNAs associated with "
                                 "`{}`!".format(cna_lbl, cur_gene))

            elif len(plt_types[cna_lbl]) == 1:
                plt_type = tuple(plt_types[cna_lbl])[0]

                for mcomb, auc_val in auc_list.iteritems():
                    if args.siml_metric == 'mean':
                        copy_siml = calculate_mean_siml(
                            use_preds.loc[mcomb][~all_phn],
                            use_preds.loc[mcomb][pheno_dict[mcomb]],
                            use_preds.loc[mcomb][pheno_dict[plt_type]]
                            )

                    elif args.siml_metric == 'ks':
                        copy_siml = calculate_ks_siml(
                            use_preds.loc[mcomb][~all_phn],
                            use_preds.loc[mcomb][pheno_dict[mcomb]],
                            use_preds.loc[mcomb][pheno_dict[plt_type]]
                            )

                    plt_lims[0] = min(plt_lims[0], copy_siml - 0.11)
                    plt_lims[1] = max(plt_lims[1], copy_siml + 0.11)

                    ax.scatter(auc_val, copy_siml,
                               s=np.mean(pheno_dict[mcomb]) * 1903,
                               c=[clr_dict[cur_gene]],
                               alpha=0.31, edgecolor='none')

    auc_clip = sorted([(gene,
                        sorted(np.clip(auc_list.values,
                                       *auc_list.quantile(q=[0.15, 0.85]))))
                       for gene, auc_list in plt_gby], key=itemgetter(0))

    random.seed(args.seed)
    random.shuffle(auc_clip)
    lbl_pos = {gene: None for gene in plt_gby.groups.keys()}
    i = -1

    while i < 1000 and any(pos is None for pos in lbl_pos.values()):
        i += 1

        for gene, auc_list in auc_clip:
            if lbl_pos[gene] is None:
                collided = False
                new_x = random.choice(auc_list)
                new_x += random.gauss(0, 0.42 * i / 10000)

                for oth_gene, oth_pos in lbl_pos.items():
                    if oth_gene != gene and oth_pos is not None:
                        if (abs(oth_pos[0] - new_x)
                                < ((len(gene) + len(oth_gene)) / 193)):
                            collided = True
                            break

                if not collided:
                    lbl_pos[gene] = new_x, 0

    plt_rng = (plt_lims[1] - plt_lims[0]) / 67
    for gene, pos in lbl_pos.items():
        if pos is not None:
            if clr_dict[gene] is None:
                use_clr = '0.87'
            else:
                use_clr = clr_dict[gene]

            gain_ax.text(pos[0], plt_lims[1] + plt_rng, gene,
                         size=17, color=use_clr, alpha=0.67,
                         fontweight='bold', ha='center', va='bottom')

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    for ax in gain_ax, loss_ax:
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.1, alpha=0.89)
        ax.plot([0.6, 1], [0, 0],
                color='black', linewidth=1.3, linestyle=':', alpha=0.53)

        for siml_val in [-1, 1, 2]:
            ax.plot([0.6, 1], [siml_val] * 2,
                    color=simil_cmap(clr_norm(siml_val)),
                    linewidth=4.1, linestyle=':', alpha=0.53)

    gain_ax.set_ylabel("Similarity to\nAll Gain Alterations",
                       size=23, weight='semibold')
    loss_ax.set_ylabel("Similarity to\nAll Loss Alterations",
                       size=23, weight='semibold')
    loss_ax.set_xlabel("Accuracy of Isolated Classifier",
                       size=23, weight='semibold')

    for ax in gain_ax, loss_ax:
        ax.set_xlim(0.59, 1.005)
        ax.set_ylim(*plt_lims)

    plt.tight_layout(pad=0, h_pad=1.9)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}-copy-adjacencies_{}.svg".format(
                         args.siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_score_symmetry(pred_dfs, pheno_dict, auc_dfs, cdata, args):
    assert sorted(auc_dfs['Iso'].index) == sorted(auc_dfs['IsoShal'].index)
    fig, (iso_ax, ish_ax) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2)

    iso_combs = {mut for mut, auc_val in auc_dfs['Iso']['mean'].iteritems()
                 if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                     and not (mut.all_mtype & shal_mtype).is_empty())}

    ish_combs = {
        mut for mut, auc_val in auc_dfs['IsoShal']['mean'].iteritems()
        if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
            and (mut.all_mtype & shal_mtype).is_empty()
            and all((mtp & shal_mtype).is_empty() for mtp in mut.mtypes))
        }

    train_samps = cdata.get_train_samples()
    clr_dict = dict()
    plt_lims = [0.1, 0.9]

    for ax, ex_lbl, use_combs in zip([iso_ax, ish_ax], ['Iso', 'IsoShal'],
                                     [tuple(iso_combs), tuple(ish_combs)]):
        for cur_gene, gene_combs in pd.Series(use_combs).reindex(
                use_combs).groupby(
                    lambda mcomb: tuple(mcomb.label_iter())[0]):

            use_pairs = [
                {mcomb1, mcomb2}
                for mcomb1, mcomb2 in combn(gene_combs.index, 2)
                if (all((mtp1 & mtp2).is_empty()
                        for mtp1, mtp2 in product(
                            mcomb1.mtypes, mcomb2.mtypes))
                    or not (pheno_dict[mcomb1] & pheno_dict[mcomb2]).any())
                ]

            if len(use_pairs) > 0:
                pair_combs = reduce(or_, use_pairs)
                use_preds = pred_dfs[ex_lbl].loc[
                    pair_combs, train_samps].applymap(np.mean)

                if args.verbose:
                    print('\n'.join([
                        '\n##########',
                        "{}({})  {} pairs from {} types".format(
                            cur_gene, ex_lbl,
                            len(use_pairs), len(pair_combs)
                            ),
                        '----------'
                        ] + ['\txxxxx\t'.join([str(mcomb) for mcomb in pair])
                             for pair in tuple(use_pairs)[
                                 ::(len(use_pairs)
                                    // (args.verbose * 3) + 1)
                                ]]
                        ))

                if cur_gene not in clr_dict:
                    clr_dict[cur_gene] = choose_label_colour(cur_gene)

                use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]
                all_mtype = MuType({('Gene', cur_gene): use_mtree.allkey()})
                if ex_lbl == 'IsoShal':
                    all_mtype -= MuType({('Gene', cur_gene): shal_mtype})

                all_phn = np.array(cdata.train_pheno(all_mtype))
                wt_vals = {mcomb: use_preds.loc[mcomb][~all_phn]
                           for mcomb in pair_combs}
                mut_vals = {mcomb: use_preds.loc[mcomb][pheno_dict[mcomb]]
                            for mcomb in pair_combs}

                if args.siml_metric == 'mean':
                    wt_means = {mcomb: vals.mean()
                                for mcomb, vals in wt_vals.items()}
                    mut_means = {mcomb: vals.mean()
                                 for mcomb, vals in mut_vals.items()}

                elif args.siml_metric == 'ks':
                    base_dists = {
                        mcomb: ks_2samp(wt_vals[mcomb], mut_vals[mcomb],
                                        alternative='greater').statistic
                        for mcomb in pair_combs
                        }

                for mcomb1, mcomb2 in use_pairs:
                    other_vals1 = use_preds.loc[mcomb1][pheno_dict[mcomb2]]
                    other_vals2 = use_preds.loc[mcomb2][pheno_dict[mcomb1]]

                    if args.siml_metric == 'mean':
                        pair_siml1 = calculate_mean_siml(
                            wt_vals[mcomb1], mut_vals[mcomb1], other_vals1,
                            wt_mean=wt_means[mcomb1],
                            mut_mean=mut_means[mcomb1],
                            )

                        pair_siml2 = calculate_mean_siml(
                            wt_vals[mcomb2], mut_vals[mcomb2], other_vals2,
                            wt_mean=wt_means[mcomb2],
                            mut_mean=mut_means[mcomb2],
                            )

                    elif args.siml_metric == 'ks':
                        pair_siml1 = calculate_ks_siml(
                            wt_vals[mcomb1], mut_vals[mcomb1], other_vals1,
                            base_dist=base_dists[mcomb1]
                            )

                        pair_siml2 = calculate_ks_siml(
                            wt_vals[mcomb2], mut_vals[mcomb2], other_vals2,
                            base_dist=base_dists[mcomb2]
                            )

                    plt_lims[0] = min(plt_lims[0],
                                      pair_siml1 - 0.19, pair_siml2 - 0.19)
                    plt_lims[1] = max(plt_lims[1],
                                      pair_siml1 + 0.19, pair_siml2 + 0.19)
                    plt_sz = 293 * (np.mean(pheno_dict[mcomb1])
                                    * np.mean(pheno_dict[mcomb2])) ** 0.5

                    ax.scatter(pair_siml1, pair_siml2,
                               s=plt_sz, c=[clr_dict[cur_gene]], alpha=0.13,
                               edgecolor='none')

    clr_norm = colors.Normalize(vmin=-1, vmax=2)
    for ax in iso_ax, ish_ax:
        ax.grid(alpha=0.53, linewidth=0.9)

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

        ax.set_xlim(*plt_lims)
        ax.set_ylim(*plt_lims)

    iso_ax.set_title(
        "Similarities Computed Treating\nShallow CNAs as Mutant\n",
        size=23, weight='semibold'
        )
    ish_ax.set_title(
        "Similarities Computed Treating\nShallow CNAs as Wild-Type\n",
        size=23, weight='semibold'
        )

    plt.tight_layout(w_pad=3.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}-siml-symmetry_{}.svg".format(
                         args.siml_metric, args.classif)),
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
    parser.add_argument('--siml_metric', '-s',
                        default='ks', choices={'mean', 'ks'})
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(Path(out_dir).glob(
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
        plot_copy_adjacencies(
            pred_dfs['Iso'], phn_dict, auc_dfs['Iso']['mean'], cdata, args)

    else:
        warnings.warn("Cannot analyze the similarities between CNAs and "
                      "point mutation types until this experiment has been "
                      "run with the `Conseqeuence__Exon` mutation level "
                      "combination on this cohort!")
 
    plot_score_symmetry(pred_dfs, phn_dict, auc_dfs, cdata, args)


if __name__ == '__main__':
    main()

