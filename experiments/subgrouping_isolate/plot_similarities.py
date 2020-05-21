
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'similarities')

from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_isolate import cna_mtypes
from HetMan.experiments.utilities.mutations import ExMcomb
from dryadic.features.mutations import MuType

from HetMan.experiments.subgrouping_isolate.utils import calculate_pair_siml
from HetMan.experiments.utilities.misc import choose_label_colour
from HetMan.experiments.subvariant_test.utils import get_cohort_label
from HetMan.experiments.utilities.colour_maps import simil_cmap

import argparse
from pathlib import Path
import bz2
import dill as pickle

import random
from operator import itemgetter
from itertools import combinations as combn

import warnings
from HetMan.experiments.utilities.misc import warning_on_one_line
warnings.formatwarning = warning_on_one_line

import numpy as np
import pandas as pd
from math import ceil

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_copy_adjacencies(siml_dicts, pheno_dict, auc_vals, pred_vals,
                          cdata, args, add_lgnd=False):
    fig, (gain_ax, loss_ax) = plt.subplots(figsize=(10, 9), nrows=2, ncols=1)

    use_combs = {mcomb for mcomb in auc_vals.index
                 if (isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
                     and not (mcomb.all_mtype
                              & dict(cna_mtypes)['Shal']).is_empty())}

    pnt_aucs = auc_vals[[
        mcomb for mcomb in use_combs
        if (auc_vals[mcomb] > 0.6
            and pnt_mtype.is_supertype(
                tuple(mcomb.mtypes)[0].subtype_list()[0][1]))
        ]]

    plt_gby = pnt_aucs.groupby(lambda mtype: mtype.get_labels()[0])
    clr_dict = {gene: None for gene in plt_gby.groups.keys()}
    lbl_pos = {gene: None for gene in plt_gby.groups.keys()}
    plt_lims = [0.1, 0.9]

    if args.test:
        test_list = list()
    else:
        test_list = None

    for cur_gene, auc_list in plt_gby:
        for mcomb, auc_val in auc_list.iteritems():
            plt_types = {
                cna_lbl: {mcomb for mcomb in use_combs
                          if (tuple(mcomb.mtypes)[0]
                              == MuType({('Gene', cur_gene): cna_type}))}
                for cna_lbl, cna_type in cna_mtypes
                }

            for cna_lbl, ax in zip(['Gain', 'Loss'], [gain_ax, loss_ax]):
                #TODO: differentiate between genes without CNAs and those with
                # too much overlap between CNAs and point mutations?
                if len(plt_types[cna_lbl]) > 1:
                    raise ValueError("Too many exclusive {} CNAs matching "
                                     "`{}`!".format(cna_lbl, mcomb))

                elif len(plt_types[cna_lbl]) == 1:
                    plt_type = tuple(plt_types[cna_lbl])[0]
                    if clr_dict[cur_gene] is None:
                        clr_dict[cur_gene] = choose_label_colour(cur_gene)

                    use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]
                    all_mtype = MuType({(
                        'Gene', cur_gene): use_mtree.allkey()})

                    if args.test:
                        copy_siml, test_list = calculate_pair_siml(
                            mcomb, plt_type, all_mtype, siml_dicts, pheno_dict,
                            pred_vals, 'Iso', cdata, test_list
                            )

                    else:
                        copy_siml = calculate_pair_siml(
                            mcomb, plt_type, all_mtype, siml_dicts, pheno_dict,
                            pred_vals, 'Iso', cdata, test_list
                            )

                    plt_lims[0] = min(plt_lims[0], copy_siml - 0.11)
                    plt_lims[1] = max(plt_lims[1], copy_siml + 0.11)

                    ax.scatter(auc_val, copy_siml,
                               s=np.mean(pheno_dict[mcomb]) * 2307,
                               c=[clr_dict[cur_gene]],
                               alpha=0.43, edgecolor='none')

    if args.test:
        print("Successfully tested the copy-similarities of {} "
              "mutation types from {} different genes for internal "
              "consistency!".format(len(test_list),
                                    len(set(mcomb.get_labels()[0]
                                            for mcomb, _ in test_list))))

    auc_clip = sorted([(gene,
                        sorted(np.clip(auc_list.values,
                                       *auc_list.quantile(q=[0.15, 0.85]))))
                       for gene, auc_list in plt_gby], key=itemgetter(0))

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
                         size=17, color=use_clr, fontweight='bold',
                         ha='center', va='bottom')

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
                     "copy-adjacencies_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_score_symmetry(siml_dicts, pheno_dict, auc_dfs, pred_dfs,
                        cdata, args):
    fig, (iso_ax, ish_ax) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2)

    assert sorted(auc_dfs['Iso'].index) == sorted(auc_dfs['IsoShal'].index)
    plt_gby = auc_dfs['Iso']['mean'][[
        mcomb for mcomb in auc_dfs['Iso'].index
        if isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
        ]].groupby(lambda mcomb: mcomb.get_labels()[0])

    if args.test:
        test_list = list()
    else:
        test_list = None

    plt_lims = [0.1, 0.9]
    for cur_gene, iso_aucs in plt_gby:
        gene_clr = choose_label_colour(cur_gene)

        use_mtree = tuple(cdata.mtrees.values())[0][cur_gene]
        all_mtypes = {'Iso': MuType({('Gene', cur_gene): use_mtree.allkey()})}
        all_mtypes['IsoShal'] = all_mtypes['Iso'] - MuType({
            ('Gene', cur_gene): dict(cna_mtypes)['Shal']})

        iso_combs = {mcomb for mcomb, auc_val in iso_aucs.iteritems()
                     if (auc_val >= 0.7
                         and not (mcomb.all_mtype
                                  & dict(cna_mtypes)['Shal']).is_empty())}

        ish_combs = {
            mcomb for mcomb, auc_val in auc_dfs['IsoShal']['mean'].iteritems()
            if (mcomb in iso_aucs.index and auc_val >= 0.7
                and (tuple(mcomb.mtypes)[0]
                     & dict(cna_mtypes)['Shal']).is_empty()
                and (mcomb.all_mtype & dict(cna_mtypes)['Shal']).is_empty())
            }

        for ax, ex_lbl, use_combs in zip([iso_ax, ish_ax], ['Iso', 'IsoShal'],
                                         [iso_combs, ish_combs]):
            use_pairs = {
                (mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
                if (not (pheno_dict[mcomb1] & pheno_dict[mcomb2]).any()
                    or (tuple(mcomb1.mtypes)[0]
                        & tuple(mcomb2.mtypes)[0]).is_empty())
                }

            if args.verbose and use_pairs:
                print('\n'.join([
                    '\n##########', "{}({})  {} pairs from {} types".format(
                        cur_gene, ex_lbl, len(use_pairs), len(use_combs)),
                    '----------'
                    ] + ['\txxxxx\t'.join([str(mcomb) for mcomb in pair])
                         for pair in tuple(use_pairs)[
                             ::(len(use_pairs) // (args.verbose * 3) + 1)]]
                    ))

            for mcomb1, mcomb2 in use_pairs:
                if args.test:
                    copy_siml1, test_list = calculate_pair_siml(
                        mcomb1, mcomb2, all_mtypes[ex_lbl],
                        siml_dicts[ex_lbl], pheno_dict, pred_dfs[ex_lbl],
                        ex_lbl, cdata, test_list
                        )

                    copy_siml2, test_list = calculate_pair_siml(
                        mcomb2, mcomb1, all_mtypes[ex_lbl],
                        siml_dicts[ex_lbl], pheno_dict, pred_dfs[ex_lbl],
                        ex_lbl, cdata, test_list
                        )

                else:
                    copy_siml1 = calculate_pair_siml(
                        mcomb1, mcomb2, all_mtypes[ex_lbl],
                        siml_dicts[ex_lbl], pheno_dict, pred_dfs[ex_lbl],
                        ex_lbl, cdata, test_list
                        )

                    copy_siml2 = calculate_pair_siml(
                        mcomb2, mcomb1, all_mtypes[ex_lbl],
                        siml_dicts[ex_lbl], pheno_dict, pred_dfs[ex_lbl],
                        ex_lbl, cdata, test_list
                        )

                plt_lims[0] = min(plt_lims[0],
                                  copy_siml1 - 0.19, copy_siml2 - 0.19)
                plt_lims[1] = max(plt_lims[1],
                                  copy_siml1 + 0.19, copy_siml2 + 0.19)
                plt_sz = 377 * (np.mean(pheno_dict[mcomb1])
                                * np.mean(pheno_dict[mcomb2])) ** 0.5

                ax.scatter(copy_siml1, copy_siml2, s=plt_sz, c=[gene_clr],
                           alpha=0.13, edgecolor='none')

    if args.test:
        print("Successfully tested the inferred similarities of {} mutation "
              "type pairs which included {} different genes for internal "
              "consistency!".format(len(test_list),
                                    len(set(mcomb.get_labels()[0]
                                            for mcomb, _ in test_list))))

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
                     "score-symmetry_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the similarities between various pairs of genes' "
        "subgroupings with a cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--seed', type=int)
    parser.add_argument('--test', action='store_true',
                        help="run diagnostic tests?")
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(Path(out_dir).glob(
        "out-siml__*__*__{}.p.gz".format(args.classif)))

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
    out_aucs = {lvls: list() for lvls, _ in out_iter}
    out_simls = {lvls: list() for lvls, _ in out_iter}
    out_preds = {lvls: list() for lvls, _ in out_iter}

    phn_dict = dict()
    cdata = None

    auc_dfs = {ex_lbl: pd.DataFrame([])
               for ex_lbl in ['All', 'Iso', 'IsoShal']}
    siml_dicts = {ex_lbl: {lvls: None for lvls, _ in out_iter}
                  for ex_lbl in ['Iso', 'IsoShal']}
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

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-siml", out_tag])),
                             'r') as f:
                out_simls[lvls] += [pickle.load(f)]

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

            for ex_lbl in ['Iso', 'IsoShal']:
                siml_dicts[ex_lbl][lvls] = out_simls[lvls][super_indx][ex_lbl]

    auc_dfs = {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
               for ex_lbl, auc_df in auc_dfs.items()}

    if 'Consequence__Exon' in out_iter.groups.keys():
        plot_copy_adjacencies(siml_dicts['Iso'], phn_dict,
                              auc_dfs['Iso']['mean'], pred_dfs['Iso'],
                              cdata, args)

    else:
        warnings.warn("Cannot analyze the similarities between CNAs and "
                      "point mutation types until this experiment has been "
                      "run with the `Conseqeuence__Exon` mutation level "
                      "combination on this cohort!")

    plot_score_symmetry(siml_dicts, phn_dict, auc_dfs, pred_dfs, cdata, args)


if __name__ == '__main__':
    main()

