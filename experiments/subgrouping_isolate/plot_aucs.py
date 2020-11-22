
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, shal_mtype,
    dup_mtype, gains_mtype, loss_mtype, dels_mtype, Mcomb, ExMcomb
    )
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir
from ..utilities.metrics import calc_conf
from ..utilities.labels import get_fancy_label
from ..utilities.label_placement import place_scatter_labels
from ..subgrouping_test.plot_aucs import add_scatterpie_legend
from ..utilities.labels import get_cohort_label
from ..utilities.misc import choose_label_colour

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from itertools import permutations, product

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')


def plot_sub_comparisons(auc_vals, pheno_dict, conf_vals,
                         args, add_lgnd=False):
    fig, ax = plt.subplots(figsize=(11, 11))

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.57

    use_aucs = auc_vals[[
        not isinstance(mtype, (Mcomb, ExMcomb))
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
            line_dict[auc_tupl] = dict(c=choose_label_colour(gene))
            base_size = np.mean(pheno_dict[base_mtype])
            plt_size = 0.07 * base_size ** 0.5

            plot_dict[auc_tupl] = [plt_size, ('', '')]
            plt_min = min(plt_min, auc_tupl[0] - 0.03, auc_tupl[1] - 0.029)
            best_prop = np.mean(pheno_dict[best_subtype]) / base_size
            conf_sc = calc_conf(conf_vals[best_subtype],
                                conf_vals[base_mtype])

            if auc_tupl[1] > 0.7 and conf_sc > 0.8:
                plot_dict[auc_tupl][1] = gene, get_fancy_label(
                    tuple(best_subtype.subtype_iter())[0][1],
                    pnt_link='\nor ', phrase_link=' '
                    )

            # ...if we are not sure but the respective AUCs are still
            # pretty great then add a label with just the gene name...
            elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                plot_dict[auc_tupl][1] = gene, ''

            auc_bbox = (auc_tupl[0] - plt_size / 2,
                        auc_tupl[1] - plt_size / 2, plt_size, plt_size)

            pie_ax = inset_axes(ax, width='100%', height='100%',
                                bbox_to_anchor=auc_bbox,
                                bbox_transform=ax.transData,
                                axes_kwargs=dict(aspect='equal'), borderpad=0)

            pie_ax.pie(x=[best_prop, 1 - best_prop],
                       colors=[line_dict[auc_tupl]['c'] + (0.77,),
                               line_dict[auc_tupl]['c'] + (0.29,)],
                       explode=[0.29, 0], startangle=90)

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

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=23, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=23, weight='semibold')

    if add_lgnd:
        ax, plot_dict = add_scatterpie_legend(ax, plot_dict, plt_min, args)

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[[plt_min + 0.01, 0.99]] * 2,
                                       line_dict=line_dict)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_copy_comparisons(auc_vals, pheno_dict, conf_vals,
                          args, add_lgnd=False):
    fig, (gain_ax, loss_ax) = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)

    use_aucs = auc_vals[[not isinstance(mtype, (Mcomb, ExMcomb))
                         for mtype in auc_vals.index]]
    subt_dict = {mtype: tuple(mtype.subtype_iter())[0][1]
                 for mtype in use_aucs.index}

    plot_dicts = {gains_mtype: dict(), dels_mtype: dict()}
    line_dicts = {gains_mtype: dict(), dels_mtype: dict()}
    plt_min = 0.57

    for ax, copy_type in zip([gain_ax, loss_ax], [gains_mtype, dels_mtype]):
        copy_aucs = use_aucs[[
            not (subt_dict[mtype] & pnt_mtype).is_empty()
            and not (subt_dict[mtype] & copy_type).is_empty()
            for mtype in use_aucs.index
            ]]

        for gene, auc_vec in copy_aucs.groupby(
                lambda mtype: tuple(mtype.label_iter())[0]):
            if len(auc_vec) > 1:
                base_mtype = MuType({('Gene', gene): copy_type | pnt_mtype})

                base_indx = auc_vec.index.get_loc(base_mtype)
                best_subtype = auc_vec[:base_indx].append(
                    auc_vec[(base_indx + 1):]).idxmax()

                auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
                use_clr = choose_label_colour(gene)
                line_dicts[copy_type][auc_tupl] = dict(c=use_clr)

                base_size = np.mean(pheno_dict[base_mtype])
                plt_size = 0.07 * base_size ** 0.5
                plot_dicts[copy_type][auc_tupl] = [plt_size, ('', '')]
                plt_min = min(plt_min, auc_tupl[0] - 0.03, auc_tupl[1] - 0.03)

                best_prop = np.mean(pheno_dict[best_subtype]) / base_size
                conf_sc = calc_conf(conf_vals[best_subtype],
                                    conf_vals[base_mtype])

                # ...and if we are sure that the optimal subgrouping AUC is
                # better than the point mutation AUC then add a label with the
                # gene name and a description of the best found subgrouping...
                if auc_tupl[1] > 0.7 and conf_sc > 0.8:
                    plot_dicts[copy_type][auc_tupl][1] = (
                        gene, get_fancy_label(
                            tuple(best_subtype.subtype_iter())[0][1],
                            pnt_link='\nor ', phrase_link=' '
                            )
                        )

                # ...if we are not sure but the respective AUCs are still
                # pretty great then add a label with just the gene name...
                elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                    plot_dicts[copy_type][auc_tupl][1] = gene, ''

                auc_bbox = (auc_tupl[0] - plt_size / 2,
                            auc_tupl[1] - plt_size / 2, plt_size, plt_size)

                pie_ax = inset_axes(ax, width='100%', height='100%',
                                    bbox_to_anchor=auc_bbox,
                                    bbox_transform=ax.transData,
                                    axes_kwargs=dict(aspect='equal'),
                                    borderpad=0)

                pie_ax.pie(x=[best_prop, 1 - best_prop],
                           colors=[use_clr + (0.77, ), use_clr + (0.29, )],
                           explode=[0.29, 0], startangle=90)

    gain_ax.set_xlabel("Accuracy of\n(All Gains + Gene-Wide) Classifier",
                       size=21, weight='semibold')
    gain_ax.set_ylabel("Accuracy of Best\n(Gains + Subgrouping) Classifier",
                       size=21, weight='semibold')

    loss_ax.set_xlabel("Accuracy of\n(All Losses + Gene-Wide) Classifier",
                       size=21, weight='semibold')
    loss_ax.set_ylabel(
        "\n\nAccuracy of Best\n(Losses + Subgrouping) Classifier",
        size=21, weight='semibold'
        )

    plt_lims = plt_min, 1 + (1 - plt_min) / 103
    lbl_lims = [[plt_min + 0.01, 0.99]] * 2

    for ax, copy_type in zip([gain_ax, loss_ax], [gains_mtype, dels_mtype]):
        ax.grid(linewidth=0.83, alpha=0.41)

        ax.plot(plt_lims, [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims,
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

        if plot_dicts[copy_type]:
            lbl_pos = place_scatter_labels(plot_dicts[copy_type], ax,
                                           plt_lims=lbl_lims, font_size=11,
                                           line_dict=line_dicts[copy_type])

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "copy-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_comparisons(auc_dfs, pheno_dict, args):
    fig, axarr = plt.subplots(figsize=(15, 15), nrows=3, ncols=3)

    base_aucs = {
        ex_lbl: auc_df.loc[[
            not isinstance(mtype, (Mcomb, ExMcomb))
            and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
            for mtype in auc_df.index
            ], 'mean']
        for ex_lbl, auc_df in auc_dfs.items()
        }

    base_mtypes = {tuple(sorted(auc_vals.index))
                   for auc_vals in base_aucs.values()}
    assert len(base_mtypes) == 1, ("Mismatching mutation types across "
                                   "isolation testing holdout modes!")

    base_mtypes = tuple(base_mtypes)[0]
    iso_aucs = {'All': base_aucs['All']}

    iso_aucs['Iso'] = auc_dfs['Iso'].loc[[
        isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
        and tuple(mcomb.mtypes)[0] in base_mtypes
        and not (mcomb.all_mtype & shal_mtype).is_empty()
        for mcomb in auc_dfs['Iso'].index
        ], 'mean']

    iso_aucs['IsoShal'] = auc_dfs['IsoShal'].loc[[
        isinstance(mcomb, ExMcomb) and len(mcomb.mtypes) == 1
        and tuple(mcomb.mtypes)[0] in base_mtypes
        and (mcomb.all_mtype & shal_mtype).is_empty()
        for mcomb in auc_dfs['IsoShal'].index
        ], 'mean']

    assert not set(iso_aucs['Iso'].index & iso_aucs['IsoShal'].index)
    for ex_lbl in ('Iso', 'IsoShal'):
        iso_aucs[ex_lbl].index = [tuple(mcomb.mtypes)[0]
                                  for mcomb in iso_aucs[ex_lbl].index]

    clr_dict = dict()
    plt_min = 0.57

    for (i, ex_lbl1), (j, ex_lbl2) in combn(enumerate(base_aucs.keys()), 2):
        for mtype, auc_val1 in base_aucs[ex_lbl1].iteritems():
            plt_min = min(plt_min, auc_val1 - 0.013,
                          base_aucs[ex_lbl2][mtype] - 0.013)
            mtype_sz = 301 * np.mean(pheno_dict[mtype])

            cur_gene = tuple(mtype.label_iter())[0]
            if cur_gene not in clr_dict:
                clr_dict[cur_gene] = choose_label_colour(cur_gene)

            axarr[i, j].scatter(base_aucs[ex_lbl2][mtype], auc_val1,
                                c=[clr_dict[cur_gene]], s=mtype_sz,
                                alpha=0.23, edgecolor='none')

        for mtype in set(iso_aucs[ex_lbl1].index & iso_aucs[ex_lbl2].index):
            plt_x = iso_aucs[ex_lbl1][mtype]
            plt_y = iso_aucs[ex_lbl2][mtype]

            plt_min = min(plt_min, plt_x - 0.013, plt_y - 0.013)
            mtype_sz = 301 * np.mean(pheno_dict[mtype])

            cur_gene = tuple(mtype.label_iter())[0]
            if cur_gene not in clr_dict:
                clr_dict[cur_gene] = choose_label_colour(cur_gene)

            axarr[j, i].scatter(plt_x, plt_y, c=[clr_dict[cur_gene]],
                                s=mtype_sz, alpha=0.23, edgecolor='none')

    for i, j in permutations(range(3), r=2):
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
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "iso-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_dyad_comparisons(auc_vals, pheno_dict, conf_vals, args):
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
        ).transpose().set_index(
            [[gains_mtype, dup_mtype, dels_mtype, loss_mtype]],
            append=True
            ).transpose()

    for pnt_type, copy_indx in product(pnt_aucs.index, plot_df.columns):
        dyad_type = MuType({
            ('Gene', tuple(pnt_type.label_iter())[0]): copy_indx[-1]})
        dyad_type |= pnt_type

        if dyad_type in auc_vals.index:
            plot_df.loc[pnt_type, copy_indx] = auc_vals[dyad_type]

    plot_dicts = {'gain': dict(), 'loss': dict()}
    font_dicts = {'gain': dict(), 'loss': dict()}
    for ax, copy_lbl in zip([gain_ax, loss_ax], ['gain', 'loss']):
        for (dpth_lbl, copy_type), plot_aucs in plot_df[copy_lbl].iteritems():
            copy_aucs = plot_aucs[~plot_aucs.isnull()]

            for gene, auc_vec in copy_aucs.groupby(
                    lambda mtype: tuple(mtype.label_iter())[0]):
                gene_clr = choose_label_colour(gene)

                auc_tupl = pnt_aucs[auc_vec.index].mean(), auc_vec.mean()
                if ((len(auc_vec) / plot_df.shape[0]) >= 0.01
                        and auc_vec.max() >= 0.7):
                    font_dicts[copy_lbl][auc_tupl] = dict(c=gene_clr)

                    plot_dicts[copy_lbl][auc_tupl] = (
                        0, (gene,
                            "(or {})".format(get_fancy_label(copy_type)))
                        )

                for pnt_type, copy_auc in auc_vec.iteritems():
                    base_size = np.mean(pheno_dict[pnt_type])
                    pnt_size = 1.41 * base_size ** 0.5

                    if dpth_lbl == 'all':
                        dpth_clr = gene_clr
                        edg_lw = 0
                    else:
                        dpth_clr = 'none'
                        edg_lw = 3.41 * base_size ** 0.29

                    plot_dicts[copy_lbl][pnt_aucs[pnt_type], copy_auc] = (
                        pnt_size, ('', ''))

                    ax.scatter(pnt_aucs[pnt_type], copy_auc,
                               facecolor=dpth_clr, edgecolor=gene_clr,
                               s=581 * base_size, alpha=0.21,
                               linewidths=edg_lw)

    plt_min = min(plot_df.min().min() - 0.02, pnt_aucs.min() - 0.02)
    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    lbl_lims = [[plt_min + 0.05, 0.93]] * 2

    for ax, copy_lbl in zip([gain_ax, loss_ax], ['gain', 'loss']):
        ax.grid(linewidth=0.71, alpha=0.41)

        ax.plot(plt_lims, [0.5, 0.5],
                color='black', linewidth=0.9, linestyle=':', alpha=0.71)
        ax.plot([0.5, 0.5], plt_lims,
                color='black', linewidth=0.9, linestyle=':', alpha=0.71)

        ax.plot(plt_lims, [1, 1], color='black', linewidth=1.1, alpha=0.89)
        ax.plot([1, 1], plt_lims, color='black', linewidth=1.1, alpha=0.89)
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=1.7, linestyle='--', alpha=0.41)

        ax.set_xlabel("Accuracy of Subgrouping Classifier",
                      size=22, weight='semibold')
        ax.set_title("{} CNAs".format(copy_lbl.capitalize()),
                     size=26, weight='semibold')

        if plot_dicts[copy_lbl]:
            lbl_pos = place_scatter_labels(plot_dicts[copy_lbl], ax,
                                           plt_lims=[plt_lims] * 2,
                                           plc_lims=lbl_lims,
                                           plt_type='scatter', font_size=11,
                                           line_dict=font_dicts[copy_lbl],
                                           font_dict=font_dicts[copy_lbl],
                                           linewidth=1.13, alpha=0.31)

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    gain_ax.set_ylabel("Accuracy of\n(Subgrouping or CNAs) Classifier",
                       size=23, weight='semibold')

    plt.tight_layout(w_pad=3.1)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "dyad-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(out_dir.glob(
        "out-conf__*__*__{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    if 'Consequence__Exon' not in set(out_use.Levels):
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort])), exist_ok=True)

    out_iter = out_use.groupby('Levels')['File']
    out_aucs = {lvls: list() for lvls, _ in out_iter}
    out_confs = {lvls: list() for lvls, _ in out_iter}
    phn_dict = dict()

    auc_dfs = {ex_lbl: pd.DataFrame([])
               for ex_lbl in ['All', 'Iso', 'IsoShal']}
    conf_dfs = {ex_lbl: pd.DataFrame([])
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

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-conf", out_tag])),
                             'r') as f:
                out_confs[lvls] += [pickle.load(f)]

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[ex_lbl] = pd.concat([
                    auc_dfs[ex_lbl], out_aucs[lvls][super_indx][ex_lbl]])
                conf_dfs[ex_lbl] = pd.concat([
                    conf_dfs[ex_lbl], out_confs[lvls][super_indx][ex_lbl]])

    auc_dfs = {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
               for ex_lbl, auc_df in auc_dfs.items()}
    conf_dfs = {ex_lbl: conf_df.loc[~conf_df.index.duplicated()]
                for ex_lbl, conf_df in conf_dfs.items()}

    plot_sub_comparisons(auc_dfs['All']['mean'], phn_dict,
                         conf_dfs['All']['mean'], args)
    plot_copy_comparisons(auc_dfs['All']['mean'], phn_dict,
                          conf_dfs['All']['mean'], args)

    plot_iso_comparisons(auc_dfs, phn_dict, args)
    plot_dyad_comparisons(auc_dfs['All']['mean'], phn_dict,
                          conf_dfs['All']['mean'], args)


if __name__ == '__main__':
    main()

