
from ..utilities.mutations import (pnt_mtype, shal_mtype, deep_mtype,
                                   copy_mtype, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import (siml_fxs, cna_mtypes, remove_pheno_dups,
                    get_mut_ex, choose_subtype_colour)
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.misc import choose_label_colour
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
from itertools import permutations as permt

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'copy')


def plot_point_similarity(pred_dfs, pheno_dicts, auc_lists,
                          cdata_dict, args, cna_lbl, siml_metric):
    fig, (pnt_ax, cpy_ax) = plt.subplots(figsize=(12, 14), nrows=2)
    cna_mtype = cna_mtypes[cna_lbl]

    copy_dict = dict()
    gn_dict = dict()
    siml_dicts = {k: {(src, coh): dict() for src, coh in auc_lists}
                  for k in ['Pnt', 'Cpy']}
    annt_lists = {(src, coh): set() for src, coh in auc_lists}

    plot_dicts = {'Pnt': dict(), 'Cpy': dict()}
    line_dicts = {'Pnt': dict(), 'Cpy': dict()}
    clr_dict = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined and subsets of point mutations...
    for (src, coh), auc_list in auc_lists.items():
        use_aucs = auc_list[
            list(remove_pheno_dups({
                mut for mut, auc_val in auc_list.iteritems()
                if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                    and get_mut_ex(mut) == args.ex_lbl)
                }, pheno_dicts[src, coh]))
            ]

        if len(use_aucs) == 0:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {tuple(mcomb.label_iter())[0] for mcomb in use_aucs.index}
        train_samps = cdata_dict[src, coh].get_train_samples()
        coh_lbl = get_cohort_label(coh)

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

        for gene, auc_vals in use_aucs.groupby(
                lambda mcomb: tuple(mcomb.label_iter())[0]):
            pnt_comb = {mcomb for mcomb in auc_vals.index
                        if all(pnt_mtype == tuple(mtype.subtype_iter())[0][1]
                               for mtype in mcomb.mtypes)}

            cpy_combs = {
                mcomb for mcomb in auc_vals.index
                if all(
                    cna_mtype.is_supertype(tuple(mtype.subtype_iter())[0][1])
                    for mtype in mcomb.mtypes
                    )
                }

            if args.ex_lbl == 'IsoShal':
                cpy_combs = {
                    mcomb for mcomb in cpy_combs
                    if all(
                        deep_mtype.is_supertype(
                            tuple(mtype.subtype_iter())[0][1])
                        for mtype in mcomb.mtypes
                        )
                    }

            if len(pnt_comb) == 1 or len(cpy_combs) > 0:
                clr_dict[gene] = None

                if args.ex_lbl == 'Iso':
                    ex_all = ExMcomb(MuType({('Gene', gene): copy_mtype}),
                                     MuType({('Gene', gene): pnt_mtype}))

                else:
                    ex_all = ExMcomb(MuType({('Gene', gene): deep_mtype}),
                                     MuType({('Gene', gene): pnt_mtype}))

                if (src, coh, gene) not in copy_dict:
                    copy_dict[src, coh, gene] = np.array(
                        cdata_dict[src, coh].train_pheno(
                            ExMcomb(MuType({('Gene', gene): pnt_mtype}),
                                    MuType({('Gene', gene): cna_mtype}))
                            )
                        )

                    gn_dict[src, coh, gene] = np.array(
                        cdata_dict[src, coh].train_pheno(ex_all))

            if len(pnt_comb) == 1:
                pnt_comb = tuple(pnt_comb)[0]
                assert not (pheno_dicts[src, coh][pnt_comb]
                            & copy_dict[src, coh, gene]).any()

                if copy_dict[src, coh, gene].sum() >= 10:
                    use_preds = pred_dfs[src, coh].loc[pnt_comb, train_samps]

                    siml_dicts['Pnt'][src, coh][pnt_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][pnt_comb]],
                            use_preds.loc[copy_dict[src, coh, gene]]
                            )

                    plt_tupl = (auc_vals[pnt_comb],
                                siml_dicts['Pnt'][src, coh][pnt_comb])

                    if (siml_dicts['Pnt'][src, coh][pnt_comb] >= 0.5
                            or gene in {'TP53', 'PIK3CA', 'GATA3'}):
                        plot_dicts['Pnt'][plt_tupl] = [
                            None, ("{} in {}".format(gene, coh_lbl), '')]
                        line_dicts['Pnt'][plt_tupl] = gene

                    else:
                        plot_dicts['Pnt'][plt_tupl] = [None, ('', '')]

            elif len(pnt_comb) > 1:
                raise ValueError

            for cpy_comb in cpy_combs:
                if gn_dict[src, coh, gene].sum() >= 10:
                    use_preds = pred_dfs[src, coh].loc[cpy_comb, train_samps]

                    siml_dicts['Cpy'][src, coh][cpy_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][cpy_comb]],
                            use_preds.loc[gn_dict[src, coh, gene]]
                            )

                    plt_tupl = (auc_vals[cpy_comb],
                                siml_dicts['Cpy'][src, coh][cpy_comb])

                    if (siml_dicts['Cpy'][src, coh][cpy_comb] >= 0.75
                            or gene in {'TP53', 'PIK3CA', 'GATA3'}):
                        plot_dicts['Cpy'][plt_tupl] = [
                            None, ("{} in {}".format(gene, coh_lbl), '')]
                        line_dicts['Cpy'][plt_tupl] = gene

                    else:
                        plot_dicts['Cpy'][plt_tupl] = [None, ('', '')]

                    if (isinstance(pnt_comb, ExMcomb)
                            and gn_dict[src, coh, gene].sum() >= 20):
                        annt_lists[src, coh] |= {(cpy_comb, pnt_comb)}

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    size_mult = sum(len(siml_vals) for siml_dict in siml_dicts.values()
                    for siml_vals in siml_dict.values()) ** -0.23

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]

    ymin = min(min(siml_vals.values()) for siml_dict in siml_dicts.values()
               for siml_vals in siml_dict.values() if siml_vals)
    ymax = max(max(siml_vals.values()) for siml_dict in siml_dicts.values()
               for siml_vals in siml_dict.values() if siml_vals)
    yrng = ymax - ymin
    ylims = [ymin - yrng / 23, ymax + yrng / 23]

    ylbls = {'Pnt': ("Inferred {} Similarity"
                     "\nto Point Mutations").format(cna_lbl),
             'Cpy': ("Inferred Point Mutation Similarity"
                     "\nto {} Alterations").format(cna_lbl)}

    for k, ax in zip(['Pnt', 'Cpy'], [pnt_ax, cpy_ax]):
        for (src, coh), siml_vals in siml_dicts[k].items():
            for mcomb, siml_val in siml_vals.items():
                cur_gene = tuple(mcomb.label_iter())[0]

                auc_val = auc_lists[src, coh][mcomb]
                plt_size = size_mult * np.mean(pheno_dicts[src, coh][mcomb])
                plot_dicts[k][auc_val, siml_val][0] = plt_size * 3.1

                ax.scatter(auc_val, siml_val,
                           c=[clr_dict[cur_gene]], s=1473 * plt_size,
                           alpha=0.37, edgecolor='none')

        ax.grid(alpha=0.47, linewidth=0.9)
        ax.plot(xlims, [0, 0],
                color='black', linewidth=1.11, linestyle='--', alpha=0.67)
        ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

        ax.set_ylabel(ylbls[k], size=21, weight='bold')
        if k == 'Cpy':
            ax.set_xlabel("Subgrouping\nClassification Accuracy",
                          size=21, weight='bold')

        ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))

        for tupl in line_dicts[k]:
            line_dicts[k][tupl] = {'c': clr_dict[line_dicts[k][tupl]]}

        for val in np.linspace(args.auc_cutoff, 0.99, 200):
            if (val, 0) not in plot_dicts[k]:
                plot_dicts[k][val, 0] = [1 / 11, ('', '')]

        lbl_pos = place_scatter_labels(plot_dicts[k], ax,
                                       plt_lims=[xlims, ylims],
                                       font_size=9, line_dict=line_dicts[k],
                                       linewidth=1.19, alpha=0.37)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}-point-similarity_{}.svg".format(
                         args.ex_lbl, cna_lbl, siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()
    return annt_lists


def plot_pair_scores(cpy_mcomb, pnt_mcomb, pred_df, auc_vals, pheno_dict,
                     cdata, data_tag, args, cna_lbl, siml_metric):
    fig, ((mcomb2_ax, sctr_ax), (crnr_ax, mcomb1_ax)) = plt.subplots(
        figsize=(13, 12), nrows=2, ncols=2,
        gridspec_kw=dict(height_ratios=[4, 1], width_ratios=[1, 4])
        )

    use_gene = tuple(cpy_mcomb.label_iter())[0]
    assert tuple(pnt_mcomb.label_iter())[0] == use_gene
    cna_mtype = cna_mtypes[cna_lbl]
    if args.ex_lbl == 'IsoShal':
        cna_mtype -= shal_mtype

    assert len(cpy_mcomb.mtypes) == len(pnt_mcomb.mtypes) == 1
    cpy_type = tuple(tuple(cpy_mcomb.mtypes)[0].subtype_iter())[0][1]
    pnt_type = tuple(tuple(pnt_mcomb.mtypes)[0].subtype_iter())[0][1]

    base_mtree = tuple(cdata.mtrees.values())[0]
    all_mtype = MuType({('Gene', use_gene): base_mtree[use_gene].allkey()})
    if args.ex_lbl == 'IsoShal':
        all_mtype -= MuType({('Gene', use_gene): shal_mtype})

    use_preds = pred_df.loc[[cpy_mcomb, pnt_mcomb],
                            cdata.get_train_samples()].T
    use_preds.columns = ['Subg1', 'Subg2']

    x_min, y_min = use_preds.min()
    x_max, y_max = use_preds.max()
    x_rng, y_rng = x_max - x_min, y_max - y_min
    xlims = x_min - x_rng / 31, x_max + x_rng / 31
    ylims = y_min - y_rng / 31, y_max + y_rng / 31

    all_phn = np.array(cdata.train_pheno(all_mtype))
    use_preds['Phn'] = np.array(['Other' if phn else 'WT' for phn in all_phn])
    use_preds.loc[pheno_dict[cpy_mcomb], 'Phn'] = 'Subg1'
    use_preds.loc[pheno_dict[pnt_mcomb], 'Phn'] = 'Subg2'

    use_clrs = {'WT': variant_clrs['WT'],
                'Subg1': choose_subtype_colour(cpy_type),
                'Subg2': choose_subtype_colour(pnt_type),
                'Other': 'black'}

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

    mtype_lbls = [get_fancy_label(mtype) for mtype in [cpy_type, pnt_type]]
    subg_lbls = [
        "Subgrouping {}:\nonly {} mutation is\n{}".format(
            i + 1, use_gene, mtype_lbl)
        for i, mtype_lbl in enumerate(mtype_lbls)
        ]

    sctr_ax.text(0.98, 0.03, subg_lbls[0], size=15, c=use_clrs['Subg1'],
                 ha='right', va='bottom', transform=sctr_ax.transAxes)
    sctr_ax.text(0.03, 0.98, subg_lbls[1], size=15, c=use_clrs['Subg2'],
                 ha='left', va='top', transform=sctr_ax.transAxes)

    sns.violinplot(data=use_preds, y='Phn', x='Subg1', ax=mcomb1_ax,
                   order=['WT', 'Subg1', 'Subg2', 'Other'],
                   palette=use_clrs, orient='h', linewidth=0, cut=0)

    sns.violinplot(data=use_preds, x='Phn', y='Subg2', ax=mcomb2_ax,
                   order=['WT', 'Subg2', 'Subg1', 'Other'],
                   palette=use_clrs, orient='v', linewidth=0, cut=0)

    for ax in sctr_ax, mcomb1_ax, mcomb2_ax:
        ax.grid(alpha=0.47, linewidth=0.9)

    for mcomb_ax in mcomb1_ax, mcomb2_ax:
        for i in range(3):
            mcomb_ax.get_children()[i * 2].set_alpha(0.61)
            mcomb_ax.get_children()[i * 2].set_linewidth(0)

        if (use_preds.Phn == 'Other').sum() > 1:
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

    crnr_ax.text(0.95, 0.59, "(AUC1: {:.3f})".format(auc_vals[cpy_mcomb]),
                 size=11, ha='right', transform=crnr_ax.transAxes,
                 clip_on=False)
    crnr_ax.text(0, 0.55, "(AUC2: {:.3f})".format(auc_vals[pnt_mcomb]),
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

    fig.tight_layout(w_pad=-1.3, h_pad=-1.3)
    plt.savefig(os.path.join(
        plot_dir, data_tag, "{}_{}-ortho-scores_{}__{}_{}.svg".format(
            args.ex_lbl, siml_metric, args.classif,
            use_gene, mtype_lbls[0].replace(' ', '-')
            )
        ), bbox_inches='tight', format='svg')

    plt.close()


def plot_interaction_symmetries(pred_dfs, pheno_dicts, auc_lists,
                                cdata_dict, args, cna_lbl, siml_metric):
    fig, ((pnt_ax, intx_ax1), (intx_ax2, cpy_ax)) = plt.subplots(
        figsize=(17, 17), nrows=2, ncols=2)

    cna_mtype = cna_mtypes[cna_lbl]
    copy_dict = dict()
    gn_dict = dict()
    intx_dict = dict()

    siml_dicts = {(src, coh): {k: dict() for k in ['Intx-Pnt', 'Intx-Cpy',
                                                   'Pnt-Intx', 'Cpy-Intx']}
                  for src, coh in auc_lists}
    annt_lists = {(src, coh): set() for src, coh in auc_lists}

    plot_dicts = {'Pnt': dict(), 'Cpy': dict()}
    line_dicts = {'Pnt': dict(), 'Cpy': dict()}
    clr_dict = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined and subsets of point mutations...
    for (src, coh), auc_list in auc_lists.items():
        use_aucs = auc_list[
            list(remove_pheno_dups({
                mut for mut, auc_val in auc_list.iteritems()
                if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                    and get_mut_ex(mut) == args.ex_lbl)
                }, pheno_dicts[src, coh]))
            ]

        if len(use_aucs) == 0:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {tuple(mcomb.label_iter())[0] for mcomb in use_aucs.index}
        train_samps = cdata_dict[src, coh].get_train_samples()
        coh_lbl = get_cohort_label(coh)

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

        for gene, auc_vals in use_aucs.groupby(
                lambda mcomb: tuple(mcomb.label_iter())[0]):
            cpy_combs = {
                mcomb for mcomb in auc_vals.index
                if all(
                    cna_mtype.is_supertype(tuple(mtype.subtype_iter())[0][1])
                    for mtype in mcomb.mtypes
                    )
                }

            if args.ex_lbl == 'IsoShal':
                cpy_combs = {
                    mcomb for mcomb in cpy_combs
                    if all(
                        deep_mtype.is_supertype(
                            tuple(mtype.subtype_iter())[0][1])
                        for mtype in mcomb.mtypes
                        )
                    }

            assert all(len(cpy_comb.mtypes) == 1 for cpy_comb in cpy_combs)
            cpy_types = {tuple(mcomb.mtypes)[0] for mcomb in cpy_combs}
            gene_pnt = MuType({('Gene', gene): pnt_mtype})

            intx_combs = {
                mcomb for mcomb in auc_vals.index
                if (len(mcomb.mtypes) == 2
                    and any(mtype in cpy_types for mtype in mcomb.mtypes)
                    and any(mtype == gene_pnt for mtype in mcomb.mtypes))
                }

            if (src, coh, gene) not in copy_dict:
                copy_dict[src, coh, gene] = np.array(
                    cdata_dict[src, coh].train_pheno(
                        ExMcomb(gene_pnt,
                                MuType({('Gene', gene): cna_mtype}))
                        )
                    )

                if args.ex_lbl == 'Iso':
                    ex_copy = MuType({
                        ('Gene', gene): deep_mtype | shal_mtype})
                else:
                    ex_copy = MuType({('Gene', gene): deep_mtype})

                gn_dict[src, coh, gene] = np.array(
                    cdata_dict[src, coh].train_pheno(
                        ExMcomb(ex_copy, gene_pnt))
                    )

                intx_dict[src, coh, gene] = np.array(
                    cdata_dict[src, coh].train_pheno(
                        ExMcomb(ex_copy, gene_pnt,
                                MuType({('Gene', gene): cna_mtype}))
                        )
                    )

            for intx_comb in intx_combs:
                use_preds = pred_dfs[src, coh].loc[intx_comb, train_samps]

                if copy_dict[src, coh, gene].sum() >= 10:
                    clr_dict[gene] = None

                    siml_dicts[src, coh]['Intx-Cpy'][intx_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][intx_comb]],
                            use_preds.loc[copy_dict[src, coh, gene]]
                            )

                if gn_dict[src, coh, gene].sum() >= 10:
                    clr_dict[gene] = None

                    siml_dicts[src, coh]['Intx-Pnt'][intx_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][intx_comb]],
                            use_preds.loc[gn_dict[src, coh, gene]]
                            )

            for cpy_comb in cpy_combs:
                use_preds = pred_dfs[src, coh].loc[cpy_comb, train_samps]

                if intx_dict[src, coh, gene].sum() >= 10:
                    clr_dict[gene] = None

                    siml_dicts[src, coh]['Cpy-Intx'][cpy_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][cpy_comb]],
                            use_preds.loc[intx_dict[src, coh, gene]]
                            )

            pnt_comb = {mcomb for mcomb in auc_vals.index
                        if all(mtype == gene_pnt for mtype in mcomb.mtypes)}

            if len(pnt_comb) == 1:
                pnt_comb = tuple(pnt_comb)[0]
                use_preds = pred_dfs[src, coh].loc[pnt_comb, train_samps]
                assert not (pheno_dicts[src, coh][pnt_comb]
                            & copy_dict[src, coh, gene]).any()

                if intx_dict[src, coh, gene].sum() >= 10:
                    clr_dict[gene] = None

                    siml_dicts[src, coh]['Pnt-Intx'][pnt_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][pnt_comb]],
                            use_preds.loc[intx_dict[src, coh, gene]]
                            )

            elif len(pnt_comb) > 1:
                raise ValueError

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    size_mult = sum(len(siml_vals) for siml_dict in siml_dicts.values()
                    for siml_vals in siml_dict.values()) ** -0.23

    plt_min = min(min(siml_vals.values()) for siml_dict in siml_dicts.values()
                  for siml_vals in siml_dict.values() if siml_vals)
    plt_max = max(max(siml_vals.values()) for siml_dict in siml_dicts.values()
                  for siml_vals in siml_dict.values() if siml_vals)

    plt_rng = plt_max - plt_min
    plt_lims = [min(-0.07, plt_min - plt_rng / 23),
                max(1.07, plt_max + plt_rng / 23)]

    for (src, coh), siml_dict in siml_dicts.items():
        both_intx = set(siml_dict['Intx-Pnt']) & set(siml_dict['Intx-Cpy'])

        for mcomb in both_intx:
            cur_gene = tuple(mcomb.label_iter())[0]
            plt_size = size_mult * np.mean(pheno_dicts[src, coh][mcomb])

            intx_ax2.scatter(siml_dict['Intx-Pnt'][mcomb],
                             siml_dict['Intx-Cpy'][mcomb],
                             c=[clr_dict[cur_gene]], s=1473 * plt_size,
                             alpha=0.31, edgecolor='none')

        for pnt_comb, pnt_siml in siml_dict['Pnt-Intx'].items():
            cur_gene = tuple(pnt_comb.label_iter())[0]
            plt_size = size_mult * np.mean(pheno_dicts[src, coh][pnt_comb])

            cpy_intx = {mcomb for mcomb in siml_dict['Intx-Cpy']
                        if tuple(mcomb.label_iter())[0] == cur_gene}

            for intx_comb in cpy_intx:
                intx_ax1.scatter(pnt_siml, siml_dict['Intx-Cpy'][intx_comb],
                                 c=[clr_dict[cur_gene]], s=1473 * plt_size,
                                 alpha=0.31, edgecolor='none')

            pnt_intx = {mcomb for mcomb in siml_dict['Intx-Pnt']
                        if tuple(mcomb.label_iter())[0] == cur_gene}

            for intx_comb in pnt_intx:
                pnt_ax.scatter(siml_dict['Intx-Pnt'][intx_comb], pnt_siml,
                               c=[clr_dict[cur_gene]], s=1473 * plt_size,
                               alpha=0.31, edgecolor='none')

        for cpy_comb, cpy_siml in siml_dict['Cpy-Intx'].items():
            cur_gene = tuple(cpy_comb.label_iter())[0]
            plt_size = size_mult * np.mean(pheno_dicts[src, coh][cpy_comb])

            cpy_intx = {mcomb for mcomb in siml_dict['Intx-Cpy']
                        if tuple(mcomb.label_iter())[0] == cur_gene}

            for intx_comb in cpy_intx:
                cpy_ax.scatter(cpy_siml, siml_dict['Intx-Cpy'][intx_comb],
                               c=[clr_dict[cur_gene]], s=1473 * plt_size,
                               alpha=0.31, edgecolor='none')

    for ax in (pnt_ax, intx_ax1, intx_ax2, cpy_ax):
        ax.grid(alpha=0.37, linewidth=0.71)
        ax.plot(plt_lims, plt_lims,
                color='black', linewidth=1.13, linestyle='--', alpha=0.47)

        ax.xaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 5]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 5]))
        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    for ax in (pnt_ax, intx_ax1):
        ax.set_xticklabels([])
    for ax in (cpy_ax, intx_ax1):
        ax.set_yticklabels([])

    intx_ax2.set_xlabel(
        "Point Mutations' Similarity\nto Point & {}".format(cna_lbl),
        size=27, weight='bold'
        )
    intx_ax2.set_ylabel(
        "{0} Alterations' Similarity\nto Point & {0}".format(cna_lbl),
        size=27, weight='bold'
        )

    cpy_ax.set_xlabel(
        "Point & {0} Similarity\nto {0} Alterations".format(cna_lbl),
        size=27, weight='bold'
        )
    pnt_ax.set_ylabel(
        "Point & {} Similarity\nto Point Mutations".format(cna_lbl),
        size=27, weight='bold'
        )

    fig.tight_layout(w_pad=3, h_pad=3)
    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}-interaction-symmetries_{}.svg".format(
                         args.ex_lbl, cna_lbl, siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()
    return annt_lists


def main():
    parser = argparse.ArgumentParser(
        'plot_copy',
        description="Compares copy # alterations subgroupings with a cohort."
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

    # parse command line arguments, find completed runs for this classifier
    args = parser.parse_args()
    out_datas = tuple(Path(base_dir).glob(
        os.path.join("*", "out-aucs__*__*__{}.p.gz".format(args.classif))))

    os.makedirs(plot_dir, exist_ok=True)
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

    for siml_metric in args.siml_metrics:
        if args.auc_cutoff < 1:
            for cna_lbl in ['Gain', 'Loss']:
                annt_types = plot_point_similarity(
                    pred_dfs, phn_dicts, auc_lists,
                    cdata_dict, args, cna_lbl, siml_metric
                    )

                for (src, coh), annt_list in annt_types.items():
                    if annt_list:
                        os.makedirs(
                            os.path.join(plot_dir, '__'.join([src, coh])),
                            exist_ok=True
                            )

                    for cpy_mcomb, pnt_mcomb in annt_list:
                        plot_pair_scores(
                            cpy_mcomb, pnt_mcomb, pred_dfs[src, coh],
                            auc_lists[src, coh], phn_dicts[src, coh],
                            cdata_dict[src, coh], '__'.join([src, coh]),
                            args, cna_lbl, siml_metric
                            )

                intx_types = plot_interaction_symmetries(
                    pred_dfs, phn_dicts, auc_lists,
                    cdata_dict, args, cna_lbl, siml_metric
                    )


if __name__ == '__main__':
    main()

