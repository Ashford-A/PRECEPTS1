
from ..utilities.mutations import pnt_mtype, shal_mtype, copy_mtype, ExMcomb
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import (load_cohorts_data, siml_fxs, cna_mtypes,
                    remove_pheno_dups, get_mut_ex, choose_subtype_colour)
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path

from itertools import product
from itertools import combinations as combn
from itertools import permutations as permt

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'copy')


def get_mcomb_type(mcomb):
    mcomb_type = 'other'

    if all(get_subtype(mtype) == pnt_mtype for mtype in mcomb.mtypes):
        mcomb_type = 'pnt'
    elif all(copy_mtype.is_supertype(get_subtype(mtype))
             for mtype in mcomb.mtypes):
        mcomb_type = 'cpy'

    elif (len(mcomb.mtypes) == 2
            and all(get_subtype(mtype) == pnt_mtype
                    or copy_mtype.is_supertype(get_subtype(mtype))
                    for mtype in mcomb.mtypes)):
        mcomb_type = 'intx'

    return mcomb_type


def plot_point_similarity(auc_lists, siml_dicts, cdata_dict,
                          args, cna_lbl, siml_metric):
    fig, (pnt_ax, cpy_ax) = plt.subplots(figsize=(12, 14), nrows=2)

    cna_mtype = cna_mtypes[args.ex_lbl][cna_lbl][0]
    plt_simls = {'pnt': dict(), 'cpy': dict()}
    plot_dicts = {'pnt': dict(), 'cpy': dict()}
    line_dicts = {'pnt': dict(), 'cpy': dict()}
    clr_dict = dict()
    ymin, ymax = -0.53, 0.53

    for (src, coh), siml_dict in siml_dicts.items():
        coh_lbl = get_cohort_label(coh)

        use_simls = {
            (comb1, comb2): simls
            for (comb1, comb2), simls in siml_dict.items()
            if ({get_mcomb_type(comb1),
                 get_mcomb_type(comb2)} == {'pnt', 'cpy'}
                and all(all(cna_mtype.is_supertype(get_subtype(mtype))
                            for mtype in comb.mtypes)
                        or all(get_subtype(mtype) == pnt_mtype
                               for mtype in comb.mtypes)
                        for comb in [comb1, comb2]))
            }

        if use_simls:
            ymin = min(ymin, min(siml_val for simls in use_simls.values()
                                 for siml_val in simls))
            ymax = max(ymax, max(siml_val for simls in use_simls.values()
                                 for siml_val in simls))

        for combs, simls in use_simls.items():
            cur_gene = get_label(combs[0])
            clr_dict[cur_gene] = None

            for (trn_comb, prj_comb), siml_val in zip(permt(combs), simls):
                comb_type = get_mcomb_type(trn_comb)

                plt_simls[comb_type][src, coh, trn_comb, prj_comb] = siml_val
                plt_tupl = (auc_lists[src, coh][trn_comb], siml_val)

                if (siml_val >= 0.5
                        or cur_gene in {'TP53', 'PIK3CA', 'GATA3'}):
                    plot_dicts[comb_type][plt_tupl] = [
                        None, ("{} in {}".format(cur_gene, coh_lbl), '')]
                    line_dicts[comb_type][plt_tupl] = cur_gene

                else:
                    plot_dicts[comb_type][plt_tupl] = [None, ('', '')]

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]
    yrng = ymax - ymin
    ylims = [ymin - yrng / 7, ymax + yrng / 7]

    ylbls = {'pnt': ("Inferred {} Similarity"
                     "\nto Point Mutations").format(cna_lbl),
             'cpy': ("Inferred Point Mutation Similarity"
                     "\nto {} Alterations").format(cna_lbl)}

    size_mult = sum(len(simls) for simls in plt_simls.values()) ** -0.23
    for k, ax in zip(['pnt', 'cpy'], [pnt_ax, cpy_ax]):
        for (src, coh, trn_comb, prj_comb), siml_val in plt_simls[k].items():
            cur_gene = get_label(trn_comb)

            auc_val = auc_lists[src, coh][trn_comb]
            plt_size = size_mult * np.mean(
                cdata_dict[src, coh].train_pheno(prj_comb))
            plot_dicts[k][auc_val, siml_val][0] = 0.23 * plt_size

            ax.scatter(auc_val, siml_val,
                       c=[clr_dict[cur_gene]], s=2891 * plt_size,
                       alpha=0.37, edgecolor='none')

        ax.grid(alpha=0.47, linewidth=0.9)
        ax.plot(xlims, [0, 0],
                color='black', linewidth=1.11, linestyle='--', alpha=0.67)
        ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

        ax.set_ylabel(ylbls[k], size=21, weight='bold')
        if k == 'cpy':
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
                                       font_size=10, line_dict=line_dicts[k],
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


def plot_similarity_symmetry(siml_dicts, pheno_dicts, cdata_dict,
                             args, cna_lbl, siml_metric):
    fig, ax = plt.subplots(figsize=(9.4, 10))

    cna_mtype = cna_mtypes[args.ex_lbl][cna_lbl][0]
    plt_simls = dict()
    plot_dict = dict()
    line_dict = dict()
    clr_dict = dict()
    annt_lists = {(src, coh): set() for src, coh in siml_dicts}
    plt_min, plt_max = -0.53, 0.53

    for (src, coh), siml_dict in siml_dicts.items():
        coh_lbl = get_cohort_label(coh).replace("TCGA-", '')

        use_simls = {
            (comb1, comb2): simls
            for (comb1, comb2), simls in siml_dict.items()
            if ({get_mcomb_type(comb1),
                 get_mcomb_type(comb2)} == {'pnt', 'cpy'}
                and len(simls) == 2
                and all(all(cna_mtype.is_supertype(get_subtype(mtype))
                            for mtype in comb.mtypes)
                        or all(get_subtype(mtype) == pnt_mtype
                               for mtype in comb.mtypes)
                        for comb in [comb1, comb2]))
            }

        if use_simls:
            plt_min = min(plt_min,
                          min(siml_val for simls in use_simls.values()
                              for siml_val in simls))
            plt_max = max(plt_max,
                          max(siml_val for simls in use_simls.values()
                              for siml_val in simls))

        for combs, simls in use_simls.items():
            cur_gene = get_label(combs[0])

            clr_dict[cur_gene] = None
            ordr = 2 * (get_mcomb_type(combs[0]) == 'cpy') - 1
            plt_simls[(src, coh, *combs[::ordr])] = simls[::ordr]
            annt_lists[src, coh] |= {combs[::ordr]}

            if (get_subtype(tuple(combs[::ordr][0].mtypes)[0])
                    & shal_mtype).is_empty():
                if cna_lbl == 'Gain':
                    cpy_lbl = "only deep gains"
                else:
                    cpy_lbl = "only deep losses"

            else:
                cpy_lbl = ''

            if (max(np.abs(np.array(simls))) > 0.5
                    or cur_gene in {'TP53', 'PIK3CA', 'GATA3'}):
                plot_dict[simls[::ordr]] = [
                    None, ("{} in {}".format(cur_gene, coh_lbl), cpy_lbl)]
                line_dict[simls[::ordr]] = cur_gene

            else:
                plot_dict[simls[::ordr]] = [None, ('', '')]

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    size_mult = len(plt_simls) ** -0.23
    plt_rng = plt_max - plt_min
    plt_lims = [min(-0.07, plt_min - plt_rng / 31),
                max(1.07, plt_max + plt_rng / 31)]

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot(plt_lims, plt_lims,
            '#550000', linewidth=1.3, linestyle='--', alpha=0.41)

    for j in [0, 1]:
        ax.plot(plt_lims, [j, j],
                color='black', linewidth=0.83, linestyle=':', alpha=0.67)
        ax.plot([j, j], plt_lims,
                color='black', linewidth=0.83, linestyle=':', alpha=0.67)

    for (src, coh, cpy_comb, pnt_comb), simls in plt_simls.items():
        cur_gene = get_label(cpy_comb)

        plt_size = np.mean(pheno_dicts[src, coh][cpy_comb])
        plt_size *= np.mean(pheno_dicts[src, coh][pnt_comb])
        plt_size = size_mult * plt_size ** 0.5
        plot_dict[simls][0] = 0.19 * plt_size

        ax.scatter(*simls, c=[clr_dict[cur_gene]],
                   s=2371 * plt_size, alpha=0.41, edgecolor='none')

    # makes sure plot labels don't overlap with equal-similarity diagonal line
    for k in np.linspace(plt_min, plt_max, 100):
        if (k, k) not in plot_dict:
            plot_dict[k, k] = [plt_rng / 387, ('', '')]

    for tupl in line_dict:
        line_dict[tupl] = {'c': clr_dict[line_dict[tupl]]}

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[plt_lims, plt_lims],
                                       font_size=12, line_dict=line_dict,
                                       linewidth=1.19, alpha=0.37)

    ax.set_xlabel(
        "Point Mutations' Similarity\nto {} Alterations".format(cna_lbl),
        size=23, weight='bold'
        )
    ax.set_ylabel(
        "{} Alterations' Similarity\nto Point Mutations".format(cna_lbl),
        size=23, weight='bold'
        )

    ax.xaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))
    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}-similarity-symmetry_{}.svg".format(
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

    use_gene = get_label(cpy_mcomb)
    assert get_label(pnt_mcomb) == use_gene
    cna_mtype = cna_mtypes[args.ex_lbl][cna_lbl][0]

    assert len(cpy_mcomb.mtypes) == len(pnt_mcomb.mtypes) == 1
    cpy_type = get_subtype(tuple(cpy_mcomb.mtypes)[0])
    pnt_type = get_subtype(tuple(pnt_mcomb.mtypes)[0])

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


def plot_interaction_symmetries(siml_dicts, pheno_dicts, cdata_dict,
                                args, cna_lbl, siml_metric):
    fig, axarr = plt.subplots(figsize=(17, 17), nrows=2, ncols=2)

    plt_simls = {
        (src, coh): {k: dict() for k in [('intx', 'pnt'), ('intx', 'cpy'),
                                         ('pnt', 'intx'), ('cpy', 'intx')]}
        for src, coh in siml_dicts
        }

    cna_mtype = cna_mtypes[args.ex_lbl][cna_lbl][0]
    annt_lists = {(src, coh): set() for src, coh in siml_dicts}
    clr_dict = dict()
    plt_min, plt_max = -0.53, 0.53

    for (src, coh), siml_dict in siml_dicts.items():
        use_simls = {
            (comb1, comb2): simls
            for (comb1, comb2), simls in siml_dict.items()
            if (all(all(cna_mtype.is_supertype(get_subtype(mtype))
                        or get_subtype(mtype) == pnt_mtype
                        for mtype in comb.mtypes)
                    for comb in [comb1, comb2])

                and sum(len(comb.mtypes) == 2
                        and any(get_subtype(mtype) == pnt_mtype
                                for mtype in comb.mtypes)
                        for comb in [comb1, comb2]) == 1)
            }

        if use_simls:
            plt_min = min(plt_min,
                          min(siml_val for simls in use_simls.values()
                              for siml_val in simls))
            plt_max = max(plt_max,
                          max(siml_val for simls in use_simls.values()
                              for siml_val in simls))

        for combs, simls in use_simls.items():
            cur_gene = get_label(combs[0])

            if cur_gene in clr_dict:
                clr_dict[cur_gene] += 1
            else:
                clr_dict[cur_gene] = 1

            for (trn_comb, prj_comb), siml_val in zip(permt(combs), simls):
                comb_types = (get_mcomb_type(trn_comb),
                              get_mcomb_type(prj_comb))

                if trn_comb in plt_simls[src, coh][comb_types]:
                    plt_simls[src, coh][comb_types][
                        trn_comb][prj_comb] = siml_val

                else:
                    plt_simls[src, coh][comb_types][trn_comb] = {
                        prj_comb: siml_val}

    lgnd_gns = pd.Series(clr_dict).sort_values().index[:-13:-1]
    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    lgnd_mrks = [Line2D([], [], marker='o', linestyle='None',
                        markersize=25, alpha=0.61,
                        markerfacecolor=clr_dict[gene],
                        markeredgecolor='none')
                 for gene in lgnd_gns]

    size_mult = sum(len(simls) for comb_simls in plt_simls.values()
                    for simls in comb_simls.values()) ** -0.23

    plt_rng = plt_max - plt_min
    plt_lims = [min(-0.07, plt_min - plt_rng / 53),
                max(1.07, plt_max + plt_rng / 53)]

    for (src, coh), siml_dict in plt_simls.items():
        both_intx = set(siml_dict['intx', 'pnt'])
        both_intx &= set(siml_dict['intx', 'cpy'])

        for trn_comb in both_intx:
            cur_gene = get_label(trn_comb)
            plt_size = size_mult * np.mean(pheno_dicts[src, coh][trn_comb])

            for (pnt_comb, pnt_siml), (cpy_comb, cpy_siml) in product(
                    siml_dict['intx', 'pnt'][trn_comb].items(),
                    siml_dict['intx', 'cpy'][trn_comb].items()
                    ):

                axarr[1, 0].scatter(pnt_siml, cpy_siml,
                                    c=[clr_dict[cur_gene]], s=5173 * plt_size,
                                    alpha=0.31, edgecolor='none')

        for pnt_comb, pnt_simls in siml_dict['pnt', 'intx'].items():
            cur_gene = get_label(pnt_comb)
            pnt_size = np.mean(pheno_dicts[src, coh][pnt_comb])

            for intx_comb, intx_siml1 in pnt_simls.items():
                cpy_intx = {
                    comb: simls[intx_comb]
                    for comb, simls in siml_dict['cpy', 'intx'].items()
                    if intx_comb in simls
                    }

                for cpy_comb, intx_siml2 in cpy_intx.items():
                    cpy_size = np.mean(pheno_dicts[src, coh][cpy_comb])
                    plt_size = size_mult * (pnt_size * cpy_size) ** 0.5

                    axarr[0, 1].scatter(intx_siml2, intx_siml1,
                                        c=[clr_dict[cur_gene]],
                                        s=5173 * plt_size, alpha=0.31,
                                        edgecolor='none')

                if intx_comb in siml_dict['intx', 'pnt']:
                    intx_size = np.mean(pheno_dicts[src, coh][intx_comb])
                    plt_size = size_mult * (pnt_size * intx_size) ** 0.5
                    intx_siml2 = siml_dict['intx', 'pnt'][intx_comb][pnt_comb]

                    axarr[0, 0].scatter(intx_siml2, intx_siml1,
                                        c=[clr_dict[cur_gene]],
                                        s=5173 * plt_size, alpha=0.31,
                                        edgecolor='none')

        for cpy_comb, cpy_simls in siml_dict['cpy', 'intx'].items():
            cur_gene = get_label(cpy_comb)
            cpy_size = np.mean(pheno_dicts[src, coh][cpy_comb])

            for intx_comb, intx_siml1 in cpy_simls.items():
                if intx_comb in siml_dict['intx', 'cpy']:
                    intx_size = np.mean(pheno_dicts[src, coh][intx_comb])
                    plt_size = size_mult * (cpy_size * intx_size) ** 0.5
                    intx_siml2 = siml_dict['intx', 'cpy'][intx_comb][cpy_comb]

                    axarr[1, 1].scatter(intx_siml1, intx_siml2,
                                        c=[clr_dict[cur_gene]],
                                        s=5173 * plt_size, alpha=0.27,
                                        edgecolor='none')

    for ax in axarr.flatten():
        ax.grid(alpha=0.37, linewidth=0.71)
        ax.plot(plt_lims, plt_lims,
                color='black', linewidth=1.13, linestyle='--', alpha=0.47)

        for j in [0, 1]:
            ax.plot(plt_lims, [j, j],
                    color='black', linewidth=0.71, linestyle=':', alpha=0.67)
            ax.plot([j, j], plt_lims,
                    color='black', linewidth=0.71, linestyle=':', alpha=0.67)

        ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

    for ax in axarr[0, :]:
        ax.set_xticklabels([])
    for ax in axarr[:, 1]:
        ax.set_yticklabels([])

    axarr[1, 0].set_xlabel(
        "Point Mutations' Similarity\nto Point & {}".format(cna_lbl),
        size=33, weight='bold'
        )
    axarr[1, 0].set_ylabel(
        "{0} Alterations' Similarity\nto Point & {0}".format(cna_lbl),
        size=33, weight='bold'
        )

    axarr[1, 1].set_xlabel(
        "Point & {0} Similarity\nto {0} Alterations".format(cna_lbl),
        size=33, weight='bold'
        )
    axarr[0, 0].set_ylabel(
        "Point & {} Similarity\nto Point Mutations".format(cna_lbl),
        size=33, weight='bold'
        )

    fig.legend(lgnd_mrks, lgnd_gns, bbox_to_anchor=(0.5, -0.007),
               frameon=False, fontsize=31, ncol=4, loc=9, handletextpad=0.13)

    fig.tight_layout(w_pad=3, h_pad=3)
    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}-interaction-symmetries_{}.svg".format(
                         args.ex_lbl, cna_lbl, siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()
    return annt_lists


def plot_overlap_similarity(siml_dicts, pheno_dicts, cdata_dict,
                            args, siml_metric):
    fig, ax = plt.subplots(figsize=(11, 7))

    cna_mtype = cna_mtypes[args.ex_lbl]['All']
    ovlp_dict = dict()
    plot_dict = dict()
    ymin, ymax = -0.53, 0.53

    for (src, coh), siml_dict in siml_dicts.items():
        coh_lbl = get_cohort_label(coh)

        use_simls = {
            (comb1, comb2): simls
            for (comb1, comb2), simls in siml_dict.items()
            if ({get_mcomb_type(comb1),
                 get_mcomb_type(comb2)} == {'pnt', 'cpy'}
                and len(simls) == 2
                and all(all(cna_mtype.is_supertype(get_subtype(mtype))
                            for mtype in comb.mtypes)
                        or all(get_subtype(mtype) == pnt_mtype
                               for mtype in comb.mtypes)
                        for comb in [comb1, comb2]))
            }

        if use_simls:
            ymin = min(ymin, min(siml_val for simls in use_simls.values()
                                 for siml_val in simls) - 0.19)
            ymax = max(ymax, max(siml_val for simls in use_simls.values()
                                 for siml_val in simls) + 0.19)

        for combs, simls in use_simls.items():
            cur_gene = get_label(combs[0])

            mtype_phns = [
                np.array(cdata_dict[src, coh].train_pheno(
                    tuple(comb.mtypes)[0]))
                for comb in combs
                ]

            ovlp_odds, ovlp_pval = fisher_exact(
                table=pd.crosstab(*mtype_phns))

            ovlp_dict[(src, coh, *combs)] = (
                np.sign(ovlp_odds - 1) * -np.log10(ovlp_pval),
                (simls[0] + simls[1]) / 2
                )

            cpy_comb = combs[int(get_mcomb_type(combs[1]) == 'cpy')]
            cpy_subt = get_subtype(tuple(cpy_comb.mtypes)[0])

            plot_dict[ovlp_dict[(src, coh, *combs)]] = [
                None, ("{} in {}".format(cur_gene, coh_lbl),
                       get_fancy_label(cpy_subt))
                ]

    size_mult = len(ovlp_dict) ** -0.23
    ovlp_df = pd.DataFrame(ovlp_dict, index=['Ovlp', 'Siml']).transpose()

    gene_gby = ovlp_df.groupby(lambda x: get_label(x[2]))
    clr_dict = {gene: choose_label_colour(gene) for gene, _ in gene_gby}
    line_dict = {
        (ovlp_val, siml_val): dict(c=clr_dict[get_label(comb)])
        for (_, _, comb, _), (ovlp_val, siml_val) in ovlp_dict.items()
        }

    plot_lims = ovlp_df.quantile(q=[0, 1])
    plot_diff = plot_lims.diff().iloc[1]
    plot_lims.Ovlp += plot_diff.Ovlp * np.array([-4.3, 4.3]) ** -1
    plot_lims.Siml += plot_diff.Siml * np.array([-4.3, 4.3]) ** -1
    plot_gaps = plot_lims.diff().iloc[1] / 4.73

    plot_lims.Ovlp[0] = min(plot_lims.Ovlp[0], -1.07)
    plot_lims.Ovlp[1] = max(plot_lims.Ovlp[1], plot_gaps.Ovlp, 1.07)
    plot_lims.Siml[0] = min(plot_lims.Siml[0], -0.53)
    plot_lims.Siml[1] = max(plot_lims.Siml[1], plot_gaps.Siml, 0.53)

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot(plot_lims.Ovlp, [0, 0],
            color='black', linewidth=1.1, linestyle=':', alpha=0.71)
    ax.plot([0, 0], plot_lims.Siml,
            color='black', linewidth=1.1, linestyle=':', alpha=0.71)

    for (src, coh, comb1, comb2), (ovlp_val, siml_val) in ovlp_df.iterrows():
        cur_gene = get_label(comb1)

        plt_size = np.sqrt(np.mean(pheno_dicts[src, coh][comb1])
                           * np.mean(pheno_dicts[src, coh][comb2]))
        plt_size *= size_mult
        plot_dict[ovlp_val, siml_val][0] = 0.17 * plt_size

        ax.scatter(ovlp_val, siml_val, c=[clr_dict[cur_gene]],
                   s=2071 * plt_size, alpha=0.37, edgecolor='none')

    if plot_dict:
        lbl_pos = place_scatter_labels(
            plot_dict, ax, plt_lims=[plot_lims.Ovlp, plot_lims.Siml],
            font_size=11, line_dict=line_dict, linewidth=1.19, alpha=0.37
            )

    plt.xlabel("Genomic Co-occurence", size=23, weight='semibold')
    plt.ylabel("Transcriptomic Similarity", size=23, weight='semibold')

    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.set_xlim(*plot_lims.Ovlp)
    ax.set_ylim(*plot_lims.Siml)

    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}-overlap-similarity_{}.svg".format(
                         args.ex_lbl, siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


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
    parser.add_argument('--data_cache')
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
    pred_dfs, phn_dicts, auc_lists, cdata_dict = load_cohorts_data(
        out_list, args.ex_lbl, args.data_cache)

    for siml_metric in args.siml_metrics:
        siml_fx = siml_fxs[siml_metric]
        siml_dicts = {(src, coh): dict() for src, coh in auc_lists}

        for (src, coh), auc_list in auc_lists.items():
            use_aucs = auc_list[
                list(remove_pheno_dups({
                    mut for mut, auc_val in auc_list.iteritems()
                    if (isinstance(mut, ExMcomb)
                        and auc_val >= args.auc_cutoff
                        and get_mut_ex(mut) == args.ex_lbl
                        and all(get_subtype(mtype) == pnt_mtype
                                or copy_mtype.is_supertype(get_subtype(mtype))
                                for mtype in mut.mtypes))
                    }, phn_dicts[src, coh]))
                ]

            if len(use_aucs) == 0:
                continue

            base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
            use_genes = {get_label(mcomb) for mcomb in use_aucs.index}
            train_samps = cdata_dict[src, coh].get_train_samples()

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

            for gene, auc_vals in use_aucs.groupby(get_label):
                gene_pnt = MuType({('Gene', gene): pnt_mtype})
                cna_all = MuType({
                    ('Gene', gene): cna_mtypes[args.ex_lbl]['All']})

                for mcomb1, mcomb2 in combn(auc_vals.index, 2):
                    use_preds1 = pred_dfs[src, coh].loc[mcomb1, train_samps]
                    use_preds2 = pred_dfs[src, coh].loc[mcomb2, train_samps]

                    siml_dicts[src, coh][mcomb1, mcomb2] = (
                        siml_fx(use_preds1.loc[~all_phns[gene]],
                                use_preds1.loc[phn_dicts[src, coh][mcomb1]],
                                use_preds1.loc[phn_dicts[src, coh][mcomb2]]),
                        siml_fx(use_preds2.loc[~all_phns[gene]],
                                use_preds2.loc[phn_dicts[src, coh][mcomb2]],
                                use_preds2.loc[phn_dicts[src, coh][mcomb1]])
                        )

                proj_combs = {ExMcomb(cna_all, gene_pnt)}
                proj_combs |= {
                    ExMcomb(gene_pnt, MuType({('Gene', gene): cna_type}))
                    for cna_type in (cna_mtypes[args.ex_lbl]['Gain']
                                     + cna_mtypes[args.ex_lbl]['Loss'])
                    }

                proj_combs |= {
                    ExMcomb(cna_all, gene_pnt,
                            MuType({('Gene', gene): cna_type}))
                    for cna_type in (cna_mtypes[args.ex_lbl]['Gain']
                                     + cna_mtypes[args.ex_lbl]['Loss'])
                    }

                proj_phns = {
                    comb: np.array(cdata_dict[src, coh].train_pheno(comb))
                    for comb in proj_combs
                    }

                proj_combs = {prj_comb for prj_comb, phn in proj_phns.items()
                              if (phn.sum() >= 10
                                  and not any(mcomb.mtypes == prj_comb.mtypes
                                              for mcomb in auc_vals.index))}

                for mcomb in auc_vals.index:
                    use_preds = pred_dfs[src, coh].loc[mcomb, train_samps]

                    for prj_comb in proj_combs:
                        siml_dicts[src, coh][mcomb, prj_comb] = (siml_fx(
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[phn_dicts[src, coh][mcomb]],
                            use_preds.loc[proj_phns[prj_comb]]
                            ), )

        if args.auc_cutoff < 1:
            for cna_lbl in ['Gain', 'Loss']:
                plot_point_similarity(auc_lists, siml_dicts, cdata_dict,
                                      args, cna_lbl, siml_metric)

                annt_types = plot_similarity_symmetry(siml_dicts, phn_dicts,
                                                      cdata_dict, args,
                                                      cna_lbl, siml_metric)

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
                    siml_dicts, phn_dicts,
                    cdata_dict, args, cna_lbl, siml_metric
                    )

            plot_overlap_similarity(siml_dicts, phn_dicts, cdata_dict,
                                    args, siml_metric)


if __name__ == '__main__':
    main()

