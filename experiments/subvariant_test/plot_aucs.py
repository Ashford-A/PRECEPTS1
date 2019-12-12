
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

from HetMan.experiments.subvariant_test import pnt_mtype, copy_mtype
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.utils import get_fancy_label
from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.mutations import MuType

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import average_precision_score as aupr_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from colorsys import hls_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def choose_gene_colour(gene, clr_seed=15707, clr_lum=0.5, clr_sat=0.8):
    np.random.seed(int((clr_seed + np.prod([ord(char) for char in gene]))
                       % (2 ** 14)))

    return hls_to_rgb(h=np.random.uniform(size=1)[0], l=clr_lum, s=clr_sat)


def place_labels(pnt_dict, lims=(0.48, 1.01), lbl_dens=1.):
    lim_gap = (lbl_dens * 17) / (lims[1] - lims[0])

    # initialize objects storing where each label will be positioned, and how
    # much space needs to be left around already placed points and labels
    lbl_pos = {pnt: None for pnt, (_, lbls) in pnt_dict.items() if lbls[0]}
    pnt_gaps = {pnt: sz / lim_gap for pnt, (sz, _) in pnt_dict.items()}
    pnt_boxs = {pnt: [[gap * 1.53, gap * 1.53], [gap * 1.53, gap * 1.53]]
                for pnt, gap in pnt_gaps.items()}

    # calculate how much space each label to plot will occupy once placed
    lbl_wdths = {
        pnt: (max(len(ln) for ln in lbls[1].split('\n')) * 0.17 / lim_gap
              if lbls[1] else len(lbls[0]) * 0.26 / lim_gap)
        for pnt, (_, lbls) in pnt_dict.items()
        }

    lbl_hghts = {
        pnt: ((0.64 + lbls[1].count('\n') * 0.29) / lim_gap
              if lbls[1] else 0.37 / lim_gap)
        for pnt, (_, lbls) in pnt_dict.items()
        }

    # for each point, check if there is enough space to plot its label
    # to the left of it...
    for pnt in sorted(set(lbl_pos)):
        if (pnt[0] > (lims[0] + lbl_wdths[pnt])
            and not any((((pnt[0] - pnt_boxs[pnt][0][0] - lbl_wdths[pnt])
                          < (pnt2[0] - pnt_boxs[pnt2][0][0])
                          < (pnt[0] + pnt_boxs[pnt][0][1]))
                         or ((pnt[0] - pnt_boxs[pnt][0][0] - lbl_wdths[pnt])
                             < (pnt2[0] + pnt_boxs[pnt2][0][1])
                             < (pnt[0] + pnt_boxs[pnt][0][1]))
                         or ((pnt2[0] - pnt_boxs[pnt2][0][0])
                             < pnt[0] < (pnt2[0] + pnt_boxs[pnt2][0][1])))

                        and (((pnt[1] - pnt_boxs[pnt][1][0]
                               - lbl_hghts[pnt] / 1.9)
                              < (pnt2[1] - pnt_boxs[pnt2][1][0])
                              < (pnt[1] + pnt_boxs[pnt][1][1]
                                 + lbl_hghts[pnt] / 2.1))
                             or ((pnt[1] - pnt_boxs[pnt][1][0]
                                  - lbl_hghts[pnt] / 1.9)
                                 < (pnt2[1] + pnt_boxs[pnt2][1][1])
                                 < (pnt[1] + pnt_boxs[pnt][1][1]))
                             or ((pnt2[1] - pnt_boxs[pnt2][1][0])
                                 < pnt[1] < (pnt2[1] + pnt_boxs[pnt2][1][1])))
                        for pnt2 in pnt_dict if pnt2 != pnt)):
 
            lbl_pos[pnt] = (pnt[0] - pnt_gaps[pnt], pnt[1]), 'right'
            pnt_boxs[pnt][0][0] = max(pnt_boxs[pnt][0][0], lbl_wdths[pnt])

            pnt_boxs[pnt][1][0] = max(pnt_boxs[pnt][1][0], lbl_hghts[pnt])
            pnt_boxs[pnt][1][1] = max(pnt_boxs[pnt][1][1],
                                      lbl_hghts[pnt] / 1.3)

        # ...if there isn't, check if there is enough space to plot its
        # label to the right of it
        elif (pnt[0] < (lims[1] - lbl_wdths[pnt])
              and not any((((pnt[0] - pnt_boxs[pnt][0][0])
                            < (pnt2[0] - pnt_boxs[pnt2][0][0])
                            < (pnt[0] + pnt_boxs[pnt][0][1] + lbl_wdths[pnt]))
                           or ((pnt[0] - pnt_boxs[pnt][0][0])
                               < (pnt2[0] + pnt_boxs[pnt2][0][1])
                               < (pnt[0] + pnt_boxs[pnt][0][1]
                                  + lbl_wdths[pnt]))
                           or ((pnt2[0] - pnt_boxs[pnt2][0][0])
                               < pnt[0] < (pnt2[0] + pnt_boxs[pnt2][0][1])))

                          and (((pnt[1] - pnt_boxs[pnt][1][0]
                                 - lbl_hghts[pnt] / 1.9)
                                < (pnt2[1] - pnt_boxs[pnt2][1][0])
                                < (pnt[1] + pnt_boxs[pnt][1][1]
                                   + lbl_hghts[pnt] / 2.1))
                               or ((pnt[1] - pnt_boxs[pnt][1][0])
                                   < (pnt2[1] + pnt_boxs[pnt2][1][1])
                                   < (pnt[1] + pnt_boxs[pnt][1][1]
                                      + lbl_hghts[pnt] / 2.1))
                               or ((pnt2[1] - pnt_boxs[pnt2][1][0])
                                   < pnt[1]
                                   < (pnt2[1] + pnt_boxs[pnt2][1][1])))
                          for pnt2 in pnt_dict if pnt2 != pnt)):

            lbl_pos[pnt] = (pnt[0] + pnt_gaps[pnt], pnt[1]), 'left'
            pnt_boxs[pnt][0][1] = max(pnt_boxs[pnt][0][1], lbl_wdths[pnt])

            pnt_boxs[pnt][1][0] = max(pnt_boxs[pnt][1][0], lbl_hghts[pnt])
            pnt_boxs[pnt][1][1] = max(pnt_boxs[pnt][1][1],
                                      lbl_hghts[pnt] / 1.3)

    # for labels that couldn't be placed right beside their points, look for
    # empty space in the vicinity
    i = 0
    while i < 1491 and any(lbl is None for lbl in lbl_pos.values()):
        i += 0.5

        for pnt in tuple(pnt_dict):
            if pnt in lbl_pos and lbl_pos[pnt] is None:
                new_pos = ((67 + (i * np.random.randn(2))) / (lim_gap * 131)
                           + [pnt[0], pnt[1]])

                # exclude areas too close to the edge of the plot from the
                # vicinity to search over for the label
                new_pos[0] = new_pos[0].round(5).clip(
                    lims[0] + lbl_wdths[pnt] * 1.7, lims[1] - lbl_wdths[pnt])
                new_pos[1] = new_pos[1].round(5).clip(
                    lims[0] + lbl_hghts[pnt] * 1.7, lims[1] - lbl_hghts[pnt])

                # exclude areas too far from the corresponding point from
                # the vicinity to search over for the label
                new_pos[0] = new_pos[0].round(5).clip(
                    pnt[0] - (lims[1] - lims[0]) * 0.51,
                    pnt[0] + (lims[1] - lims[0]) * 0.51
                    )
                new_pos[1] = new_pos[1].round(5).clip(
                    pnt[1] - (lims[1] - lims[0]) * 0.51,
                    pnt[1] + (lims[1] - lims[0]) * 0.51
                    )
 
                if not (any((((new_pos[0] - lbl_wdths[pnt] / 1.6)
                              < (pnt2[0] - pnt_boxs[pnt2][0][0])
                              < (new_pos[0] + lbl_wdths[pnt] / 1.6))
                             or ((new_pos[0] - lbl_wdths[pnt] / 1.6)
                                 < (pnt2[0] + pnt_boxs[pnt2][0][1])
                                 < (new_pos[0] + lbl_wdths[pnt] / 1.6))
                             or ((pnt2[0] - pnt_boxs[pnt2][0][0])
                                 < new_pos[0]
                                 < (pnt2[0] + pnt_boxs[pnt2][0][1])))

                            and (((new_pos[1] - lbl_hghts[pnt] / 1.4)
                                  < (pnt2[1] - pnt_boxs[pnt2][1][0])
                                  < (new_pos[1] + lbl_hghts[pnt] / 1.4))
                                 or ((new_pos[1] - lbl_hghts[pnt] / 1.4)
                                      < (pnt2[1] + pnt_boxs[pnt2][1][1])
                                      < (new_pos[1] + lbl_hghts[pnt] / 1.4))
                                 or ((pnt2[1] - pnt_boxs[pnt2][1][0])
                                     < new_pos[1]
                                     < (pnt2[1] + pnt_boxs[pnt2][1][1])))
                            for pnt2 in pnt_dict)

                        or any(((new_pos[0] - pos2[0][0]) ** 2
                                + (new_pos[1] - pos2[0][1]) ** 2) ** 0.5
                               < (lim_gap ** -0.93)
                               for pos2 in lbl_pos.values()
                               if pos2 is not None)

                        or any((((new_pos[0] - lbl_wdths[pnt] / 1.6)
                                 < (pos2[0][0] - lbl_wdths[pnt2] / 1.6)
                                 < (new_pos[0] + lbl_wdths[pnt] / 1.6))
                                or ((new_pos[0] - lbl_wdths[pnt] / 1.6)
                                    < (pos2[0][0] + lbl_wdths[pnt2] / 1.6)
                                    < (new_pos[0] + lbl_wdths[pnt] / 1.6)))

                               and (((new_pos[1] - lbl_hghts[pnt] / 1.3)
                                     < (pos2[0][1] - lbl_hghts[pnt2] / 1.3)
                                     < (new_pos[1] + lbl_hghts[pnt] / 1.3))
                                    or ((new_pos[1] - lbl_hghts[pnt] / 1.3)
                                        < (pos2[0][1] + lbl_hghts[pnt2] / 1.3)
                                        < (new_pos[1]
                                           + lbl_hghts[pnt] / 1.3)))
                               for pnt2, pos2 in lbl_pos.items()
                               if pos2 is not None and pos2[1] == 'center')):

                    lbl_pos[pnt] = (new_pos[0], new_pos[1]), 'center'
                    pnt_boxs[pnt][0][0] = max(pnt_boxs[pnt][0][0],
                                              lbl_wdths[pnt] / 1.2)
                    pnt_boxs[pnt][0][1] = max(pnt_boxs[pnt][0][1],
                                              lbl_wdths[pnt] / 1.2)

                    pnt_boxs[pnt][1][0] = max(pnt_boxs[pnt][1][0],
                                              lbl_hghts[pnt] / 0.9)
                    pnt_boxs[pnt][1][1] = max(pnt_boxs[pnt][1][1],
                                              lbl_hghts[pnt] / 1.1)

    return {pos: lbl for pos, lbl in lbl_pos.items() if lbl}


def plot_random_comparison(auc_vals, pheno_dict, args):
    fig, (viol_ax, sctr_ax) = plt.subplots(
        figsize=(11, 7), nrows=1, ncols=2,
        gridspec_kw=dict(width_ratios=[1, 1.51])
        )

    mtype_genes = pd.Series([mtype.get_labels()[0]
                             for mtype in auc_vals.index
                             if (not isinstance(mtype, RandomType)
                                 and (mtype.subtype_list()[0][1]
                                      & copy_mtype).is_empty())])

    sbgp_genes = mtype_genes.value_counts()[
        mtype_genes.value_counts() > 1].index
    lbl_order = ['Random', 'Point w/o Sub', 'Point w/ Sub', 'Subgroupings']

    gene_stat = pd.Series({
        mtype: ('Random' if (isinstance(mtype, RandomType)
                             and mtype.base_mtype is None)
                else 'RandomGene' if isinstance(mtype, RandomType)
                else 'Point w/ Sub'
                if (mtype.subtype_list()[0][1] == pnt_mtype
                    and mtype.get_labels()[0] in sbgp_genes)
                else 'Point w/o Sub'
                if (mtype.subtype_list()[0][1] == pnt_mtype
                    and not mtype.get_labels()[0] in sbgp_genes)
                else 'Copy' if not (mtype.subtype_list()[0][1]
                                    & copy_mtype).is_empty()
                else 'Subgroupings')
        for mtype in auc_vals.index
        })

    use_aucs = auc_vals[~gene_stat.isin(['Copy', 'RandomGene'])]
    gene_stat = gene_stat[~gene_stat.isin(['Copy', 'RandomGene'])]

    sns.violinplot(x=gene_stat, y=use_aucs, ax=viol_ax, order=lbl_order,
                   palette=['0.61', *[variant_clrs['Point']] * 3],
                   cut=0, linewidth=0, width=0.93)

    viol_ax.set_xlabel('')
    viol_ax.set_ylabel('AUC', size=23, weight='semibold')
    viol_ax.set_xticklabels(lbl_order, rotation=37, ha='right', size=18)

    viol_ax.get_children()[2].set_linewidth(3.1)
    viol_ax.get_children()[4].set_linewidth(3.1)
    viol_ax.get_children()[2].set_facecolor('white')
    viol_ax.get_children()[2].set_edgecolor(variant_clrs['Point'])
    viol_ylims = viol_ax.get_ylim()

    if 'Point w/o Sub' in gene_stat:
        viol_ax.get_children()[4].set_edgecolor(variant_clrs['Point'])

    for i, lbl in enumerate(lbl_order):
        viol_ax.get_children()[i * 2].set_alpha(0.41)

        viol_ax.text(i - 0.13, viol_ax.get_ylim()[1],
                     "n={}".format((gene_stat == lbl).sum()),
                     size=15, rotation=37, ha='left', va='bottom')

    viol_xlims = viol_ax.get_xlim()
    viol_ax.plot(viol_xlims, [0.5, 0.5],
                 color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    viol_ax.plot(viol_xlims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    viol_ax.set_ylim(viol_ylims)

    size_dict = dict()
    for mtype in use_aucs.index:
        if isinstance(mtype, RandomType):
            if mtype.size_dist in size_dict:
                size_dict[mtype.size_dist] += [mtype]
            else:
                size_dict[mtype.size_dist] = [mtype]

    for mtype_stat, face_clr, edge_clr, ln_wdth in zip(
            lbl_order[1:],
            ['none', variant_clrs['Point'], variant_clrs['Point']],
            [variant_clrs['Point'], variant_clrs['Point'], 'none'],
            [1.7, 1.7, 0]
            ):

        plt_mtypes = gene_stat.index[gene_stat == mtype_stat]
        size_rtypes = [size_dict[np.sum(pheno_dict[mtype])]
                       for mtype in plt_mtypes]
        mean_vals = [701 * np.mean(pheno_dict[mtype]) for mtype in plt_mtypes]

        sctr_ax.scatter([use_aucs[plt_mtype] for plt_mtype in plt_mtypes],
                        [use_aucs[rtype].max() for rtype in size_rtypes],
                        s=mean_vals, alpha=0.11, facecolor=face_clr,
                        edgecolor=edge_clr, linewidth=ln_wdth)

    sctr_ax.plot(viol_ylims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    sctr_ax.plot([0.5, 0.5], viol_ylims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    sctr_ax.plot(viol_ylims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    sctr_ax.plot([1, 1], viol_ylims, color='black', linewidth=1.9, alpha=0.89)
    sctr_ax.plot(viol_ylims, viol_ylims,
            color='#550000', linewidth=1.7, linestyle='--', alpha=0.41)

    sctr_ax.set_xlabel("AUC of Oncogene Mutation", size=19, weight='semibold')
    sctr_ax.set_ylabel("Best AUC of\nSize-Matched Randoms",
                       size=19, weight='semibold')

    sctr_ax.set_xlim(viol_ylims)
    sctr_ax.set_ylim(viol_ylims)

    plt.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "random-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_size_comparison(auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    plot_df = pd.DataFrame({'Size': [pheno_dict[mtype].sum()
                                     for mtype in use_aucs.index],
                            'AUC': use_aucs.values,
                            'Gene': [mtype.get_labels()[0]
                                     for mtype in use_aucs.index]})

    ax.scatter(plot_df.Size, plot_df.AUC,
               c=[choose_gene_colour(gene) for gene in plot_df.Gene],
               s=23, alpha=0.17, edgecolor='none')

    size_lm = smf.ols('AUC ~ C(Gene) + Size', data=plot_df).fit()
    coef_vals = size_lm.params.sort_values()[::-1]
    gene_coefs = coef_vals[coef_vals.index.str.match('^C(Gene)*')]

    coef_lbl = '\n'.join(
        ["Size Coef: {:.3g}".format(coef_vals.Size)]
        + ["{} Coef: {:.3g}".format(gn.split("[T.")[1].split("]")[0], coef)
           for gn, coef in gene_coefs[:4].iteritems()]
        )

    ax.text(0.97, 0.07, coef_lbl,
            size=14, ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel("# of Samples Affected", size=20, weight='semibold')
    ax.set_ylabel("AUC", size=20, weight='semibold')

    ax.set_xlim([10, ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0], 1.01])

    ax.plot(ax.get_xlim(), [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot(ax.get_xlim(), [1.0, 1.0],
            color='black', linewidth=1.9, alpha=0.89)

    plt.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "size-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_sub_comparisons(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    pnt_dict = dict()
    clr_dict = dict()

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: mtype.get_labels()[0]):

        # if there were subgroupings tested for the gene, find the results
        # for the mutation representing all point mutations for this gene...
        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            # ...as well as the results for the best subgrouping of
            # mutations found for this gene
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            # if the AUC for the optimal subgrouping is good enough, plot it
            # against the AUC for all point mutations of the gene...
            if auc_vec[best_indx] > 0.6:
                clr_dict[gene] = choose_gene_colour(gene)
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                # ...and if we are sure that the optimal subgrouping AUC is
                # better than the point mutation AUC then add a label with the
                # gene name and a description of the best found subgrouping...
                if conf_sc > 0.9:
                    mtype_lbl = '\n'.join(
                        get_fancy_label(best_subtype).split('\n')[1:])

                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene, mtype_lbl))

                # ...if we are not sure but the respective AUCs are still
                # pretty great then add a label with just the gene name...
                elif auc_vec[base_indx] > 0.7 or auc_vec[best_indx] > 0.7:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene, ''))

                # ...otherwise plot the point with no label
                else:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, ('', ''))

                pie_ax = inset_axes(
                    ax, width=base_size ** 0.5, height=base_size ** 0.5,
                    bbox_to_anchor=(auc_vec[base_indx], auc_vec[best_indx]),
                    bbox_transform=ax.transData, loc=10,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                           colors=[clr_dict[gene] + (0.77, ),
                                   clr_dict[gene] + (0.29, )])

    # figure out where to place the labels for each point, and plot them
    lbl_pos = place_labels(pnt_dict)
    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=13, ha=pos[1], va='bottom')
        ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][1],
                size=9, ha=pos[1], va='top')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.plot([0.48, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [0.48, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])
    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found subgrouping", size=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_copy_comparisons(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    pnt_dict = dict()
    clr_dict = dict()

    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] == pnt_mtype
             or not (mtype.subtype_list()[0][1] & copy_mtype).is_empty())
        for mtype in auc_vals.index
        ]]

    plt_min = 0.48
    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: mtype.get_labels()[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            if auc_vec[best_indx] > 0.6:
                plt_min = min(plt_min, auc_vec[base_indx] - 0.02)

                base_size = np.mean(pheno_dict[base_mtype])
                clr_dict[gene] = choose_gene_colour(gene)
                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                if conf_sc > 0.9:
                    mtype_lbl = '\n'.join(
                        get_fancy_label(best_subtype).split('\n')[1:])

                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene, mtype_lbl))

                elif auc_vec[base_indx] > 0.7 or auc_vec[best_indx] > 0.7:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene, ''))

                else:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, ('', ''))

                ax.scatter(auc_vec[base_indx], auc_vec[best_indx],
                           s=3197 * base_size, facecolor=clr_dict[gene],
                           edgecolor='none', alpha=0.53)

    # figure out where to place the labels for each point, and plot them
    lbl_pos = place_labels(pnt_dict)
    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=13, ha=pos[1], va='bottom')
        ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][1],
                size=9, ha=pos[1], va='top')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.plot([plt_min, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [plt_min, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([plt_min + 0.007, 0.997], [plt_min + 0.007, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim([plt_min, 1.01])
    ax.set_ylim([plt_min, 1.01])

    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found CNA subgrouping",
                  size=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "copy-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_aupr_comparisons(auc_vals, pred_df, pheno_dict, conf_vals, args):
    fig, (base_ax, subg_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    pnt_dict = {'Base': dict(), 'Subg': dict()}
    clr_dict = dict()
    plt_max = 0.53

    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: mtype.get_labels()[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            if auc_vec[best_indx] > 0.6:
                clr_dict[gene] = choose_gene_colour(gene)

                base_size = 0.47 * np.mean(pheno_dict[base_mtype])
                best_prop = 0.47 * np.mean(
                    pheno_dict[best_subtype]) / base_size

                base_infr = pred_df.loc[base_mtype].apply(np.mean)
                best_infr = pred_df.loc[best_subtype].apply(np.mean)

                base_auprs = (aupr_score(pheno_dict[base_mtype], base_infr),
                              aupr_score(pheno_dict[base_mtype], best_infr))
                subg_auprs = (aupr_score(pheno_dict[best_subtype], base_infr),
                              aupr_score(pheno_dict[best_subtype], best_infr))

                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                base_lbl = '', ''
                subg_lbl = '', ''
                min_diff = np.log2(1.25)

                if conf_sc > 0.9:
                    base_lbl = gene, get_fancy_label(best_subtype)
                    subg_lbl = gene, get_fancy_label(best_subtype)

                elif auc_vec[base_indx] > 0.75 or auc_vec[best_indx] > 0.75:
                    base_lbl = gene, ''
                    subg_lbl = gene, ''

                elif auc_vec[base_indx] > 0.6 or auc_vec[best_indx] > 0.6:
                    if abs(np.log2(base_auprs[1] / base_auprs[0])) > min_diff:
                        base_lbl = gene, ''
                    if abs(np.log2(subg_auprs[1] / subg_auprs[0])) > min_diff:
                        subg_lbl = gene, ''

                pnt_dict['Base'][base_auprs] = base_size ** 0.53, base_lbl
                pnt_dict['Subg'][subg_auprs] = base_size ** 0.53, subg_lbl

                for ax, (base_aupr, subg_aupr) in zip(
                        [base_ax, subg_ax], [base_auprs, subg_auprs]):
                    plt_max = min(1.005,
                                  max(plt_max,
                                      base_aupr + 0.11, subg_aupr + 0.11))

                    pie_ax = inset_axes(
                        ax, width=base_size ** 0.5, height=base_size ** 0.5,
                        bbox_to_anchor=(base_aupr, subg_aupr),
                        bbox_transform=ax.transData, loc=10,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.29, 0],
                               colors=[clr_dict[gene] + (0.77,),
                                       clr_dict[gene] + (0.29,)])

    for ax, lbl in zip([base_ax, subg_ax], ['Base', 'Subg']):
        lbl_pos = place_labels(pnt_dict[lbl],
                               lims=(0, plt_max), lbl_dens=0.63)

        for (pnt_x, pnt_y), pos in lbl_pos.items():
            ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                    pnt_dict[lbl][pnt_x, pnt_y][1][0],
                    size=11, ha=pos[1], va='bottom')
            ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                    pnt_dict[lbl][pnt_x, pnt_y][1][1],
                    size=7, ha=pos[1], va='top')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]
            ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

            # if the label is sufficiently far away from its point...
            if ln_lngth > (0.043 + pnt_dict[lbl][pnt_x, pnt_y][0] / 19):
                use_clr = clr_dict[pnt_dict[lbl][pnt_x, pnt_y][1][0]]
                pnt_gap = pnt_dict[lbl][pnt_x, pnt_y][0] / (13 * ln_lngth)
                lbl_gap = 0.006 / ln_lngth

                ax.plot([pnt_x - pnt_gap * x_delta,
                         pos[0][0] + lbl_gap * x_delta],
                        [pnt_y - pnt_gap * y_delta,
                         pos[0][1] + lbl_gap * y_delta
                         + 0.008 + 0.004 * np.sign(y_delta)],
                        c=use_clr, linewidth=1.1, alpha=0.23)

        ax.plot([0, plt_max], [0, 0],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([0, 0], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [1, 1],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([1, 1], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [0, plt_max],
                color='#550000', linewidth=1.7, linestyle='--', alpha=0.37)

        ax.set_xlim([-0.01, plt_max])
        ax.set_ylim([-0.01, plt_max])

        ax.set_xlabel("using all point mutation inferred scores",
                      size=19, weight='semibold')
        ax.set_ylabel("using best found subgrouping inferred scores",
                      size=19, weight='semibold')

    base_ax.set_title("AUPR on all point mutations",
                      size=21, weight='semibold')
    subg_ax.set_title("AUPR on best subgrouping mutations",
                      size=21, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "aupr-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the AUCs for a particular classifier on the mutations "
        "enumerated for a given cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "trnsf-vals__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'trnsf-vals__')[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby('Levels')['Samps'].min()

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    pred_dict = dict()
    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_dict[lvls] = pickle.load(f)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_dict[lvls] = pd.DataFrame.from_dict(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_dict[lvls] = pd.DataFrame.from_dict(pickle.load(f))

    pred_df = pd.concat(pred_dict.values())
    auc_df = pd.concat(auc_dict.values())
    conf_df = pd.concat(conf_dict.values())
    assert auc_df.index.isin(phn_dict).all()

    # create the plots
    plot_random_comparison(auc_df['mean'], phn_dict, args)
    plot_size_comparison(auc_df['mean'], phn_dict, args)

    plot_sub_comparisons(auc_df['mean'], phn_dict, conf_df['mean'], args)
    plot_copy_comparisons(auc_df['mean'], phn_dict, conf_df['mean'], args)
    plot_aupr_comparisons(auc_df['mean'], pred_df, phn_dict,
                          conf_df['mean'], args)


if __name__ == '__main__':
    main()

