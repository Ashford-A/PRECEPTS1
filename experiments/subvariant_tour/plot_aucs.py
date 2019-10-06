
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

from HetMan.experiments.subvariant_tour import cis_lbls, pnt_mtype
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from HetMan.experiments.subvariant_tour.utils import (
    get_fancy_label, RandomType)
from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.mutations import MuType

import argparse
from glob import glob
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


def place_labels(pnt_dict, lims=(0.48, 1.01), lbl_dens=1.):
    lbl_pos = dict()

    lim_gap = (lbl_dens * 73) / (lims[1] - lims[0])
    pnt_gaps = {pnt: sz * lim_gap / 39 for pnt, (sz, _) in pnt_dict.items()}

    lbl_hghts = {pnt: 0.13 + lbls[1].count('\n') if lbls[1] else 0.31
                 for pnt, (_, lbls) in pnt_dict.items()}
    lbl_wdths = {pnt: (1 + max(len(ln) for ln in lbls[1].split('\n'))
                       if lbls[1] else 1 + len(lbls[0]) * 1.43)
                 for pnt, (_, lbls) in pnt_dict.items()}

    for pnt, (sz, lbls) in pnt_dict.items():
        if lbls[0]:
            lbl_pos[pnt] = None

            if (pnt[0] > (lims[0] + (lbl_wdths[pnt] / lim_gap))
                    and not any(((pnt[0] - ((pnt_gaps[pnt] + lbl_wdths[pnt])
                                            / lim_gap))
                                 < pnt2[0] + (pnt_gaps[pnt2] / lim_gap))
                                and ((pnt2[0] - (pnt_gaps[pnt2] / lim_gap))
                                     < (pnt[0] - (pnt_gaps[pnt] / lim_gap)))
                                and ((pnt[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pnt2[1] + (pnt_gaps[pnt2] / lim_gap))
                                and ((pnt2[1] - (pnt_gaps[pnt2] / lim_gap))
                                     < (pnt[1] + (2.51 / lim_gap)))
                                for pnt2 in pnt_dict if pnt2 != pnt)

                    and not any(((pnt[0] - ((pnt_gaps[pnt] + lbl_wdths[pnt])
                                            / lim_gap))
                                 < pos[0][0] + (0.89 / lim_gap))
                                and ((pos[0][0] - (lbl_wdths[pnt2] / lim_gap))
                                     < (pnt[0] - (pnt_gaps[pnt] / lim_gap)))
                                and ((pnt[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pos[0][1] + (2.51 / lim_gap))
                                and ((pos[0][1] - (lbl_hghts[pnt2] / lim_gap))
                                     < (pnt[1] + (2.51 / lim_gap)))
                                for pnt2, pos in lbl_pos.items()
                                if (pnt2 != pnt and pos is not None
                                    and pos[1] == 'right'))

                    and not any(((pnt[0] - ((pnt_gaps[pnt] + lbl_wdths[pnt])
                                            / lim_gap))
                                 < pos[0][0] + (lbl_wdths[pnt2] / lim_gap))
                                and ((pos[0][0] - (0.89 / lim_gap))
                                     < (pnt[0] - (pnt_gaps[pnt] / lim_gap)))
                                and ((pnt[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pos[0][1] + (2.51 / lim_gap))
                                and ((pos[0][1] - (lbl_hghts[pnt2] / lim_gap))
                                     < (pnt[1] + (2.51 / lim_gap)))
                                for pnt2, pos in lbl_pos.items()
                                if (pnt2 != pnt and pos is not None
                                    and pos[1] == 'left'))):

                lbl_pos[pnt] = ((pnt[0] - ((pnt_gaps[pnt] * 221)
                                           / (lim_gap ** 2.03)), pnt[1]),
                                'right')

            elif (pnt[0] < (lims[1] - (lbl_wdths[pnt] / lim_gap))
                  and not any(((pnt[0] + ((pnt_gaps[pnt] + lbl_wdths[pnt])
                                          / lim_gap))
                               > pnt2[0] - (pnt_gaps[pnt2] / lim_gap))
                              and ((pnt2[0] + (pnt_gaps[pnt2] / lim_gap))
                                   > (pnt[0] - (pnt_gaps[pnt] / lim_gap)))
                              and ((pnt[1] - (lbl_hghts[pnt] / lim_gap))
                                   < pnt2[1] + (pnt_gaps[pnt2] / lim_gap))
                              and ((pnt2[1] - (pnt_gaps[pnt2] / lim_gap))
                                   < (pnt[1] + (2.51 / lim_gap)))
                              for pnt2 in pnt_dict if pnt2 != pnt)

                    and not any(((pnt[0] + ((pnt_gaps[pnt] + lbl_wdths[pnt])
                                            / lim_gap))
                                 > pos[0][0] + (0.89 / lim_gap))
                                and ((pos[0][0] - (lbl_wdths[pnt2] / lim_gap))
                                     > (pnt[0] - (pnt_gaps[pnt] / lim_gap)))
                                and ((pnt[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pos[0][1] + (2.51 / lim_gap))
                                and ((pos[0][1] - (lbl_hghts[pnt2] / lim_gap))
                                     < (pnt[1] + (2.51 / lim_gap)))
                                for pnt2, pos in lbl_pos.items()
                                if (pnt2 != pnt and pos is not None
                                    and pos[1] == 'right'))

                    and not any(((pnt[0] - ((pnt_gaps[pnt] + lbl_wdths[pnt])
                                            / lim_gap))
                                 > pos[0][0] + (0.89 / lim_gap))
                                and ((pos[0][0] - (lbl_wdths[pnt2] / lim_gap))
                                     > (pnt[0] - (pnt_gaps[pnt] / lim_gap)))
                                and ((pnt[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pos[0][1] + (2.51 / lim_gap))
                                and ((pos[0][1] - (lbl_hghts[pnt2] / lim_gap))
                                     < (pnt[1] + (2.51 / lim_gap)))
                                for pnt2, pos in lbl_pos.items()
                                if (pnt2 != pnt and pos is not None
                                    and pos[1] == 'left'))):

                lbl_pos[pnt] = ((pnt[0] + ((pnt_gaps[pnt] * 221)
                                           / (lim_gap ** 2.03)), pnt[1]),
                                'left')

    i = 1491
    while i < 16371 and any(lbl is None for lbl in lbl_pos.values()):
        i += 1

        for pnt in tuple(pnt_dict):
            if pnt in lbl_pos and lbl_pos[pnt] is None:
                new_pos = ((i / (371 * lim_gap)) * np.random.randn(2)
                           + [pnt[0], pnt[1]])

                new_pos[0] = new_pos[0].round(5).clip(
                    lims[0] + lbl_wdths[pnt] / lim_gap,
                    lims[1] - lbl_wdths[pnt] / lim_gap
                    )
                new_pos[1] = new_pos[1].round(5).clip(
                    lims[0] + 2.19 * lbl_hghts[pnt] / lim_gap,
                    lims[1] - 3.79 / lim_gap
                    )

                new_pos[0] = new_pos[0].round(5).clip(
                    pnt[0] - (lims[1] - lims[0]) * 0.47,
                    pnt[0] + (lims[1] - lims[0]) * 0.47
                    )
                new_pos[1] = new_pos[1].round(5).clip(
                    pnt[1] - (lims[1] - lims[0]) * 0.47,
                    pnt[1] + (lims[1] - lims[0]) * 0.47
                    )

                if (not any(((new_pos[0]
                              - ((pnt_gaps[pnt] + lbl_wdths[pnt] / 1.8)
                                 / lim_gap))
                             < pnt2[0] + (pnt_gaps[pnt2] / lim_gap))
                            and ((pnt2[0] - (pnt_gaps[pnt2] / lim_gap))
                                 < (new_pos[0]
                                    + ((pnt_gaps[pnt] + lbl_wdths[pnt] / 1.8)
                                       / lim_gap)))
                            and ((new_pos[1] - (lbl_hghts[pnt] / lim_gap))
                                 < pnt2[1] + (pnt_gaps[pnt2] / lim_gap))
                            and ((pnt2[1] - (pnt_gaps[pnt2] / lim_gap))
                                 < (new_pos[1] + (3.51 / lim_gap)))
                            for pnt2 in pnt_dict if pnt2 != pnt)
                    
                    and not any(((new_pos[0]
                                  + ((pnt_gaps[pnt] + lbl_wdths[pnt] / 1.8)
                                     / lim_gap))
                                 > pos[0][0] - (lbl_wdths[pnt2] / lim_gap))
                                and ((pos[0][0] - (0.89 / lim_gap))
                                     > (new_pos[0]
                                        - (lbl_wdths[pnt] / 1.8 / lim_gap)))
                                and ((new_pos[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pos[0][1] + (3.51 / lim_gap))
                                and ((pos[0][1] - (lbl_hghts[pnt2] / lim_gap))
                                     < (new_pos[1] + (3.51 / lim_gap)))
                                for pnt2, pos in lbl_pos.items()
                                if (pnt2 != pnt and pos is not None
                                    and pos[1] in ['right', 'center']))

                    and not any(((new_pos[0]
                                  - ((pnt_gaps[pnt] + lbl_wdths[pnt] / 1.8)
                                     / lim_gap))
                                 < pos[0][0] + (lbl_wdths[pnt2] / lim_gap))
                                and ((pos[0][0] - (0.89 / lim_gap))
                                     < (new_pos[0]
                                        + (pnt_gaps[pnt] / lim_gap)))
                                and ((new_pos[1] - (lbl_hghts[pnt] / lim_gap))
                                     < pos[0][1] + (3.51 / lim_gap))
                                and ((pos[0][1] - (lbl_hghts[pnt2] / lim_gap))
                                     < (new_pos[1] + (3.51 / lim_gap)))
                                for pnt2, pos in lbl_pos.items()
                                if (pnt2 != pnt and pos is not None
                                    and pos[1] in ['left', 'center']))):

                    lbl_pos[pnt] = new_pos, 'center'

    return {pos: lbl for pos, lbl in lbl_pos.items() if lbl}


def plot_random_comparison(auc_vals, pheno_dict, args):
    fig, (viol_ax, sctr_ax) = plt.subplots(
        figsize=(11, 7), nrows=1, ncols=2,
        gridspec_kw=dict(width_ratios=[1, 1.51])
        )

    mtype_genes = pd.Series([mtype.subtype_list()[0][0]
                             for mtype in auc_vals.index
                             if len(mtype.subtype_list()) > 0])

    sbgp_genes = mtype_genes.value_counts()[
        mtype_genes.value_counts() > 1].index
    lbl_order = ['Random', 'Point w/o Sub', 'Point w/ Sub', 'Subgrouping']

    gene_stat = pd.Series({
        mtype: ('Random' if isinstance(mtype, RandomType)
                else 'Point w/ Sub'
                if (mtype.subtype_list()[0][1] == pnt_mtype
                    and mtype.get_labels()[0] in sbgp_genes)
                else 'Point w/o Sub'
                if (mtype.subtype_list()[0][1] == pnt_mtype
                    and not mtype.get_labels()[0] in sbgp_genes)
                else 'Other'
                if (mtype.subtype_list()[0][1] != pnt_mtype
                    and pheno_dict[mtype].sum() == pheno_dict[MuType(
                        {('Gene', mtype.get_labels()[0]): pnt_mtype})].sum())
                else 'Subgrouping')
        for mtype in auc_vals.index
        })

    auc_vals = auc_vals[gene_stat != 'Other']
    gene_stat = gene_stat[gene_stat != 'Other']

    sns.violinplot(x=gene_stat, y=auc_vals, ax=viol_ax, order=lbl_order,
                   palette=['0.61', *[variant_clrs['Point']] * 3],
                   cut=0, linewidth=0, width=0.93)

    viol_ax.set_xlabel('')
    viol_ax.set_ylabel('AUC', size=23, weight='semibold')
    viol_ax.set_xticklabels(lbl_order, rotation=37, ha='right', size=18)

    viol_ax.get_children()[2].set_linewidth(3.1)
    viol_ax.get_children()[4].set_linewidth(3.1)
    viol_ax.get_children()[2].set_facecolor('white')
    viol_ax.get_children()[2].set_edgecolor(variant_clrs['Point'])
    viol_ax.get_children()[4].set_edgecolor(variant_clrs['Point'])

    for i, lbl in enumerate(lbl_order):
        viol_ax.get_children()[i * 2].set_alpha(0.41)

        viol_ax.text(i - 0.13, viol_ax.get_ylim()[1],
                     "n={}".format((gene_stat == lbl).sum()),
                     size=15, rotation=37, ha='left', va='bottom')

    size_dict = dict()
    for mtype in auc_vals.index:
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

        sctr_ax.scatter([auc_vals[plt_mtype] for plt_mtype in plt_mtypes],
                        [auc_vals[rtype].max() for rtype in size_rtypes],
                        s=mean_vals, alpha=0.11, facecolor=face_clr,
                        edgecolor=edge_clr, linewidth=ln_wdth)

    sctr_ax.set_xlabel('AUC of Oncogene Mutation', size=20, weight='semibold')
    sctr_ax.set_ylabel('AUC of Size-Matched\nRandom Sample Set',
                       size=20, weight='semibold')

    sctr_ax.set_xlim(viol_ax.get_ylim())
    sctr_ax.set_ylim(viol_ax.get_ylim())
    sctr_ax.plot(viol_ax.get_ylim(), viol_ax.get_ylim(),
                 linewidth=1.7, linestyle='--', color='#550000', alpha=0.41)

    plt.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "random-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_size_comparison(auc_vals, pheno_dict, clr_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    mtype_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and not (mtype.subtype_list()[0][1] != pnt_mtype
                 and pheno_dict[mtype].sum() == pheno_dict[MuType(
                     {('Gene', mtype.get_labels()[0]): pnt_mtype})].sum())
        for mtype in auc_vals.index
        ]]

    plot_df = pd.DataFrame({
        'Size': [pheno_dict[mtype].sum() for mtype in mtype_aucs.index],
        'AUC': mtype_aucs.values,
        'Gene': [mtype.get_labels()[0] for mtype in mtype_aucs.index]
        })

    clr_vals = [clr_dict[mtype.get_labels()[0]] for mtype in mtype_aucs.index]
    ax.scatter(plot_df.Size, plot_df.AUC,
               c=[clr_dict[gn] for gn in plot_df.Gene],
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


def plot_sub_comparisons(auc_vals, pheno_dict, conf_df, clr_dict, args):
    fig, ax = plt.subplots(figsize=(11, 11))
    pnt_dict = dict()

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    auc_vals = auc_vals[[
        not isinstance(mtype, RandomType)
        and not (mtype.subtype_list()[0][1] != pnt_mtype
                 and pheno_dict[mtype].sum() == pheno_dict[MuType(
                     {('Gene', mtype.get_labels()[0]): pnt_mtype})].sum())
        for mtype in auc_vals.index
        ]]

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for gene, auc_vec in auc_vals.groupby(
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
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                conf_sc = np.greater.outer(
                    conf_df.loc[best_subtype].values[0],
                    conf_df.loc[base_mtype].values[0]
                    ).mean()

                # ...and if it is really good then add a label with the gene
                # name and a description of the best found subgrouping
                if conf_sc > 0.9:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene,
                                            get_fancy_label(best_subtype))
                        )

                elif auc_vec[base_indx] > 0.7 or auc_vec[best_indx] > 0.7:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene, ''))

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
                           colors=[clr_dict[gene] + (0.77,),
                                   clr_dict[gene] + (0.29,)])

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
        if ln_lngth > (0.013 + pnt_dict[pnt_x, pnt_y][0] / 31):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])
    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found subgrouping", size=23, weight='semibold')

    ax.plot([0.48, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [0.48, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_aupr_comparisons(auc_vals, infer_df,
                          pheno_dict, conf_df, clr_dict, args):
    fig, ax = plt.subplots(figsize=(11, 11))
    pnt_dict = dict()

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    auc_vals = auc_vals[[
        not isinstance(mtype, RandomType)
        and not (mtype.subtype_list()[0][1] != pnt_mtype
                 and pheno_dict[mtype].sum() == pheno_dict[MuType(
                     {('Gene', mtype.get_labels()[0]): pnt_mtype})].sum())
        for mtype in auc_vals.index
        ]]

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for gene, auc_vec in auc_vals.groupby(
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

            base_infr = infer_df.loc[base_mtype].apply(np.mean)
            best_infr = infer_df.loc[best_subtype].apply(np.mean)
            base_aupr = aupr_score(pheno_dict[base_mtype], base_infr)
            best_aupr = aupr_score(pheno_dict[best_subtype], best_infr)

            # if the AUC for the optimal subgrouping is good enough, plot it
            # against the AUC for all point mutations of the gene...
            if auc_vec[best_indx] > 0.6:
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                conf_sc = np.greater.outer(
                    conf_df.loc[best_subtype].values[0],
                    conf_df.loc[base_mtype].values[0]
                    ).mean()

                # ...and if it is really good then add a label with the gene
                # name and a description of the best found subgrouping
                if conf_sc > 0.9:
                    pnt_dict[base_aupr, best_aupr] = (
                        base_size ** 0.53, (gene,
                                            get_fancy_label(best_subtype))
                        )

                elif auc_vec[base_indx] > 0.7 or auc_vec[best_indx] > 0.7:
                    pnt_dict[base_aupr, best_aupr] = (base_size ** 0.53,
                                                      (gene, ''))

                else:
                    pnt_dict[base_aupr, best_aupr] = (base_size ** 0.53,
                                                      ('', ''))

                pie_ax = inset_axes(
                    ax, width=base_size ** 0.5, height=base_size ** 0.5,
                    bbox_to_anchor=(base_aupr, best_aupr),
                    bbox_transform=ax.transData, loc=10,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                           colors=[clr_dict[gene] + (0.77,),
                                   clr_dict[gene] + (0.29,)])

    plt_lims = (max(min(min(xval for xval, _ in pnt_dict),
                        min(yval for _, yval in pnt_dict)) - 0.11, -0.01),
                min(max(max(xval for xval, _ in pnt_dict),
                        max(yval for _, yval in pnt_dict)) + 0.11, 1.01))

    lbl_pos = place_labels(pnt_dict, lims=plt_lims)
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
        if ln_lngth > (0.043 + pnt_dict[pnt_x, pnt_y][0] / 19):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (13 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    ax.set_xlabel("point mutation AUPR"
                  "\nusing point mutation inferred scores",
                  size=23, weight='semibold')
    ax.set_ylabel("point mutation AUPR"
                  "\nusing best found subgrouping inferred scores",
                  size=23, weight='semibold')

    ax.plot(plt_lims, [0, 0], color='black', linewidth=1.7, alpha=0.89)
    ax.plot([0, 0], plt_lims, color='black', linewidth=1.7, alpha=0.89)
    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.7, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.7, alpha=0.89)

    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

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

    parser.add_argument(
        '--seed', default=3401, type=int,
        help="the random seed to use for setting plotting colours"
        )

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-conf__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif)
            )
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-conf__')[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby(['Levels'])['Samps'].min()

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    infer_dict = dict()
    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-data__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            infer_dict[lvls] = pickle.load(f)['Infer']['Chrm']

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_dict[lvls] = pickle.load(f)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_dict[lvls] = pickle.load(f)

    auc_dfs = {cis_lbl: pd.concat([auc_df[cis_lbl]
                                   for auc_df in auc_dict.values()])
               for cis_lbl in cis_lbls}

    for cis_lbl in cis_lbls:
        assert auc_dfs[cis_lbl].index.isin(phn_dict).all()

    infer_df = pd.concat(infer_dict.values())
    conf_dfs = {cis_lbl: pd.concat([conf_df[cis_lbl]
                                    for conf_df in conf_dict.values()])
                for cis_lbl in cis_lbls}

    np.random.seed(args.seed)
    clr_dict = {gene: hls_to_rgb(h=np.random.uniform(size=1)[0], l=0.5, s=0.8)
                for gene in sorted({mtype.get_labels()[0]
                                    for mtype in auc_dfs['Chrm'].index
                                    if not isinstance(mtype, RandomType)})}

    plot_random_comparison(auc_dfs['Chrm'], phn_dict, args)
    plot_size_comparison(auc_dfs['Chrm'], phn_dict, clr_dict, args)

    plot_sub_comparisons(auc_dfs['Chrm'], phn_dict, conf_dfs['Chrm'],
                         clr_dict, args)
    plot_aupr_comparisons(auc_dfs['Chrm'], infer_df,
                          phn_dict, conf_dfs['Chrm'], clr_dict, args)


if __name__ == '__main__':
    main()

