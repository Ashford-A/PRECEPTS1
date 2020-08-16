"""
Plots diagrams demonstrating how the overlapping presence other types of
mutations of the same gene can affect a mutation classification task.
"""

from ..utilities.mutations import (pnt_mtype, copy_mtype, shal_mtype,
                                   dels_mtype, gains_mtype, Mcomb, ExMcomb)
from dryadic.features.mutations import MuType

from ..utilities.labels import get_fancy_label
from ..utilities.metrics import calc_auc
from ..utilities.colour_maps import simil_cmap
from ..subvariant_test import variant_clrs
from ..subvariant_test.utils import get_cohort_label

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as ptchs
from matplotlib.colorbar import ColorbarBase
from matplotlib import colors

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'example')


def plot_base_classification(plt_mtype, plt_mcomb, pred_df,
                             pheno_dict, cdata, args):
    fig, (coh_ax, clf_ax, ovp_ax) = plt.subplots(
        figsize=(5, 8), nrows=3, ncols=1,
        gridspec_kw=dict(height_ratios=[1, 3, 3])
        )

    plt_df = pd.DataFrame({
        'Value': pred_df.loc[
            plt_mtype, cdata.get_train_samples()].apply(np.mean),
        'cStat': pheno_dict[plt_mtype],
        'rStat': np.array(cdata.train_pheno(plt_mcomb.not_mtype))
        })

    mut_prop = np.sum(plt_df.cStat) / len(cdata.get_samples())
    ovlp_prop = np.mean(~plt_df.rStat[~plt_df.cStat]) * (1 - mut_prop)
    cur_gene = tuple(plt_mtype.label_iter())[0]

    mtype_lbl = get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1])
    mtype_tbox = '\n'.join([
        cur_gene,
        get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1],
                        phrase_link='\n')
        ])

    for ax in coh_ax, clf_ax, ovp_ax:
        ax.axis('off')

    coh_lbl = get_cohort_label(args.cohort)
    if args.expr_source == 'Firehose':
        coh_lbl = "TCGA-{}".format(coh_lbl)

    coh_ax.text(0.66, 1,
                "{}\n({} samples)".format(coh_lbl, len(cdata.get_samples())),
                size=12, ha='center', va='top')

    coh_ax.add_patch(ptchs.FancyArrowPatch(
        arrowstyle=ptchs.ArrowStyle('-[', lengthB=6.1, widthB=100),
        posA=(0.66, 0.57), posB=(0.66, 0.39), color='black', lw=0.61,
        alpha=1., clip_on=False
        ))

    coh_ax.add_patch(ptchs.Rectangle((0.33, 0.15),
                                     (1 - mut_prop) * 0.66, 0.22,
                                     facecolor=variant_clrs['WT'],
                                     alpha=0.41, hatch='/', linewidth=1.3,
                                     edgecolor='0.51'))
    coh_ax.add_patch(ptchs.Rectangle((0.33 + (1 - mut_prop) * 0.66, 0.15),
                                     mut_prop * 0.66, 0.22,
                                     facecolor=variant_clrs['Point'],
                                     alpha=0.41, hatch='/', linewidth=1.3,
                                     edgecolor='0.51'))

    coh_ax.text(0.29, 0.36, "mutated status for:",
                size=11, ha='right', va='bottom')
    coh_ax.text(0.29, 0.27, mtype_tbox,
                fontweight='bold', size=11, ha='right', va='top')

    coh_ax.add_patch(ptchs.Rectangle((0.33 + ovlp_prop * 0.66, 0.15),
                                     np.mean(plt_df.rStat) * 0.66, 0.22,
                                     hatch='\\', linewidth=1.3, clip_on=False,
                                     edgecolor='0.51', facecolor='None'))

    coh_ax.add_patch(ptchs.Rectangle((0.33 + ovlp_prop * 0.66, -0.12),
                                     np.mean(plt_df.rStat) * 0.66, 0.22,
                                     alpha=0.83, hatch='\\', clip_on=False,
                                     linewidth=1.3, edgecolor='0.51',
                                     facecolor=variant_clrs['Point']))

    coh_ax.text(0.32 + ovlp_prop * 0.66, 0.1,
                "{} mutations\nother than\n{}".format(
                    cur_gene, '\n'.join(mtype_tbox.split('\n')[1:])),
                color=variant_clrs['Point'], size=9, ha='right', va='top')

    coh_ax.text(0.33 + ovlp_prop * 0.66 + np.mean(plt_df.rStat) * 0.33, -0.15,
                "({} samples)".format(np.sum(plt_df.rStat)),
                color=variant_clrs['Point'], size=8, ha='center', va='top')

    diag_ax1 = clf_ax.inset_axes(bounds=(0, 0, 0.73, 1))
    vio_ax1 = clf_ax.inset_axes(bounds=(0.73, 0, 0.27, 1))
    diag_ax2 = ovp_ax.inset_axes(bounds=(0, 0, 0.73, 1))
    vio_ax2 = ovp_ax.inset_axes(bounds=(0.73, -0.11, 0.27, 0.89))

    for diag_ax in diag_ax1, diag_ax2:
        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

        diag_ax.add_patch(ptchs.FancyArrow(
            0.95, 0.57, dx=0.14, dy=0, width=0.04,
            length_includes_head=True, head_length=0.06, linewidth=1.7,
            facecolor='white', edgecolor='black', clip_on=False
            ))

    diag_ax1.add_patch(ptchs.Circle((0.53, 0.95), radius=0.19,
                                    facecolor=variant_clrs['Point'],
                                    alpha=0.41, clip_on=False))
    diag_ax1.text(0.53, 0.95,
                  "{}\nmutants".format(mtype_tbox, np.sum(plt_df.cStat)),
                  size=9, ha='center', va='center')

    diag_ax1.add_patch(ptchs.Circle((0.53, 0.33), radius=0.37,
                                    facecolor=variant_clrs['WT'],
                                    alpha=0.41, clip_on=False))
    diag_ax1.text(0.53, 0.33,
                  "{}\nwild-types".format(mtype_tbox, np.sum(~plt_df.cStat)),
                  size=13, ha='center', va='center')

    diag_ax1.text(0.23, 0.75, "{} mut samples".format(np.sum(plt_df.cStat)),
                  color='red', size=11, fontstyle='italic',
                  ha='right', va='bottom')
    diag_ax1.text(0.23, 0.71, "{} wt samples".format(np.sum(~plt_df.cStat)),
                  color='red', size=11, fontstyle='italic',
                  ha='right', va='top')

    diag_ax1.text(0.97, 0.89, "predict\nmutated\nstatus", color='red',
                  size=11, fontstyle='italic', ha='center', va='center')
    diag_ax1.axhline(y=0.73, xmin=0.19, xmax=0.89, color='red',
                     linestyle='--', linewidth=2.7, alpha=0.83)

    sns.violinplot(data=plt_df[~plt_df.cStat], y='Value', ax=vio_ax1,
                   palette=[variant_clrs['WT']], linewidth=0, cut=0)
    sns.violinplot(data=plt_df[plt_df.cStat], y='Value', ax=vio_ax1,
                   palette=[variant_clrs['Point']], linewidth=0, cut=0)

    vio_ax1.text(0.5, 113 / 111,
                 "AUC: {:.3f}".format(calc_auc(plt_df.Value.values,
                                               plt_df.cStat)),
                 color='red', size=15, fontstyle='italic',
                 ha='center', va='bottom', transform=vio_ax1.transAxes)

    diag_ax2.add_patch(ptchs.Wedge((0.51, 0.95), 0.19, 90, 270,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51', clip_on=False))

    diag_ax2.add_patch(ptchs.Wedge((0.55, 0.95), 0.19, 270, 90,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51', clip_on=False))
    diag_ax2.add_patch(ptchs.Wedge((0.55, 0.95), 0.19, 270, 90,
                                   facecolor='None', edgecolor='0.61',
                                   hatch='\\', linewidth=0.8, clip_on=False))

    diag_ax2.text(0.25, 0.7, "same\nclassifier\nresults", color='red',
                  size=11, fontstyle='italic', ha='right', va='center')
    diag_ax2.axhline(y=0.7, xmin=0.27, xmax=0.83, color='red',
                     linestyle='--', linewidth=1.7, alpha=0.67)

    diag_ax2.add_patch(ptchs.Wedge((0.51, 0.31), 0.33, 90, 270,
                                   facecolor=variant_clrs['WT'],
                                   edgecolor='0.51', hatch='/', alpha=0.41,
                                   linewidth=0.8, clip_on=False))

    diag_ax2.add_patch(ptchs.Wedge((0.55, 0.31), 0.33, 270, 90,
                                   facecolor=variant_clrs['WT'],
                                   edgecolor='0.51', hatch='/', alpha=0.41,
                                   linewidth=0.8, clip_on=False))
    diag_ax2.add_patch(ptchs.Wedge((0.55, 0.31), 0.33, 270, 90,
                                   facecolor='None', edgecolor='0.61',
                                   hatch='\\', alpha=0.41, linewidth=0.8,
                                   clip_on=False))

    diag_ax2.text(-0.37, 0.95, "{}\nmutants".format(mtype_tbox),
                  size=11, ha='left', va='center', clip_on=False)
    diag_ax2.text(-0.37, 0.32, "{}\nwild-types".format(mtype_tbox),
                  size=11, ha='left', va='center', clip_on=False)

    diag_ax2.text(0.5, 0.95,
                  "wild-type for\nother {}\nmutations\n({} samps)".format(
                      cur_gene, np.sum(plt_df.cStat & ~plt_df.rStat)),
                  size=8, ha='right', va='center')

    diag_ax2.text(0.56, 0.95,
                  "mutant for\nother {}\nmutations\n({} samps)".format(
                      cur_gene, np.sum(plt_df.cStat & plt_df.rStat)),
                  size=8, ha='left', va='center')

    diag_ax2.text(0.5, 0.32,
                  "wild-type for\nother {}\nmutations\n({} samps)".format(
                      cur_gene, np.sum(~plt_df.cStat & ~plt_df.rStat)),
                  size=10, ha='right', va='center')
    diag_ax2.text(0.56, 0.32,
                  "mutant for\nother {}\nmutations\n({} samps)".format(
                      cur_gene, np.sum(~plt_df.cStat & plt_df.rStat)),
                  size=10, ha='left', va='center')

    sns.violinplot(data=plt_df[~plt_df.cStat], x='cStat', y='Value',
                   hue='rStat', palette=[variant_clrs['WT']],
                   hue_order=[False, True], split=True, linewidth=0,
                   cut=0, ax=vio_ax2)
    sns.violinplot(data=plt_df[plt_df.cStat], x='cStat', y='Value',
                   hue='rStat', palette=[variant_clrs['Point']],
                   hue_order=[False, True], split=True, linewidth=0,
                   cut=0, ax=vio_ax2)

    vio_ax2.get_legend().remove()
    diag_ax2.axvline(x=0.53, ymin=-0.04, ymax=1.17, clip_on=False,
                     color=variant_clrs['Point'], linewidth=1.41, alpha=0.91,
                     linestyle=':')

    diag_ax2.text(0.53, -0.06,
                  "partition scored samples according\nto overlap with "
                  "{} mutations\nthat are not {}".format(cur_gene, mtype_lbl),
                  color=variant_clrs['Point'], size=11,
                  fontstyle='italic', ha='center', va='top')

    for vio_ax in vio_ax1, vio_ax2:
        vio_ax.set_xticks([])
        vio_ax.set_xticklabels([])
        vio_ax.set_yticklabels([])
        vio_ax.xaxis.label.set_visible(False)
        vio_ax.yaxis.label.set_visible(False)

    vio_ax1.get_children()[0].set_alpha(0.41)
    vio_ax1.get_children()[2].set_alpha(0.41)
    for i in [0, 1, 3, 4]:
        vio_ax2.get_children()[i].set_alpha(0.41)

    for i in [0, 3]:
        vio_ax2.get_children()[i].set_linewidth(0.8)
        vio_ax2.get_children()[i].set_hatch('/')
        vio_ax2.get_children()[i].set_edgecolor('0.61')

    for i in [1, 4]:
        vio_ax2.get_children()[i].set_linewidth(1.0)
        vio_ax2.get_children()[i].set_hatch('/\\')
        vio_ax2.get_children()[i].set_edgecolor('0.47')

    vio_ax2.text(0.45, 1.23, "without\noverlap",
                 color=variant_clrs['Point'], size=10,
                 fontstyle='italic', ha='right', va='bottom',
                 transform=vio_ax2.transAxes)

    vio_ax2.text(
        0.45, 113 / 111,
        "AUC:\n{:.3f}".format(calc_auc(plt_df.Value[~plt_df.rStat].values,
                                       plt_df.cStat[~plt_df.rStat])),
        color='red', size=12, fontstyle='italic',
        ha='right', va='bottom', transform=vio_ax2.transAxes
        )

    vio_ax2.text(0.55, 1.23, "with\noverlap",
                 color=variant_clrs['Point'], size=10,
                 fontstyle='italic', ha='left', va='bottom',
                 transform=vio_ax2.transAxes)

    vio_ax2.text(
        0.55, 113 / 111,
        "AUC:\n{:.3f}".format(calc_auc(plt_df.Value[plt_df.rStat].values,
                                       plt_df.cStat[plt_df.rStat])),
        color='red', size=12, fontstyle='italic',
        ha='left', va='bottom', transform=vio_ax2.transAxes
        )

    plt.tight_layout(pad=1, w_pad=0, h_pad=1)
    plt.savefig(
        os.path.join(plot_dir, args.expr_source,
                     "{}__base-classification_{}.svg".format(
                         args.cohort, cur_gene)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_classification(plt_mtype, plt_mcomb, pred_dfs,
                            pheno_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

    rest_stat = np.array(cdata.train_pheno(plt_mcomb.not_mtype))
    plt_muts = {'All': plt_mtype, 'Ex': plt_mcomb}
    cur_gene = tuple(plt_mtype.label_iter())[0]

    mcomb_masks = [('All', {lbl: np.array([True] * len(cdata.get_samples()))
                            for lbl in plt_muts}),
                   ('Iso', {lbl: ~(rest_stat & ~pheno_dict[mut])
                            for lbl, mut in plt_muts.items()})]

    mtype_lbl = get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1],
                                phrase_link='\n')
    mtype_tbox = '\n'.join([
        cur_gene,
        get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1],
                        phrase_link='\n')
        ])

    lbl_dict = {(0, 0): 'base', (0, 1): 'excl',
                (1, 0): 'isol', (1, 1): 'eiso'}
    title_dict = {'base': "Default Scenario", 'excl': "Exclusion Scenario",
                  'isol': "Isolation Scenario", 'eiso': "Exclusive Isolation"}

    wdg_props = {'mut-ovlp': {diag_lbl: {'pos': (0.52, 0.87),
                                         'clr': variant_clrs['Point']}
                              for diag_lbl in ['base', 'isol']},

                 'wt-nvlp': {diag_lbl: {'pos': (0.48, 0.29),
                                         'clr': variant_clrs['WT']}
                             for diag_lbl in ['base', 'isol', 'eiso']},

                 'wt-ovlp': {'base': {'pos': (0.52, 0.29),
                                      'clr': variant_clrs['WT']}}}

    wdg_props['mut-ovlp']['excl'] = {'pos': (0.78, 0.29),
                                     'clr': variant_clrs['WT']}
    wdg_props['wt-nvlp']['excl'] = {'pos': (0.32, 0.29),
                                    'clr': variant_clrs['WT']}
    wdg_props['wt-ovlp']['excl'] = {'pos': (0.38, 0.29),
                                    'clr': variant_clrs['WT']}

    for i, (smp_lbl, msk) in enumerate(mcomb_masks):
        for j, (mut_lbl, mut) in enumerate(plt_muts.items()):
            diag_ax = axarr[i, j].inset_axes(bounds=(0, 0, 0.79, 1))
            vio_ax = axarr[i, j].inset_axes(bounds=(0.79, 0.09, 0.21, 0.82))

            axarr[i, j].axis('off')
            diag_ax.axis('off')
            diag_ax.set_aspect('equal')

            diag_lbl = lbl_dict[i, j]
            diag_ax.text(0.5, 1.09, title_dict[diag_lbl],
                         size=21, fontweight='semibold',
                         ha='center', va='bottom')

            plt_df = pd.DataFrame({
                'Value': pred_dfs[smp_lbl].loc[
                    mut, cdata.get_train_samples()].apply(np.mean),
                'cStat': pheno_dict[mut], 'uStat': msk[mut_lbl]
                })

            if diag_lbl == 'base':
                txt_dict = {
                    'mut-nvlp': "{}\nmutant\nw/o overlap\n({} samps)".format(
                        mtype_tbox, np.sum(plt_df.cStat & ~rest_stat)),

                    'mut-ovlp': "{}\nmutant\nw/ overlap\n({} samps)".format(
                        mtype_tbox, np.sum(plt_df.cStat & rest_stat)),

                    'wt-nvlp': ("{}\nwild-type\nw/o overlap"
                                "\n({} samps)").format(
                                    mtype_tbox,
                                    np.sum(~plt_df.cStat & ~rest_stat)
                                    ),

                    'wt-ovlp': "{}\nwild-type\nw/ overlap\n({} samps)".format(
                            mtype_tbox, np.sum(~plt_df.cStat & rest_stat))
                    }

            else:
                txt_dict = {
                    mut_lbl: mut_str.format(str(mtype_lbl), cur_gene)
                    for mut_lbl, mut_str in zip(
                        ['mut-nvlp', 'mut-ovlp', 'wt-nvlp', 'wt-ovlp'],
                        ["{}\nmut\n&\nother\n{}\nwt",
                         "{}\nmut\n&\nother\n{}\nmut",
                         "{}\nwt\n&\nother\n{}\nwt",
                         "{}\nwt\n&\nother\n{}\nmut"]
                        )
                    }

            diag_ax.text(0.2, 0.65, "predict\nmutated\nstatus",
                         color='red', size=12, fontstyle='italic',
                         ha='right', va='center')
            diag_ax.axhline(y=0.65, xmin=0.22, xmax=0.83, color='red',
                            linestyle='--', linewidth=1.9, alpha=0.83)

            diag_ax.text(0.82, 0.66,
                         "{} (+)".format(np.sum(plt_df.cStat[plt_df.uStat])),
                         color='red', size=10, fontstyle='italic',
                         ha='right', va='bottom')

            diag_ax.text(
                0.82, 0.635,
                "{} (\u2212)".format(np.sum(~plt_df.cStat[plt_df.uStat])),
                color='red', size=10, fontstyle='italic', ha='right', va='top'
                )

            diag_ax.add_patch(ptchs.Wedge((0.48, 0.87), 0.17, 90, 270,
                                          facecolor=variant_clrs['Point'],
                                          alpha=0.41, clip_on=False))
            diag_ax.text(0.47, 0.87, txt_dict['mut-nvlp'],
                         size=8, ha='right', va='center')

            if diag_lbl in wdg_props['mut-ovlp']:
                wdg_pos = wdg_props['mut-ovlp'][diag_lbl]['pos']
                wdg_clr = wdg_props['mut-ovlp'][diag_lbl]['clr']

                diag_ax.add_patch(ptchs.Wedge(wdg_pos, 0.17, 270, 90,
                                              facecolor=wdg_clr,
                                              alpha=0.41, clip_on=False))
                diag_ax.text(wdg_pos[0] + 0.01, wdg_pos[1],
                             txt_dict['mut-ovlp'],
                             size=8, ha='left', va='center')

            if diag_lbl in wdg_props['wt-nvlp']:
                wdg_pos = wdg_props['wt-nvlp'][diag_lbl]['pos']
                wdg_clr = wdg_props['wt-nvlp'][diag_lbl]['clr']

                diag_ax.add_patch(ptchs.Wedge(wdg_pos, 0.31, 90, 270,
                                              facecolor=wdg_clr,
                                              alpha=0.41, clip_on=False))
                diag_ax.text(wdg_pos[0] - 0.01, wdg_pos[1],
                             txt_dict['wt-nvlp'],
                             size=11, ha='right', va='center')

            if diag_lbl in wdg_props['wt-ovlp']:
                wdg_pos = wdg_props['wt-ovlp'][diag_lbl]['pos']
                wdg_clr = wdg_props['wt-ovlp'][diag_lbl]['clr']

                diag_ax.add_patch(ptchs.Wedge(wdg_pos, 0.31, 270, 90,
                                              facecolor=wdg_clr,
                                              alpha=0.41, clip_on=False))
                diag_ax.text(wdg_pos[0] + 0.01, 0.29,
                             txt_dict['wt-ovlp'],
                             size=11, ha='left', va='center')

            diag_ax.add_patch(ptchs.FancyArrow(
                0.83, 0.51, dx=0.11, dy=0, width=0.03,
                length_includes_head=True, head_length=0.047, linewidth=1.7,
                facecolor='white', edgecolor='black', clip_on=False
                ))

            vio_ax.set_xticks([])
            vio_ax.set_xticklabels([])
            vio_ax.set_yticklabels([])
            vio_ax.yaxis.label.set_visible(False)

            sns.violinplot(data=plt_df.loc[~plt_df.cStat].loc[plt_df.uStat],
                           y='Value', ax=vio_ax, palette=[variant_clrs['WT']],
                           linewidth=0, cut=0)
            sns.violinplot(data=plt_df.loc[plt_df.cStat].loc[plt_df.uStat],
                           y='Value', ax=vio_ax,
                           palette=[variant_clrs['Point']],
                           linewidth=0, cut=0)

            vio_ax.text(0.5, 1.02,
                        "AUC: {:.3f}".format(
                            calc_auc(plt_df.Value[plt_df.uStat].values,
                                     plt_df.cStat[plt_df.uStat])
                            ),
                        color='red', size=13, fontstyle='italic',
                        ha='center', va='bottom', transform=vio_ax.transAxes)

            vio_ax.get_children()[0].set_alpha(0.41)
            vio_ax.get_children()[2].set_alpha(0.41)

    plt.tight_layout(pad=0, w_pad=1.1, h_pad=2.9)
    plt.savefig(
        os.path.join(plot_dir, args.expr_source,
                     "{}__iso-classification_{}.svg".format(
                         args.cohort, cur_gene)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_projection(plt_mtype, plt_mcomb, pred_df,
                        pheno_dict, cdata, args):
    fig, ((base_ax, pnt_ax), (loss_ax, gain_ax)) = plt.subplots(
        figsize=(11, 9), nrows=2, ncols=2)

    cur_gene = tuple(plt_mtype.label_iter())[0]
    mtype_tbox = '\n'.join([
        cur_gene,
        get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1],
                        phrase_link='\n')
        ])

    rest_stat = np.array(cdata.train_pheno(plt_mcomb.not_mtype))
    vals_df = pd.DataFrame({
        'Value': pred_df.loc[
            plt_mcomb, cdata.get_train_samples()].apply(np.mean),
        'mStat': pheno_dict[plt_mtype], 'eStat': rest_stat, 'dummy': 0
        })

    base_diag_ax = base_ax.inset_axes(bounds=(0, 0, 0.63, 0.9))
    base_vio_ax = base_ax.inset_axes(bounds=(0.63, 0, 0.37, 0.8))
    base_diag_ax.axis('off')
    base_diag_ax.set_aspect('equal')

    base_ax.text(0.1, 0.95, "1) train\nclassifier", color='red',
                 size=15, fontstyle='italic', ha='center', va='top')

    for ax in base_ax, pnt_ax, loss_ax, gain_ax:
        ax.set_aspect('equal')
        ax.axis('off')

        ax.text(0.42, 0.95, "2) apply classifier\nto held-out\nsamples",
                color='red', size=15, fontstyle='italic',
                ha='center', va='top')
        ax.text(0.83, 0.95, "3) measure\nperformance", color='red',
                size=15, fontstyle='italic', ha='center', va='top')

        ax.add_patch(ptchs.FancyArrow(
            0.51, 0.47, dx=0.08, dy=0, width=0.02,
            length_includes_head=True, head_length=0.038, alpha=0.93,
            linewidth=1.7, facecolor='none', edgecolor='black', zorder=100
            ))

    base_diag_ax.axhline(y=0.61, xmin=0.07, xmax=0.38, color='red',
                         alpha=0.91, linestyle='--', linewidth=2.3)
    base_diag_ax.axhline(y=0.61, xmin=0.44, xmax=0.72, color='red',
                         alpha=0.91, linestyle='--', linewidth=1.3)
 
    base_diag_ax.add_patch(ptchs.Wedge((0.35, 0.8), 0.13, 90, 270,
                                       facecolor=variant_clrs['Point'],
                                       alpha=0.41))
    base_diag_ax.text(0.34, 0.8,
                      "mutant for:\n{}\nw/o overlap\n({} samps)".format(
                          mtype_tbox, np.sum(vals_df.mStat & ~vals_df.eStat)),
                      size=9, ha='right', va='center')

    base_diag_ax.add_patch(ptchs.Wedge((0.35, 0.28), 0.27, 90, 270,
                                       facecolor=variant_clrs['WT'],
                                       alpha=0.41))
    base_diag_ax.text(
        0.34, 0.28,
        "wild-type for:\n{}\nw/o overlap\n({} samps)".format(
            mtype_tbox, np.sum(~vals_df.mStat & ~vals_df.eStat)),
        size=11, ha='right', va='center'
        )

    base_diag_ax.add_patch(ptchs.Wedge((0.47, 0.8), 0.13, 270, 90,
                                       facecolor=variant_clrs['Point'],
                                       edgecolor='0.53', hatch='\\',
                                       alpha=0.41, linewidth=1.7))

    base_diag_ax.text(0.48, 0.8,
                      "mutant for:\n{}\nw/ overlap\n({} samps)".format(
                          mtype_tbox, np.sum(vals_df.mStat & vals_df.eStat)),
                      size=9, ha='left', va='center')

    base_diag_ax.add_patch(ptchs.Wedge((0.47, 0.28), 0.27, 270, 90,
                                       facecolor=variant_clrs['WT'],
                                       edgecolor='0.59', hatch='\\',
                                       alpha=0.41, linewidth=1.7))

    base_diag_ax.text(0.48, 0.28,
                      "wild-type for:\n{}\nw/ overlap\n({} samps)".format(
                          mtype_tbox, np.sum(~vals_df.mStat & vals_df.eStat)),
                      size=11, ha='left', va='center')

    sns.violinplot(data=vals_df[~vals_df.mStat],
                   x='dummy', y='Value', hue='eStat',
                   palette=[variant_clrs['WT']], hue_order=[False, True],
                   split=True, linewidth=0, cut=0, ax=base_vio_ax)
    sns.violinplot(data=vals_df[vals_df.mStat],
                   x='dummy', y='Value', hue='eStat',
                   palette=[variant_clrs['Point']], hue_order=[False, True],
                   split=True, linewidth=0, cut=0, ax=base_vio_ax)

    for i in [0, 1, 3, 4]:
        base_vio_ax.get_children()[i].set_alpha(0.41)

    for i in [1, 4]:
        base_vio_ax.get_children()[i].set_linewidth(2.9)
        base_vio_ax.get_children()[i].set_edgecolor('0.59')
        base_vio_ax.get_children()[i].set_hatch('\\')

    for x_pos, use_stat in zip([0.17, 0.83], [~vals_df.eStat, vals_df.eStat]):
        base_vio_ax.text(x=x_pos, y=0.95, s="AUC", color='red', size=12,
                         fontstyle='italic', ha='center', va='bottom',
                         transform=base_vio_ax.transAxes)

        base_vio_ax.text(
            x_pos, 0.95, calc_auc(vals_df.Value[use_stat].values,
                                  vals_df.mStat[use_stat]).round(3),
            color='red', size=15, fontweight='bold',
            ha='center', va='top', transform=base_vio_ax.transAxes
            )

    base_vio_ax.set_xticks([])
    base_vio_ax.set_xticklabels([])
    base_vio_ax.set_yticklabels([])
    base_vio_ax.xaxis.label.set_visible(False)
    base_vio_ax.yaxis.label.set_visible(False)
    base_vio_ax.get_legend().remove()

    pnl_mtypes = {'Point': pnt_mtype, 'Loss': dels_mtype, 'Gain': gains_mtype}
    pnl_lbls = {'Point': "other {}\npoint mutations".format(cur_gene),
                'Loss': "loss CNAs", 'Gain': "gain CNAs"}

    for ax, lbl in zip([pnt_ax, loss_ax, gain_ax], ['Point', 'Loss', 'Gain']):
        diag_ax = ax.inset_axes(bounds=(0, 0, 0.63, 0.9))
        vio_ax = ax.inset_axes(bounds=(0.63, 0, 0.37, 0.8))
        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

        cur_mcomb = ExMcomb(plt_mcomb.all_mtype,
                            MuType({('Gene', cur_gene): pnl_mtypes[lbl]}))
        vals_df['cStat'] = np.array(cdata.train_pheno(cur_mcomb))
        vals_df['rStat'] = np.array(cdata.train_pheno(cur_mcomb.not_mtype))

        diag_ax.axhline(y=0.61, xmin=0.07, xmax=0.38, color='red',
                        alpha=0.57, linestyle='--', linewidth=2.3)
        diag_ax.axhline(y=0.61, xmin=0.44, xmax=0.72, color='red',
                        alpha=0.91, linestyle='--', linewidth=1.3)
 
        diag_ax.add_patch(ptchs.Wedge((0.35, 0.8), 0.13, 90, 270,
                                      facecolor=variant_clrs['Point'],
                                      alpha=0.23, clip_on=False))
        diag_ax.add_patch(ptchs.Wedge((0.35, 0.28), 0.27, 90, 270,
                                      facecolor=variant_clrs['WT'],
                                      alpha=0.23, clip_on=False))

        diag_ax.add_patch(ptchs.Wedge((0.47, 0.61), 0.23, 270, 90,
                                      facecolor=variant_clrs['WT'],
                                      edgecolor=variant_clrs[lbl],
                                      hatch='\\', alpha=0.41, linewidth=2.9))

        diag_ax.text(
            0.51, 0.17,
            "mutant for:\n{}\nw/o overlap\n({} samps)".format(
                pnl_lbls[lbl], np.sum(vals_df.cStat & ~vals_df.rStat)),
            size=12, ha='left', va='center'
            )

        sns.violinplot(data=vals_df[~vals_df.mStat & ~vals_df.eStat],
                       x='dummy', y='Value', hue='eStat',
                       hue_order=[False, True], palette=[variant_clrs['WT']],
                       split=True, linewidth=0, cut=0, ax=vio_ax)
        sns.violinplot(data=vals_df[vals_df.mStat & ~vals_df.eStat],
                       x='dummy', y='Value', hue='eStat',
                       hue_order=[False, True],
                       palette=[variant_clrs['Point']],
                       split=True, linewidth=0, cut=0, ax=vio_ax)

        sns.violinplot(data=vals_df[vals_df.cStat & ~vals_df.rStat],
                       x='dummy', y='Value', hue='rStat',
                       hue_order=[True, False], palette=[variant_clrs['WT']],
                       split=True, linewidth=0, cut=0, ax=vio_ax)

        vio_ax.get_children()[0].set_alpha(0.19)
        vio_ax.get_children()[1].set_alpha(0.19)

        vio_ax.get_children()[2].set_alpha(0.53)
        vio_ax.get_children()[2].set_linewidth(2.9)
        vio_ax.get_children()[2].set_edgecolor(variant_clrs[lbl])
        vio_ax.get_children()[2].set_hatch('\\')

        vio_ax.text(0.83, 0.95, "AUC",
                    color='red', size=12, fontstyle='italic',
                    ha='center', va='bottom', transform=vio_ax.transAxes)

        vio_ax.text(
            0.83, 0.95,
            np.greater.outer(vals_df.Value[
                vals_df.cStat & ~vals_df.rStat].values,
                vals_df.Value[
                    ~vals_df.mStat & ~vals_df.eStat].values
                ).mean().round(3),
            color='red', size=15, fontweight='bold',
            ha='center', va='top', transform=vio_ax.transAxes
            )

        vio_ax.set_xticks([])
        vio_ax.set_xticklabels([])
        vio_ax.set_yticklabels([])
        vio_ax.xaxis.label.set_visible(False)
        vio_ax.yaxis.label.set_visible(False)
        vio_ax.get_legend().remove()

    plt.tight_layout(pad=0, w_pad=-3, h_pad=-1)
    plt.savefig(
        os.path.join(plot_dir, args.expr_source,
                     "{}__iso-projection_{}.svg".format(
                         args.cohort, cur_gene)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_similarities(plt_clf, plt_mtype, plt_mcomb,
                          pred_df, pheno_dict, cdata, args):
    cur_gene = tuple(plt_mtype.label_iter())[0]

    mtype_lbl = get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1])
    mtype_tbox = '\n'.join([
        cur_gene,
        get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1],
                        phrase_link='\n')
        ])

    #TODO: handle case where this doesn't find any mutation types
    sim_mcombs = {
        mcomb for mcomb in pred_df.index
        if (isinstance(mcomb, ExMcomb) and mcomb != plt_mcomb
            and (tuple(mcomb.label_iter())[0]
                 == tuple(plt_mtype.label_iter())[0])
            and len(mcomb.mtypes) == 1
            and not (mcomb.all_mtype & shal_mtype).is_empty()
            and (copy_mtype.is_supertype(tuple(mcomb.mtypes)[0])
                 or (len(tuple(mcomb.mtypes)[0].leaves()) == 1
                     and not any('domain' in lvl for lvl
                                 in tuple(mcomb.mtypes)[0].get_levels()))))
        }

    SIZE_RATIO = 43 / 41
    fig, (vio_ax, sim_ax, lgnd_ax) = plt.subplots(
        figsize=(2 + len(sim_mcombs) / SIZE_RATIO, 6), nrows=1, ncols=3,
        gridspec_kw=dict(width_ratios=[1, len(sim_mcombs) / SIZE_RATIO, 1])
        )

    vals_df = pd.DataFrame({
        'Value': pred_df.loc[
            plt_mcomb, cdata.get_train_samples()].apply(np.mean),
        'mStat': pheno_dict[plt_mtype], 'dummy': 0,
        'eStat': np.array(cdata.train_pheno(plt_mcomb.not_mtype))
        })

    sim_df = pd.concat([
        pd.DataFrame({
            'Mcomb': mcomb,
            'Value': pred_df.loc[mcomb, cdata.get_train_samples()][
                pheno_dict[mcomb]].apply(np.mean)
            })
        for mcomb in sim_mcombs
        ])

    sns.violinplot(data=vals_df[~vals_df.mStat & ~vals_df.eStat],
                   x='dummy', y='Value', hue='eStat',
                   palette=[variant_clrs['WT']], hue_order=[False, True],
                   width=1, split=True, linewidth=0, cut=0, ax=vio_ax)
    sns.violinplot(data=vals_df[vals_df.mStat & ~vals_df.eStat],
                   x='dummy', y='Value', hue='eStat',
                   palette=[variant_clrs['Point']], hue_order=[False, True],
                   width=1, split=True, linewidth=0, cut=0, ax=vio_ax)

    vio_ax.set_xlim(-0.5, 0.01)
    for art in vio_ax.get_children()[:2]:
        art.set_alpha(0.41)

    vio_ax.set_yticks([])
    vio_ax.get_legend().remove()
    vio_ax.set_zorder(1)

    wt_mean = np.mean(vals_df.Value[~vals_df.mStat & ~vals_df.eStat])
    vio_ax.axhline(y=wt_mean, xmin=-0.079,
                   xmax=SIZE_RATIO * 1.47 + len(sim_mcombs) / SIZE_RATIO,
                   color=variant_clrs['WT'], clip_on=False,
                   linestyle='--', linewidth=1.7, alpha=0.47)

    mut_mean = np.mean(vals_df.Value[vals_df.mStat & ~vals_df.eStat])
    vio_ax.axhline(y=mut_mean, xmin=-0.079,
                   xmax=SIZE_RATIO * 1.47 + len(sim_mcombs) / SIZE_RATIO,
                   color=variant_clrs['Point'], clip_on=False,
                   linestyle='--', linewidth=1.7, alpha=0.47)

    vals_min, vals_max = pd.concat([vals_df, sim_df],
                                   sort=False).Value.quantile(q=[0, 1])
    vio_min, vio_max = sim_df.Value.quantile(q=[0, 1])

    vals_rng = (vio_max - vio_min) / 53
    plt_min = min(vals_min - vals_rng * 3, 2 * wt_mean - mut_mean)
    plt_max = max(vals_max + vals_rng, 2 * mut_mean - wt_mean)

    vio_ax.text(-0.57, wt_mean, "0",
                size=15, fontweight='bold', ha='right', va='center')
    vio_ax.text(-0.57, mut_mean, "1",
                size=15, fontweight='bold', ha='right', va='center')

    vio_ax.text(0.99, 0,
                "Isolated\nClassification\n of {} (M1)".format(mtype_tbox),
                size=12, fontstyle='italic', ha='right', va='top',
                transform=vio_ax.transAxes)

    mcomb_grps = sim_df.groupby('Mcomb')['Value']
    mcomb_scores = mcomb_grps.mean().sort_values(ascending=False) - wt_mean
    mcomb_scores /= (mut_mean - wt_mean)

    mcomb_mins = mcomb_grps.min()
    mcomb_maxs = mcomb_grps.max()
    mcomb_sizes = mcomb_grps.count()
    clr_norm = colors.Normalize(vmin=-1, vmax=2)

    sns.violinplot(
        data=sim_df, x='Mcomb', y='Value', order=mcomb_scores.index,
        palette=simil_cmap(clr_norm(mcomb_scores.values)), saturation=1,
        width=0.97, linewidth=1.43, cut=0, ax=sim_ax
        )

    for i, (mcomb, scr) in enumerate(mcomb_scores.iteritems()):
        sim_ax.get_children()[i * 2].set_alpha(0.73)
        mcomb_lbl = get_fancy_label(
            tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1],
            phrase_link='\n'
            )

        sim_ax.text(i, mcomb_mins[mcomb] - vals_rng,
                    "{}\n({} samples)".format(mcomb_lbl, mcomb_sizes[mcomb]),
                    size=8, ha='center', va='top')
        sim_ax.text(i, mcomb_maxs[mcomb] + vals_rng / 2, format(scr, '.2f'),
                    size=12, fontweight='bold', ha='center', va='bottom')

    sim_ax.text(0.5, 0,
                "<{}>\nClassifier Scoring of\nOther Isolated\n{} "
                "Mutations (M2)".format(mtype_lbl, cur_gene),
                size=12, fontstyle='italic', ha='center', va='top',
                transform=sim_ax.transAxes)

    clr_min = 2 * wt_mean - mut_mean
    clr_max = 2 * mut_mean - wt_mean
    clr_btm = (clr_min - plt_min) / (plt_max - plt_min)
    clr_top = (clr_max - plt_min) / (plt_max - plt_min)
    clr_rng = (clr_max - clr_min) * 1.38 / (plt_max - plt_min)
    clr_btm = clr_btm - (clr_top - clr_btm) * 0.19

    clr_ax = lgnd_ax.inset_axes(bounds=(0, clr_btm, 0.53, clr_rng))
    clr_bar = ColorbarBase(ax=clr_ax, cmap=simil_cmap, norm=clr_norm,
                           extend='both', extendfrac=0.19,
                           ticks=[-0.73, 0, 0.5, 1.0, 1.73])

    clr_bar.ax.set_yticklabels(
        ['M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'],
        size=11, fontweight='bold'
        )

    for ax in vio_ax, sim_ax, lgnd_ax:
        ax.set_ylim(plt_min, plt_max)
        ax.axis('off')

    plt.savefig(
        os.path.join(plot_dir, args.expr_source,
                     "{}__iso-similarities_{}.svg".format(
                         args.cohort, cur_gene)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_example',
        description="Creates diagrams explaining subgrouping isolation."
        )

    parser.add_argument('expr_source', help="a source of expression data")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('--genes', '-g', nargs='+',
                        help="a set of genes from which to draw an example")

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(out_dir.glob("out-aucs__*__*__*.p.gz"))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for cohort `{}` "
                         "from expression source `{}`!".format(
                             args.cohort, args.expr_source))

    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)
    out_df = pd.DataFrame(
        [{'Levels': '__'.join(out_file.stem.split('__')[1:-2]),
          'Classif': out_file.stem.split('__')[-1].split('.p')[0],
          'File': out_file}
         for out_file in out_list]
        )

    out_iter = out_df.groupby(['Levels', 'Classif'])['File']
    phn_dict = dict()
    cdata = None
    out_aucs = list()
    out_preds = {clf: list() for clf in set(out_df.Classif)}

    for (lvls, clf), out_files in out_iter:
        auc_list = [None for _ in out_files]
        pred_list = [None for _ in out_files]

        for i, out_file in enumerate(out_files):
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir,
                                  '__'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

                if cdata is None:
                    cdata = new_cdata
                else:
                    cdata.merge(new_cdata)

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                auc_dict = pickle.load(f)

                auc_list[i] = pd.DataFrame({
                    ex_lbl: auc_vals['mean']
                    for ex_lbl, auc_vals in auc_dict.items()
                    })

                auc_list[i].index = pd.MultiIndex.from_tuples(
                    [(clf, mtype) for mtype in auc_list[i].index],
                    names=['Classif', 'Mutation']
                    )

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pred", out_tag])),
                             'r') as f:
                pred_list[i] = pickle.load(f)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals.index) for auc_vals in auc_list]] * 2))
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()
            out_aucs += [auc_list[super_indx]]
            out_preds[clf] += [pred_list[super_indx]]

        else:
            raise ValueError

    auc_df = pd.concat(out_aucs, sort=True)
    auc_df = auc_df.loc[~auc_df.index.duplicated()]

    base_muts = {(clf, mtype) for clf, mtype in auc_df.index
                 if (not isinstance(mtype, (Mcomb, ExMcomb))
                     and 'Copy' not in mtype.get_levels()
                     and not any(lvl == 'Class' or 'domain' in lvl
                                 for lvl in mtype.get_levels())
                     and tuple(mtype.subtype_iter())[0][1] != pnt_mtype
                     and len(mtype.leaves()) == 1

                     and any(oth_clf == clf and isinstance(mcomb, ExMcomb)
                             and len(mcomb.mtypes) == 1
                             and tuple(mcomb.mtypes)[0] == mtype
                             and not (mcomb.all_mtype & shal_mtype).is_empty()
                             for oth_clf, mcomb in auc_df.index))}

    if args.genes:
        base_muts = {(clf, mtype) for clf, mtype in base_muts
                     if tuple(mtype.label_iter())[0] in args.genes}

    ex_muts = {
        (clf, mtype): {
            mcomb for oth_clf, mcomb in auc_df.index
            if (oth_clf == clf and isinstance(mcomb, ExMcomb)
                and len(mcomb.mtypes) == 1 and tuple(mcomb.mtypes)[0] == mtype
                and not (mcomb.all_mtype & shal_mtype).is_empty())
            }
        for clf, mtype in base_muts
        }

    for (clf, mtype), ex_mcombs in ex_muts.items():
        assert len(ex_mcombs) <= 1, ("Found multiple ExMcombs matching {} "
                                     "with testing classifier `{}`!".format(
                                         mtype, clf))

    if not any(len(ex_mcombs) > 0 for ex_mcombs in ex_muts.values()):
        raise ValueError("No simple exclusive mutation subgrouping found!")

    mut_aucs = {
        (clf, mtype): auc_df.loc[[(clf, mtype), (clf, tuple(ex_mcomb)[0])],
                                 ['All', 'Iso']].set_index([['All', 'Ex']])
        for (clf, mtype), ex_mcomb in ex_muts.items() if len(ex_mcomb) == 1
        }

    off_diags = {mut: aucs.values[~np.equal(*np.indices((2, 2)))]
                 for mut, aucs in mut_aucs.items()}

    use_clf, use_mtype = sorted(
        mut_aucs, key=lambda mut: sum([
            mut_aucs[mut].loc['All', 'All'] - np.min(off_diags[mut]),
            np.max(off_diags[mut]) - mut_aucs[mut].loc['Ex', 'Iso']
            ])
        )[0]
    use_mcomb = tuple(ex_muts[use_clf, use_mtype])[0]

    pred_dfs = {
        ex_lbl: pd.concat([pred_list[ex_lbl]
                           for pred_list in out_preds[use_clf]], sort=True)
        for ex_lbl in ['All', 'Iso']
        }

    pred_dfs = {ex_lbl: pred_df.loc[~pred_df.index.duplicated()]
                for ex_lbl, pred_df in pred_dfs.items()}

    # create the plots
    plot_base_classification(use_mtype, use_mcomb,
                             pred_dfs['All'], phn_dict, cdata, args)
    plot_iso_classification(use_mtype, use_mcomb,
                            pred_dfs, phn_dict, cdata, args)

    plot_iso_projection(use_mtype, use_mcomb,
                        pred_dfs['Iso'], phn_dict, cdata, args)
    plot_iso_similarities(use_clf, use_mtype, use_mcomb,
                          pred_dfs['Iso'], phn_dict, cdata, args)


if __name__ == '__main__':
    main()

