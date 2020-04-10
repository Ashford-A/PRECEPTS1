
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'example')

from HetMan.experiments.subvariant_isolate.utils import (
    Mcomb, ExMcomb, calc_auc)
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.utils import get_cohort_label
from dryadic.features.mutations import MuType

from HetMan.experiments.utilities import simil_cmap
from HetMan.experiments.subvariant_isolate.setup_isolate import merge_cohorts
from HetMan.experiments.subvariant_isolate.utils import get_fancy_label
from HetMan.experiments.subvariant_test import variant_clrs, pnt_mtype
from HetMan.experiments.subvariant_isolate import cna_mtypes

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as ptchs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
from matplotlib import colors

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_base_classification(plt_clf, plt_mtype, plt_mcomb,
                             pred_df, pheno_dict, cdata, args):
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
    mtype_lbl = get_fancy_label(plt_mtype)
    mtype_tbox = get_fancy_label(plt_mtype, phrase_link='\n')

    for ax in coh_ax, clf_ax, ovp_ax:
        ax.axis('off')

    coh_ax.text(0.63, 1,
                "{}\n({} samples)".format(get_cohort_label(args.cohort),
                                          len(cdata.get_samples())),
                size=12, ha='center', va='top')

    coh_ax.add_patch(ptchs.FancyArrowPatch(
        posA=(0.63, 0.66), posB=(0.63, 0.52),
        arrowstyle=ptchs.ArrowStyle('-[', lengthB=7.1, widthB=119)
        ))

    coh_ax.add_patch(ptchs.Rectangle((0.3, 0.28),
                                     (1 - mut_prop) * 0.66, 0.22,
                                     facecolor=variant_clrs['WT'],
                                     alpha=0.41, hatch='/', linewidth=1.3,
                                     edgecolor='0.51'))
    coh_ax.add_patch(ptchs.Rectangle((0.3 + (1 - mut_prop) * 0.66, 0.28),
                                     mut_prop * 0.66, 0.22,
                                     facecolor=variant_clrs['Point'],
                                     alpha=0.41, hatch='/', linewidth=1.3,
                                     edgecolor='0.51'))

    coh_ax.text(0.28, 0.39, "mutated status for:\n{}".format(mtype_tbox),
                size=11, ha='right', va='center')

    coh_ax.add_patch(ptchs.Rectangle((0.3 + ovlp_prop * 0.66, 0.28),
                                     np.mean(plt_df.rStat) * 0.66, 0.22,
                                     hatch='\\', linewidth=1.3,
                                     edgecolor='0.51', facecolor='None'))

    coh_ax.add_patch(ptchs.Rectangle((0.3 + ovlp_prop * 0.66, 0.01),
                                      np.mean(plt_df.rStat) * 0.66, 0.22,
                                      alpha=0.83, hatch='\\',
                                      linewidth=1.3, edgecolor='0.51',
                                      facecolor=variant_clrs['Point']))

    coh_ax.text(0.29 + ovlp_prop * 0.66, 0.23,
                "{} mutations\nother than {}".format(args.gene, mtype_tbox),
                color=variant_clrs['Point'], size=10, ha='right', va='top')
    coh_ax.text(0.3 + ovlp_prop * 0.66 + np.mean(plt_df.rStat) * 0.33, -0.02,
                "({} samples)".format(np.sum(plt_df.rStat)),
                color=variant_clrs['Point'], size=10, ha='center', va='top')

    diag_ax1 = inset_axes(clf_ax, width='100%', height='100%',
                          loc=10, borderpad=0,
                          bbox_to_anchor=(0, 0, 0.67, 1),
                          bbox_transform=clf_ax.transAxes)
    vio_ax1 = inset_axes(clf_ax, width='100%', height='100%',
                         loc=10, borderpad=0,
                         bbox_to_anchor=(0.67, 0, 0.33, 1),
                         bbox_transform=clf_ax.transAxes)

    diag_ax2 = inset_axes(ovp_ax, width='100%', height='100%',
                          loc=10, borderpad=0,
                          bbox_to_anchor=(0, 0, 0.67, 1),
                          bbox_transform=ovp_ax.transAxes)
    vio_ax2 = inset_axes(ovp_ax, width='100%', height='100%',
                         loc=10, borderpad=0,
                         bbox_to_anchor=(0.67, -0.11, 0.33, 0.97),
                         bbox_transform=ovp_ax.transAxes)

    for diag_ax in diag_ax1, diag_ax2:
        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

        diag_ax.add_patch(ptchs.FancyArrow(
            0.85, 0.57, dx=0.14, dy=0, width=0.03,
            length_includes_head=True, head_length=0.06,
            linewidth=1.7, facecolor='white', edgecolor='black'
            ))

    diag_ax1.add_patch(ptchs.Circle((0.5, 0.85), radius=0.14,
                                    facecolor=variant_clrs['Point'],
                                    alpha=0.41))
    diag_ax1.text(0.5, 0.85,
                  "mutant for:\n{}\n({} samples)".format(
                      mtype_tbox, np.sum(plt_df.cStat)),
                  size=8, ha='center', va='center')

    diag_ax1.add_patch(ptchs.Circle(
        (0.5, 0.32), radius=0.31, facecolor=variant_clrs['WT'], alpha=0.41))
    diag_ax1.text(0.5, 0.32,
                  "wild-type for:\n{}\n({} samples)".format(
                      mtype_tbox, np.sum(~plt_df.cStat)),
                  size=13, ha='center', va='center')

    diag_ax1.text(0.2, 0.67, "predict\nmutated\nstatus", color='red',
                  size=12, fontstyle='italic', ha='right', va='center')
    diag_ax1.axhline(y=0.67, xmin=0.23, xmax=0.86, color='red',
                     linestyle='--', linewidth=2.7, alpha=0.83)

    diag_ax1.text(0.82, 0.68, "{} (+)".format(np.sum(plt_df.cStat)),
                  color='red', size=9, fontstyle='italic', 
                  ha='right', va='bottom')
    diag_ax1.text(0.82, 0.655, "{} (\u2212)".format(np.sum(~plt_df.cStat)),
                  color='red', size=9, fontstyle='italic',
                  ha='right', va='top')

    sns.violinplot(data=plt_df[~plt_df.cStat], y='Value', ax=vio_ax1,
                   palette=[variant_clrs['WT']], linewidth=0, cut=0)
    sns.violinplot(data=plt_df[plt_df.cStat], y='Value', ax=vio_ax1,
                   palette=[variant_clrs['Point']], linewidth=0, cut=0)

    vio_ax1.text(0.5, 113 / 111,
                 "AUC: {:.3f}".format(calc_auc(plt_df.Value.values,
                                               plt_df.cStat)),
                 color='red', size=14, fontstyle='italic',
                 ha='center', va='bottom', transform=vio_ax1.transAxes)

    diag_ax2.add_patch(ptchs.Wedge((0.48, 0.89), 0.14, 90, 270,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51', clip_on=False))

    diag_ax2.add_patch(ptchs.Wedge((0.52, 0.89), 0.14, 270, 90,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51', clip_on=False))
    diag_ax2.add_patch(ptchs.Wedge((0.52, 0.89), 0.14, 270, 90,
                                   facecolor='None', edgecolor='0.61',
                                   hatch='\\', linewidth=0.8, clip_on=False))

    diag_ax2.text(0.22, 0.69, "same classifier\nresults", color='red',
                  size=10, fontstyle='italic', ha='right', va='center')
    diag_ax2.axhline(y=0.69, xmin=0.23, xmax=0.86, color='red',
                     linestyle='--', linewidth=1.3, alpha=0.67)

    diag_ax2.add_patch(ptchs.Wedge((0.48, 0.32), 0.31, 90, 270,
                                   facecolor=variant_clrs['WT'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51'))

    diag_ax2.add_patch(ptchs.Wedge((0.52, 0.32), 0.31, 270, 90,
                                   facecolor=variant_clrs['WT'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51'))
    diag_ax2.add_patch(ptchs.Wedge((0.52, 0.32), 0.31, 270, 90,
                                   facecolor='None', edgecolor='0.61',
                                   linewidth=0.8, hatch='\\'))

    diag_ax2.text(0.33, 0.89,
                  "mutant for:\n{}\nw/o overlap\n({} samps)".format(
                      mtype_tbox, np.sum(plt_df.cStat & ~plt_df.rStat)),
                  size=9, ha='right', va='center')
    diag_ax2.text(0.67, 0.89,
                  "mutant for:\n{}\nw/ overlap\n({} samps)".format(
                      mtype_tbox, np.sum(plt_df.cStat & plt_df.rStat)),
                  size=9, ha='left', va='center')

    diag_ax2.text(0.47, 0.32,
                  "wild-type for:\n{}\nw/o overlap\n({} samps)".format(
                      mtype_tbox, np.sum(~plt_df.cStat & ~plt_df.rStat)),
                  size=10, ha='right', va='center')
    diag_ax2.text(0.53, 0.32,
                  "wild-type for:\n{}\nw/ overlap\n({} samps)".format(
                      mtype_tbox, np.sum(~plt_df.cStat & plt_df.rStat)),
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
    diag_ax2.axvline(x=0.5, ymin=-0.03, ymax=1.03, clip_on=False,
                     color=variant_clrs['Point'], linewidth=1.1, alpha=0.81,
                     linestyle=':')

    diag_ax2.text(0.5, -0.05,
                  "partition scored samples according to\noverlap with "
                  "{} mutations\nthat are not {}".format(args.gene, mtype_lbl),
                  color=variant_clrs['Point'], size=10,
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

    vio_ax2.text(0.15, 1.1, "{}\nw/o overlap".format(mtype_tbox),
                 color=variant_clrs['Point'], size=10,
                 fontstyle='italic', ha='center', va='bottom',
                 transform=vio_ax2.transAxes)

    vio_ax2.text(
        0.15, 113 / 111,
        "AUC: {:.3f}".format(calc_auc(plt_df.Value[~plt_df.rStat].values,
                                      plt_df.cStat[~plt_df.rStat])),
        color='red', size=13, fontstyle='italic',
        ha='center', va='bottom', transform=vio_ax2.transAxes
        )

    vio_ax2.text(0.85, 1.1, "{}\nw/ overlap".format(mtype_tbox),
                 color=variant_clrs['Point'], size=10,
                 fontstyle='italic', ha='center', va='bottom',
                 transform=vio_ax2.transAxes)

    vio_ax2.text(
        0.85, 113 / 111,
        "AUC: {:.3f}".format(calc_auc(plt_df.Value[plt_df.rStat].values,
                                      plt_df.cStat[plt_df.rStat])),
        color='red', size=13, fontstyle='italic',
        ha='center', va='bottom', transform=vio_ax2.transAxes
        )

    plt.tight_layout(pad=-1, h_pad=2.3)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__base-classification.svg".format(args.cohort)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_classification(plt_clf, plt_mtype, plt_mcomb,
                            pred_dfs, pheno_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

    mtype_tbox = get_fancy_label(plt_mtype, phrase_link='\n')
    rest_stat = np.array(cdata.train_pheno(plt_mcomb.not_mtype))
    plt_muts = {'All': plt_mtype, 'Ex': plt_mcomb}

    mcomb_masks = [('All', {lbl: np.array([True] * len(cdata.get_samples()))
                            for lbl in plt_muts}),
                   ('Iso', {lbl: ~(rest_stat & ~pheno_dict[mut])
                            for lbl, mut in plt_muts.items()})]

    for i, (smp_lbl, msk) in enumerate(mcomb_masks):
        for j, (mut_lbl, mut) in enumerate(plt_muts.items()):
            plt_df = pd.DataFrame({
                'Value': pred_dfs[smp_lbl].loc[
                    mut, cdata.get_train_samples()].apply(np.mean),
                'cStat': pheno_dict[mut], 'uStat': msk[mut_lbl]
                })

            diag_ax = inset_axes(axarr[i, j], width='100%', height='100%',
                                 loc=10, borderpad=0,
                                 bbox_to_anchor=(0, 0, 0.67, 1),
                                 bbox_transform=axarr[i, j].transAxes)
            vio_ax = inset_axes(axarr[i, j], width='100%', height='100%',
                                loc=10, borderpad=0,
                                bbox_to_anchor=(0.67, 0, 0.33, 0.93),
                                bbox_transform=axarr[i, j].transAxes)

            axarr[i, j].axis('off')
            diag_ax.axis('off')
            diag_ax.set_aspect('equal')

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
                        color='red', size=15, fontstyle='italic',
                        ha='center', va='bottom', transform=vio_ax.transAxes)

            vio_ax.get_children()[0].set_alpha(0.41)
            vio_ax.get_children()[2].set_alpha(0.41)

            diag_ax.add_patch(ptchs.Wedge((0.48, 0.87), 0.17, 90, 270,
                                          facecolor=variant_clrs['Point'],
                                          alpha=0.41, clip_on=False))
            diag_ax.text(0.47, 0.87,
                         "mutant for:\n{}\nw/o overlap\n({} samps)".format(
                             mtype_tbox, np.sum(plt_df.cStat & ~rest_stat)),
                         size=7, ha='right', va='center')

            if np.sum(plt_df.cStat & rest_stat):
                diag_ax.add_patch(ptchs.Wedge((0.52, 0.87), 0.17, 270, 90,
                                              facecolor=variant_clrs['Point'],
                                              alpha=0.41, clip_on=False))

                diag_ax.text(
                    0.53, 0.87,
                    "mutant for:\n{}\nw/ overlap\n({} samps)".format(
                        mtype_tbox, np.sum(plt_df.cStat & rest_stat)),
                    size=7, ha='left', va='center'
                    )

            diag_ax.add_patch(ptchs.Wedge((0.48, 0.29), 0.31, 90, 270,
                                          facecolor=variant_clrs['WT'],
                                          alpha=0.41, clip_on=False))
            diag_ax.text(0.47, 0.29,
                         "wild-type for:\n{}\nw/o overlap\n({} samps)".format(
                             mtype_tbox, np.sum(~plt_df.cStat & ~rest_stat)),
                         size=10, ha='right', va='center')

            if np.sum(~plt_df.cStat & rest_stat & plt_df.uStat):
                diag_ax.add_patch(ptchs.Wedge((0.52, 0.29), 0.31, 270, 90,
                                              facecolor=variant_clrs['WT'],
                                              alpha=0.41, clip_on=False))

                diag_ax.text(
                    0.53, 0.29,
                    "wild-type for:\n{}\nw/ overlap\n({} samps)".format(
                        mtype_tbox, np.sum(~plt_df.cStat & rest_stat)),
                    size=10, ha='left', va='center'
                    )

            diag_ax.add_patch(ptchs.FancyArrow(
                0.83, 0.51, dx=0.13, dy=0, width=0.03, clip_on=False,
                length_includes_head=True, head_length=0.05, alpha=0.93,
                linewidth=1.5, facecolor='None', edgecolor='black'
                ))

            vio_ax.set_xticks([])
            vio_ax.set_xticklabels([])
            vio_ax.set_yticklabels([])
            vio_ax.yaxis.label.set_visible(False)

    plt.tight_layout(pad=0, w_pad=2.1, h_pad=1.9)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__iso-classification.svg".format(args.cohort)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_iso_projection(plt_clf, plt_mtype, plt_mcomb,
                        pred_df, pheno_dict, cdata, args):
    fig, ((base_ax, pnt_ax), (loss_ax, gain_ax)) = plt.subplots(
        figsize=(11, 9), nrows=2, ncols=2)

    mtype_tbox = get_fancy_label(plt_mtype, phrase_link='\n')
    rest_stat = np.array(cdata.train_pheno(plt_mcomb.not_mtype))

    vals_df = pd.DataFrame({
        'Value': pred_df.loc[
            plt_mcomb, cdata.get_train_samples()].apply(np.mean),
        'mStat': pheno_dict[plt_mtype], 'eStat': rest_stat, 'dummy': 0
        })

    base_diag_ax = inset_axes(base_ax, loc=10, width='100%', height='100%',
                              borderpad=0, bbox_to_anchor=(0, 0, 0.63, 0.9),
                              bbox_transform=base_ax.transAxes)
    base_vio_ax = inset_axes(base_ax, width='100%', height='100%',
                             loc=10, borderpad=0,
                             bbox_to_anchor=(0.63, 0, 0.37, 0.8),
                             bbox_transform=base_ax.transAxes)

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

    pnl_mtypes = {'Point': pnt_mtype, 'Loss': dict(cna_mtypes)['Loss'],
                  'Gain': dict(cna_mtypes)['Gain']}
    pnl_lbls = {'Point': "other {}\npoint mutations".format(args.gene),
                'Loss': "loss CNAs", 'Gain': "gain CNAs"}

    for ax, lbl in zip([pnt_ax, loss_ax, gain_ax], ['Point', 'Loss', 'Gain']):
        cur_mcomb = ExMcomb(plt_mcomb.all_mtype, pnl_mtypes[lbl])
        vals_df['cStat'] = np.array(cdata.train_pheno(cur_mcomb))
        vals_df['rStat'] = np.array(cdata.train_pheno(cur_mcomb.not_mtype))

        diag_ax = inset_axes(ax, width='100%', height='100%', loc=10,
                             borderpad=0, bbox_to_anchor=(0, 0, 0.63, 0.9),
                             bbox_transform=ax.transAxes)
        vio_ax = inset_axes(ax, width='100%', height='100%', loc=10,
                            borderpad=0, bbox_to_anchor=(0.63, 0, 0.37, 0.8),
                            bbox_transform=ax.transAxes)

        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

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
            0.55, 0.19,
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
    plt.savefig(os.path.join(plot_dir, args.gene,
                             "{}__iso-projection.svg".format(args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_iso_similarities(mut, use_vals, pheno_dict, cdata, args):
    use_lvls, use_mtype = mut
    mtype_lbl = " ".join([args.gene,
                          get_fancy_label(use_mtype)]).replace('\n', ' ')

    all_mtype = MuType({('Gene', args.gene): cdata.mtree[args.gene].allkey()})
    use_mcomb = ExMcomb(all_mtype, use_mtype)
    pnt_mtype = MuType({('Gene', args.gene): dict(variant_mtypes)['Point']})

    sim_mcombs = {
        mcomb: ExMcomb(all_mtype, *[mtype & all_mtype - use_mtype
                                    for mtype in mcomb.mtypes])
        for lvls, mcomb in use_vals.index
        if (isinstance(mcomb, ExMcomb) and mcomb.all_mtype == all_mtype
            and ((lvls == use_lvls and mcomb != use_mcomb
                  and not any(use_mtype.is_supertype(mtype)
                              for mtype in mcomb.mtypes)
                  and ((len(mcomb.mtypes) == 1
                        and len(tuple(mcomb.mtypes)[0].subkeys()) == 1)
                       or (all(((len(mtype.subkeys()) == 2
                                 and (mtype & pnt_mtype).is_empty())
                                or (len(mtype.subkeys()) == 1
                                    and not (mtype & pnt_mtype).is_empty()
                                    and (mtype.get_levels()
                                         == use_mtype.get_levels())))
                               for mtype in mcomb.mtypes)
                           and any((mtype & pnt_mtype).is_empty()
                                   for mtype in mcomb.mtypes))))
                 or lvls == 'Copy'))
        }

    fig, (vio_ax, sim_ax, clr_ax) = plt.subplots(
        figsize=(3 + len(sim_mcombs), 6), nrows=1, ncols=3,
        gridspec_kw=dict(width_ratios=[6, 5 * len(sim_mcombs), 1])
        )

    vals_df = pd.DataFrame({
        'Value': [np.mean(vals)
                  for vals in use_vals.loc[[(use_lvls, use_mcomb)]].iloc[0]],
        'cStat': pheno_dict[use_mcomb],
        'rStat': np.array(cdata.train_pheno(all_mtype - use_mtype))
        })

    sns.violinplot(data=vals_df[~vals_df.cStat & ~vals_df.rStat], x='cStat',
                   y='Value', hue='rStat', palette=[variant_clrs['WT']],
                   hue_order=[False, True], split=True, linewidth=0, cut=0,
                   ax=vio_ax)
    sns.violinplot(data=vals_df[vals_df.cStat & ~vals_df.rStat], x='cStat',
                   y='Value', hue='rStat', palette=[variant_clrs['Point']],
                   hue_order=[False, True], split=True, linewidth=0, cut=0,
                   ax=vio_ax)

    vals_min, vals_max = vals_df.Value.quantile(q=[0, 1])
    vals_rng = (vals_max - vals_min) / 101
    vio_ax.set_xlim(-0.5, 0.01)

    for art in vio_ax.get_children()[:2]:
        art.set_alpha(0.41)

    vio_ax.set_yticks([])
    vio_ax.get_legend().remove()
    vio_ax.set_zorder(1)
    clr_ax.set_zorder(2)

    wt_mean = np.mean(vals_df.Value[~vals_df.cStat & ~vals_df.rStat])
    vio_ax.axhline(y=wt_mean, xmin=0, xmax=1.91 + len(sim_mcombs) * 0.83,
                   color=variant_clrs['WT'], clip_on=False, linestyle='--',
                   linewidth=1.6, alpha=0.51)

    mut_mean = np.mean(vals_df.Value[vals_df.cStat & ~vals_df.rStat])
    vio_ax.axhline(y=mut_mean, xmin=0, xmax=1.91 + len(sim_mcombs) * 0.83,
                   color=variant_clrs['Point'], clip_on=False, linestyle='--',
                   linewidth=1.6, alpha=0.51)

    vio_ax.text(-0.52, wt_mean, "0",
                size=12, fontstyle='italic', ha='right', va='center')
    vio_ax.text(-0.52, mut_mean, "1",
                size=12, fontstyle='italic', ha='right', va='center')

    vio_ax.text(0, vals_min - 7 * vals_rng,
                "Isolated\nClassification\n of {}\n(M1)".format(
                    mtype_lbl.replace(" with ", "\n")),
                size=13, fontweight='semibold', ha='right', va='top')

    sim_df = pd.concat([
        pd.DataFrame({
            'Mcomb': mcomb, 'Value': [
                np.mean(vals) for vals in use_vals.loc[
                    [(use_lvls, use_mcomb)],
                    np.array(cdata.train_pheno(ex_mcomb))
                    ].iloc[0]
                ]
            })
        for mcomb, ex_mcomb in sim_mcombs.items()
        ])

    mcomb_grps = sim_df.groupby('Mcomb')['Value']
    mcomb_scores = mcomb_grps.mean().sort_values(ascending=False) - wt_mean
    mcomb_scores /= (mut_mean - wt_mean)

    mcomb_mins = mcomb_grps.min()
    mcomb_maxs = mcomb_grps.max()
    mcomb_sizes = mcomb_grps.count()
    clr_norm = colors.Normalize(vmin=-1, vmax=2)

    sns.violinplot(data=sim_df, x='Mcomb', y='Value',
                   order=mcomb_scores.index,
                   palette=simil_cmap(clr_norm(mcomb_scores.values)),
                   saturation=1, linewidth=10/7, cut=0, width=0.87, ax=sim_ax)

    for i, (mcomb, scr) in enumerate(mcomb_scores.iteritems()):
        sim_ax.get_children()[i * 2].set_alpha(8/11)

        mcomb_lbl = '\nAND '.join(
            ['\n'.join(["any other", get_fancy_label(mtype), "mutation"])
             if mtype.is_supertype(use_mtype) else get_fancy_label(mtype)
             for mtype in mcomb.mtypes]
            )

        sim_ax.text(i, mcomb_mins[mcomb] - vals_rng / 2,
                    "{}\n({} samples)".format(mcomb_lbl, mcomb_sizes[mcomb]),
                    size=9, ha='center', va='top')
        sim_ax.text(i, mcomb_maxs[mcomb] + vals_rng / 2, format(scr, '.2f'),
                    size=11, fontstyle='italic', ha='center', va='bottom')

    sim_ax.text(len(mcomb_scores) / 2, vals_min - 7 * vals_rng,
                "{} Classifier Scoring\nof Other "
                "Isolated {} Mutations\n(M2)".format(mtype_lbl, args.gene),
                size=13, fontweight='semibold', ha='center', va='top')

    for ax in vio_ax, sim_ax:
        ax.axis('off')
        ax.set_ylim(vals_min - 0.5 * vals_rng, vals_max + vals_rng)

    clr_min = clr_norm((vals_min - vals_rng - wt_mean) / (mut_mean - wt_mean))
    clr_max = clr_norm((vals_max + vals_rng - wt_mean) / (mut_mean - wt_mean))
    clr_ext = min(0.2, -clr_min, clr_max - 1)

    clr_bar = ColorbarBase(ax=clr_ax, cmap=simil_cmap, norm=clr_norm,
                           extend='both', extendfrac=clr_ext,
                           ticks=[-0.5, 0, 0.5, 1.0, 1.5])

    clr_bar.set_ticklabels(
        ['M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'])
    clr_ax.set_ylim(clr_min, clr_max)
    clr_ax.tick_params(labelsize=11)

    plt.tight_layout(pad=0, h_pad=0, w_pad=-4.1)
    plt.savefig(os.path.join(plot_dir, args.gene,
                             "{}__iso-similarities.svg".format(args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot an example diagram showing how overlap with other types of "
        "mutations can affect a mutation classification task."
        )

    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('cohort', help='a TCGA cohort')

    args = parser.parse_args()
    out_list = tuple(Path(base_dir, args.gene).glob(
        "out-siml__{}__*__*__*.p.gz".format(args.cohort)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for gene {} "
                         "in cohort `{}`!".format(args.gene, args.cohort))

    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.stem.split('__')[2:-2]),
          'Classif': out_file.stem.split('__')[-1].split('.p')[0],
          'File': out_file}
         for out_file in out_list]
        )

    out_iter = out_use.groupby(['Levels', 'Classif'])['File']
    phn_dict = dict()
    cdata_dict = {lvls: None for lvls in set(out_use.Levels)}

    out_aucs = list()
    out_preds = {clf: list() for clf in set(out_use.Classif)}
    out_simls = {clf: list() for clf in set(out_use.Classif)}

    for (lvls, clf), out_files in out_iter:
        auc_list = [None for _ in out_files]
        pred_list = [None for _ in out_files]
        siml_list = [None for _ in out_files]

        for i, out_file in enumerate(out_files):
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            if cdata_dict[lvls] is None:
                with bz2.BZ2File(Path(base_dir, args.gene,
                                      '__'.join(["cohort-data", out_tag])),
                                 'r') as f:
                    cdata_dict[lvls] = pickle.load(f)

            with bz2.BZ2File(Path(base_dir, args.gene,
                                  '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(base_dir, args.gene,
                                  '__'.join(["out-aucs", out_tag])),
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

            with bz2.BZ2File(Path(base_dir, args.gene,
                                  '__'.join(["out-pred", out_tag])),
                             'r') as f:
                pred_list[i] = pickle.load(f)

            with bz2.BZ2File(Path(base_dir, args.gene,
                                  '__'.join(["out-siml", out_tag])),
                             'r') as f:
                siml_list[i] = pickle.load(f)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals.index) for auc_vals in auc_list]] * 2))
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()
            out_aucs += [auc_list[super_indx]]
            out_preds[clf] += [pred_list[super_indx]]
            out_simls[clf] += [siml_list[super_indx]]

        else:
            raise ValueError

    auc_df = pd.concat(out_aucs)
    auc_df = auc_df.loc[~auc_df.index.duplicated()]

    cdata = merge_cohorts(cdata_dict.values())
    all_mtypes = {lvls: MuType(mtree.allkey())
                  for lvls, mtree in cdata.mtrees.items()}

    # get list of hotspot mutations that appear in enough samples by
    # themselves to have had inferred values calculated for them
    base_muts = {(clf, mtype) for clf, mtype in auc_df.index
                 if (not isinstance(mtype, (Mcomb, ExMcomb, RandomType))
                     and 'Copy' not in mtype.get_levels()
                     and not any('domain' in lvl
                                 for lvl in mtype.get_levels())
                     and len(mtype.subkeys()) == 1
                     and any(oth_clf == clf and isinstance(mcomb, ExMcomb)
                             and len(mcomb.mtypes) == 1
                             and tuple(mcomb.mtypes)[0] == mtype
                             for oth_clf, mcomb in auc_df.index))}

    ex_muts = {
        (clf, mtype): {
            mcomb for oth_clf, mcomb in auc_df.index
            if (oth_clf == clf and isinstance(mcomb, ExMcomb)
                and len(mcomb.mtypes) == 1 and tuple(mcomb.mtypes)[0] == mtype
                and mcomb.not_mtype | mtype in all_mtypes.values())
            }
        for clf, mtype in base_muts
        }

    for (clf, mtype), ex_mcombs in ex_muts.items():
        assert len(ex_mcombs) <= 1, ("Found multiple ExMcombs matching {} "
                                     "with testing classifier `{}`!".format(
                                         mtype, clf))

    if not any(len(ex_mcombs) > 0 for ex_mcombs in ex_muts.values()):
        raise ValueError("No simple exclusive mutation subgrouping found for "
                         "{} in cohort {}!".format(args.gene, args.cohort))

    # find experiments where the classifier performed well and also with an
    # improvement when samples with overlapping mutations were removed
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

    pred_dfs = {ex_lbl: pd.concat([pred_list[ex_lbl]
                                   for pred_list in out_preds[use_clf]])
                for ex_lbl in ['All', 'Iso']}
    pred_dfs = {ex_lbl: pred_df.loc[~pred_df.index.duplicated()]
                for ex_lbl, pred_df in pred_dfs.items()}

    plot_base_classification(use_clf, use_mtype, use_mcomb,
                             pred_dfs['All'], phn_dict, cdata, args)
    plot_iso_classification(use_clf, use_mtype, use_mcomb,
                            pred_dfs, phn_dict, cdata, args)

    plot_iso_projection(use_clf, use_mtype, use_mcomb,
                        pred_dfs['Iso'], phn_dict, cdata, args)

    """
    plot_iso_similarities((use_clf, use_mtype),
                          out_infers.copy()[use_clf]['Iso'],
                          out_datas.copy()[use_clf][0],
                          cdict[use_lvls], args)
    """

if __name__ == '__main__':
    main()

