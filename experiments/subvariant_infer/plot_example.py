
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_infer')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'example')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.subvariant_infer.setup_infer import ExMcomb
from HetMan.experiments.subvariant_infer.fit_infer import load_cohort_data
from HetMan.experiments.subvariant_infer.utils import (
    check_output, load_infer_output, calc_auc)
from HetMan.experiments.subvariant_infer import (
    variant_mtypes, variant_clrs, MuType)
from HetMan.experiments.utilities import simil_cmap

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.patches as ptchs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
from matplotlib import colors


def plot_base_classification(mtype, use_vals, cdata, args):
    fig, (coh_ax, clf_ax, ovp_ax) = plt.subplots(
        figsize=(6, 8), nrows=3, ncols=1,
        gridspec_kw=dict(height_ratios=[1, 3, 3])
        )

    mtype_str = ":".join([args.gene, str(mtype).split(':')[-1][2:]])
    mut_str = mtype_str.split(':')[-1]

    rest_mtype = MuType(cdata.train_mut.allkey()) - mtype
    use_df = pd.DataFrame({'Value': use_vals.loc[mtype],
                           'cStat': np.array(cdata.train_pheno(mtype)),
                           'rStat': np.array(cdata.train_pheno(rest_mtype))})

    mut_prop = np.sum(use_df.cStat) / len(cdata.samples)
    ovlp_prop = np.mean(~use_df.rStat[~use_df.cStat]) * (1 - mut_prop)

    for ax in coh_ax, clf_ax, ovp_ax:
        ax.axis('off')

    coh_ax.text(0.5, 1, "TCGA-{}\n({} samples)".format(args.cohort,
                                                       len(cdata.samples)),
                size=10, ha='center', va='top')
    coh_ax.add_patch(ptchs.FancyArrowPatch(
        posA=(0.5, 0.75), posB=(0.5, 0.66),
        arrowstyle=ptchs.ArrowStyle('-[', lengthB=4.7, widthB=134)
        ))

    coh_ax.add_patch(ptchs.Rectangle((0.17, 0.42),
                                     (1 - mut_prop) * 0.66, 0.23,
                                     facecolor=variant_clrs['WT'],
                                     alpha=0.41, hatch='/', linewidth=1.3,
                                     edgecolor='0.51'))
    coh_ax.add_patch(ptchs.Rectangle((0.17 + (1 - mut_prop) * 0.66, 0.42),
                                     mut_prop * 0.66, 0.23,
                                     facecolor=variant_clrs['Point'],
                                     alpha=0.41, hatch='/', linewidth=1.3,
                                     edgecolor='0.51'))

    coh_ax.text(0.15, 0.52, "{}\nmutated status".format(mtype_str),
                 size=8, ha='right', va='center')

    coh_ax.add_patch(ptchs.Rectangle((0.17 + ovlp_prop * 0.66, 0.12),
                                      np.mean(use_df.rStat) * 0.66, 0.23,
                                      alpha=0.83, hatch='\\',
                                      linewidth=1.3, edgecolor='0.51',
                                      facecolor=variant_clrs['Point']))

    coh_ax.add_patch(ptchs.Rectangle((0.17 + ovlp_prop * 0.66, 0.42),
                                     np.mean(use_df.rStat) * 0.66, 0.23,
                                     hatch='\\', linewidth=1.3,
                                     edgecolor='0.51', facecolor='None'))

    coh_ax.text(0.15 + ovlp_prop * 0.66, 0.23,
                "{} mutations\nother than {}".format(args.gene, mut_str),
                color=variant_clrs['Point'], size=8, ha='right', va='center')
    coh_ax.text(0.17 + ovlp_prop * 0.66 + np.mean(use_df.rStat) * 0.33, 0.09,
                "({} samples)".format(np.sum(use_df.rStat)),
                color=variant_clrs['Point'], size=8, ha='center', va='top')

    diag_ax1 = inset_axes(clf_ax, width='100%', height='100%',
                          loc=10, borderpad=0,
                          bbox_to_anchor=(0, 0, 0.6, 1),
                          bbox_transform=clf_ax.transAxes)
    vio_ax1 = inset_axes(clf_ax, width='100%', height='100%',
                         loc=10, borderpad=0,
                         bbox_to_anchor=(0.6, 0, 0.4, 1),
                         bbox_transform=clf_ax.transAxes)

    diag_ax2 = inset_axes(ovp_ax, width='100%', height='100%',
                          loc=10, borderpad=0,
                          bbox_to_anchor=(0, 0, 0.6, 1),
                          bbox_transform=ovp_ax.transAxes)
    vio_ax2 = inset_axes(ovp_ax, width='100%', height='100%',
                         loc=10, borderpad=0,
                         bbox_to_anchor=(0.6, 0, 0.4, 1),
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
                  "{}\nMutant\n({} samples)".format(
                      mut_str, np.sum(use_df.cStat)),
                  size=8, ha='center', va='center')

    diag_ax1.add_patch(ptchs.Circle(
        (0.5, 0.32), radius=0.31, facecolor=variant_clrs['WT'], alpha=0.41))
    diag_ax1.text(0.5, 0.32,
                  "{}\nWild-Type\n({} samples)".format(
                      mut_str, np.sum(~use_df.cStat)),
                  size=13, ha='center', va='center')

    diag_ax1.text(0.22, 0.67, "classify\nmutations", color='red',
                  size=11, fontstyle='italic', ha='right', va='center')
    diag_ax1.axhline(y=0.67, xmin=0.23, xmax=0.86, color='red',
                     linestyle='--', linewidth=2.3, alpha=0.83)

    diag_ax1.text(0.82, 0.68, "{} (+)".format(np.sum(use_df.cStat)),
                  color='red', size=8, fontstyle='italic', 
                  ha='right', va='bottom')
    diag_ax1.text(0.82, 0.655, "{} (\u2212)".format(np.sum(~use_df.cStat)),
                  color='red', size=8, fontstyle='italic',
                  ha='right', va='top')

    sns.violinplot(data=use_df[~use_df.cStat], y='Value', ax=vio_ax1,
                   palette=[variant_clrs['WT']], linewidth=0, cut=0)
    sns.violinplot(data=use_df[use_df.cStat], y='Value', ax=vio_ax1,
                   palette=[variant_clrs['Point']], linewidth=0, cut=0)

    vio_ax1.text(0.5, 0.99,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value, use_df.cStat)),
                 color='red', size=10, fontstyle='italic',
                 ha='center', va='top', transform=vio_ax1.transAxes)

    diag_ax2.add_patch(ptchs.Wedge((0.48, 0.85), 0.14, 90, 270,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51'))

    diag_ax2.add_patch(ptchs.Wedge((0.52, 0.85), 0.14, 270, 90,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, hatch='/', linewidth=0.8,
                                   edgecolor='0.51'))
    diag_ax2.add_patch(ptchs.Wedge((0.52, 0.85), 0.14, 270, 90,
                                   facecolor='None', edgecolor='0.61',
                                   hatch='\\', linewidth=0.8))

    diag_ax2.text(0.22, 0.67, "same classifier\nresults", color='red',
                  size=8, fontstyle='italic', ha='right', va='center')
    diag_ax2.axhline(y=0.67, xmin=0.23, xmax=0.86, color='red',
                     linestyle='--', linewidth=0.8, alpha=0.67)

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

    diag_ax2.text(0.33, 0.85,
                  "{}\nMutant\nw/o overlap\n({} samps)".format(
                      mut_str, np.sum(use_df.cStat & ~use_df.rStat)),
                  size=9, ha='right', va='center')
    diag_ax2.text(0.67, 0.85,
                  "{}\nMutant\nw/ overlap\n({} samps)".format(
                      mut_str, np.sum(use_df.cStat & use_df.rStat)),
                  size=9, ha='left', va='center')

    diag_ax2.text(0.47, 0.32,
                  "{}\nWild-Type\nw/o overlap\n({} samps)".format(
                      mut_str, np.sum(~use_df.cStat & ~use_df.rStat)),
                  size=10, ha='right', va='center')
    diag_ax2.text(0.53, 0.32,
                  "{}\nWild-Type\nw/ overlap\n({} samps)".format(
                      mut_str, np.sum(~use_df.cStat & use_df.rStat)),
                  size=10, ha='left', va='center')

    sns.violinplot(data=use_df[~use_df.cStat], x='cStat', y='Value',
                   hue='rStat', palette=[variant_clrs['WT']],
                   hue_order=[False, True], split=True, linewidth=0,
                   cut=0, ax=vio_ax2)
    sns.violinplot(data=use_df[use_df.cStat], x='cStat', y='Value',
                   hue='rStat', palette=[variant_clrs['Point']],
                   hue_order=[False, True], split=True, linewidth=0,
                   cut=0, ax=vio_ax2)

    vals_min, vals_max = use_df.Value.quantile(q=[0, 1])
    vals_rng = (vals_max - vals_min) / 51
    vio_ax1.set_ylim(vals_min - vals_rng, vals_max + 4 * vals_rng)
    vio_ax2.set_ylim(vals_min - vals_rng, vals_max + 2 * vals_rng)

    vio_ax2.get_legend().remove()
    diag_ax2.axvline(x=0.5, ymin=-0.03, ymax=1.03, clip_on=False,
                     color=variant_clrs['Point'], linewidth=1.1, alpha=0.81,
                     linestyle=':')

    diag_ax2.text(0.5, -0.05,
                  "partition scored samples according to\noverlap with "
                  "PIK3CA mutations\nthat are not {}".format(mut_str),
                  color=variant_clrs['Point'], size=9,
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
        clr_face = vio_ax2.get_children()[i].get_facecolor()[0]
        clr_face[-1] = 0.41
        vio_ax2.get_children()[i].set_facecolor(clr_face)

    for i in [0, 3]:
        vio_ax2.get_children()[i].set_linewidth(0.8)
        vio_ax2.get_children()[i].set_hatch('/')
        vio_ax2.get_children()[i].set_edgecolor('0.61')

    for i in [1, 4]:
        vio_ax2.get_children()[i].set_linewidth(1.0)
        vio_ax2.get_children()[i].set_hatch('/\\')
        vio_ax2.get_children()[i].set_edgecolor('0.47')

    vio_ax2.text(0.23, 0.96, "{} w/o overlap".format(mut_str),
                 color=variant_clrs['Point'], size=7,
                 fontstyle='italic', ha='center', va='bottom',
                 transform=vio_ax2.transAxes)
    vio_ax2.text(0.23, 0.95,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value[~use_df.rStat],
                                               use_df.cStat[~use_df.rStat])),
                 color='red', size=10, fontstyle='italic',
                 ha='center', va='top', transform=vio_ax2.transAxes)

    vio_ax2.text(0.77, 0.96, "{} w/ overlap".format(mut_str),
                 color=variant_clrs['Point'], size=7,
                 fontstyle='italic', ha='center', va='bottom',
                 transform=vio_ax2.transAxes)
    vio_ax2.text(0.77, 0.95,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value[use_df.rStat],
                                               use_df.cStat[use_df.rStat])),
                 color='red', size=10, fontstyle='italic',
                 ha='center', va='top', transform=vio_ax2.transAxes)

    plt.tight_layout(pad=-0.2, w_pad=0, h_pad=0.5)
    plt.savefig(os.path.join(
        plot_dir, args.cohort, "base_classification_{}_samps-{}.svg".format(
            args.gene, args.samp_cutoff)
        ),
        dpi=300, bbox_inches='tight', format='svg')

    plt.close()


def plot_iso_classification(mtype, use_vals, cdata, args):
    fig, axarr = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

    all_mtype = MuType(cdata.train_mut.allkey())
    all_stat = np.array(cdata.train_pheno(all_mtype - mtype))
    mtype_str = ":".join([args.gene, str(mtype).split(':')[-1][2:]])
    mut_str = mtype_str.split(':')[-1]

    use_mcombs = [('All', mtype), ('Ex', ExMcomb(cdata.train_mut, mtype))]
    mcomb_stats = {lbl: np.array(cdata.train_pheno(mtp))
                   for lbl, mtp in use_mcombs}

    mcomb_masks = [('All', {lbl: np.array([True] * len(cdata.train_samps))
                            for lbl in mcomb_stats}),
                   ('Iso', {lbl: ~(all_stat & ~stat)
                            for lbl, stat in mcomb_stats.items()})]

    for i, (smp_lbl, msk) in enumerate(mcomb_masks):
        for j, (mtp_lbl, mtp) in enumerate(use_mcombs):
            vals_df = pd.DataFrame({'Value': use_vals[smp_lbl].loc[mtp],
                                    'cStat': mcomb_stats[mtp_lbl],
                                    'uStat': msk[mtp_lbl]})

            diag_ax = inset_axes(axarr[i, j], width='100%', height='100%',
                                 loc=10, borderpad=0,
                                 bbox_to_anchor=(0, 0, 0.5, 1),
                                 bbox_transform=axarr[i, j].transAxes)
            vio_ax = inset_axes(axarr[i, j], width='100%', height='100%',
                                loc=10, borderpad=0,
                                bbox_to_anchor=(0.55, 0, 0.45, 1),
                                bbox_transform=axarr[i, j].transAxes)

            axarr[i, j].axis('off')
            diag_ax.axis('off')
            diag_ax.set_aspect('equal')

            diag_ax.text(-0.01, 0.67, "classify\nmutations",
                         color='red', size=8, fontstyle='italic',
                         ha='right', va='center')
            diag_ax.axhline(y=0.67, xmin=0, xmax=0.82, color='red',
                            linestyle='--', linewidth=1.6, alpha=0.83)
 
            diag_ax.text(0.82, 0.68,
                         "{} (+)".format(
                             np.sum(vals_df.cStat[vals_df.uStat])),
                         color='red', size=7, fontstyle='italic',
                         ha='right', va='bottom')
            diag_ax.text(0.82, 0.655,
                         "{} (\u2212)".format(
                             np.sum(~vals_df.cStat[vals_df.uStat])),
                         color='red', size=7, fontstyle='italic',
                         ha='right', va='top')

            sns.violinplot(data=vals_df[~vals_df.cStat][vals_df.uStat],
                           y='Value', ax=vio_ax, palette=[variant_clrs['WT']],
                           linewidth=0, cut=0)
            sns.violinplot(data=vals_df[vals_df.cStat][vals_df.uStat],
                           y='Value', ax=vio_ax,
                           palette=[variant_clrs['Point']],
                           linewidth=0, cut=0)

            vio_ax.text(0.5, 0.99,
                        "AUC: {:.3f}".format(
                            calc_auc(vals_df.Value[vals_df.uStat],
                                     vals_df.cStat[vals_df.uStat])
                            ),
                        color='red', size=11, fontstyle='italic',
                        ha='center', va='top', transform=vio_ax.transAxes)

            vals_min, vals_max = vals_df.Value[
                vals_df.uStat].quantile(q=[0, 1])
            vals_rng = (vals_max - vals_min) / 51
            vio_ax.set_ylim(vals_min - vals_rng, vals_max + 3 * vals_rng)

            vio_ax.get_children()[0].set_alpha(0.41)
            vio_ax.get_children()[2].set_alpha(0.41)

            diag_ax.add_patch(ptchs.Wedge((0.4, 0.95), 0.25, 90, 270,
                                          facecolor=variant_clrs['Point'],
                                          alpha=0.41, clip_on=False,
                                          transform=diag_ax.transData))
            diag_ax.text(0.38, 0.95,
                         "{}\nMutant\nw/o overlap\n({} samps)".format(
                             mut_str, np.sum(vals_df.cStat & ~all_stat)),
                         size=6, ha='right', va='center')

            if np.sum(vals_df.cStat & all_stat):
                diag_ax.add_patch(ptchs.Wedge((0.4, 0.95), 0.25, 270, 90,
                                              facecolor=variant_clrs['Point'],
                                              alpha=0.41, clip_on=False,
                                              transform=diag_ax.transData))
                diag_ax.add_patch(ptchs.Wedge(
                    (0.4, 0.95), 0.25, 270, 90, facecolor='None',
                    edgecolor='0.57', clip_on=False, linewidth=0.7,
                    hatch='\\', transform=diag_ax.transData
                    ))

                diag_ax.text(0.42, 0.95,
                             "{}\nMutant\nw/ overlap\n({} samps)".format(
                                 mut_str, np.sum(vals_df.cStat & all_stat)),
                             size=6, ha='left', va='center')

            diag_ax.add_patch(ptchs.Wedge((0.4, 0.22), 0.42, 90, 270,
                                          facecolor=variant_clrs['WT'],
                                          alpha=0.41, clip_on=False,
                                          transform=diag_ax.transData))
            diag_ax.text(0.38, 0.22,
                         "{}\nWild-Type\nw/o overlap\n({} samps)".format(
                             mut_str, np.sum(~vals_df.cStat & ~all_stat)),
                         size=9, ha='right', va='center')

            if np.sum(~vals_df.cStat & all_stat & vals_df.uStat):
                diag_ax.add_patch(ptchs.Wedge((0.4, 0.22), 0.42, 270, 90,
                                              facecolor=variant_clrs['WT'],
                                              alpha=0.41, clip_on=False,
                                              transform=diag_ax.transData))
                diag_ax.add_patch(ptchs.Wedge(
                    (0.4, 0.22), 0.42, 270, 90, facecolor='None',
                    edgecolor='0.57', clip_on=False, linewidth=0.7,
                    hatch='\\', transform=diag_ax.transData
                    ))

                diag_ax.text(0.42, 0.22,
                             "{}\nWild-Type\nw/ overlap\n({} samps)".format(
                                 mut_str, np.sum(~vals_df.cStat & all_stat)),
                             size=9, ha='left', va='center')

            diag_ax.add_patch(ptchs.FancyArrow(
                0.89, 0.51, dx=0.11, dy=0, width=0.02, clip_on=False,
                length_includes_head=True, head_length=0.05, alpha=0.93,
                linewidth=1.5, facecolor='None', edgecolor='black'
                ))

            vio_ax.set_xticks([])
            vio_ax.set_xticklabels([])
            vio_ax.set_yticklabels([])
            vio_ax.yaxis.label.set_visible(False)

    plt.tight_layout(pad=0, w_pad=2.3, h_pad=0)
    plt.savefig(os.path.join(
        plot_dir, args.cohort, "iso_classification_{}_samps-{}.svg".format(
            args.gene, args.samp_cutoff)
        ),
        dpi=300, bbox_inches='tight', format='svg')

    plt.close()


def plot_iso_projection(mtype, use_vals, cdata, args):
    fig, ((base_ax, pnt_ax), (loss_ax, gain_ax)) = plt.subplots(
        figsize=(11, 7), nrows=2, ncols=2)

    all_mtype = MuType(cdata.train_mut.allkey())
    all_stat = np.array(cdata.train_pheno(all_mtype - mtype))
    mtype_str = ":".join([args.gene, str(mtype).split(':')[-1][2:]])
    mut_str = mtype_str.split(':')[-1]

    use_mcomb = ExMcomb(cdata.train_mut, mtype)
    mcomb_stat = np.array(cdata.train_pheno(use_mcomb))
    vals_df = pd.DataFrame({'Value': use_vals.loc[use_mcomb],
                            'cStat': np.array(cdata.train_pheno(mtype)),
                            'rStat': np.array(cdata.train_pheno(all_mtype
                                                                - mtype))})

    for ax in base_ax, pnt_ax, loss_ax, gain_ax:
        ax.set_aspect('equal')
        ax.axis('off')

    base_diag_ax = inset_axes(base_ax, width='100%', height='100%', loc=10,
                              borderpad=0, bbox_to_anchor=(0, 0, 0.52, 1),
                              bbox_transform=base_ax.transAxes)
    base_vio_ax = inset_axes(base_ax, width='100%', height='100%', loc=10,
                             borderpad=0, bbox_to_anchor=(0.57, 0, 0.6, 1),
                             bbox_transform=base_ax.transAxes)

    base_diag_ax.axis('off')
    base_diag_ax.set_aspect('equal')

    base_diag_ax.text(-0.12, 0.67, "1) classify\nmutations", color='red',
                      size=8, fontstyle='italic', ha='right', va='center')
    base_diag_ax.axhline(y=0.67, xmin=-0.12, xmax=0.1, color='red',
                         alpha=0.83, clip_on=False,
                         linestyle='--', linewidth=1.5)

    base_diag_ax.text(0.46, 1.18, "2) apply trained\nclassifier", color='red',
                      size=8, fontstyle='italic', ha='center', va='bottom')
    base_diag_ax.axhline(y=0.67, xmin=0.16, xmax=0.86, color='red',
                         linestyle=':', linewidth=1, alpha=0.57)
 
    vals_min, vals_max = vals_df.Value.quantile(q=[0, 1])
    vals_rng = (vals_max - vals_min) / 51
    base_vio_ax.set_ylim(vals_min - vals_rng, vals_max + 3 * vals_rng)

    base_diag_ax.add_patch(ptchs.Wedge((0.07, 1), 0.27, 90, 270,
                                       facecolor=variant_clrs['Point'],
                                       alpha=0.41, clip_on=False,
                                       transform=base_diag_ax.transData))
    base_diag_ax.text(0.05, 1.01,
                      "{}\nMutant\nw/o overlap\n({} samps)".format(
                          mut_str, np.sum(vals_df.cStat & ~all_stat)),
                      size=6, ha='right', va='center')

    base_diag_ax.add_patch(ptchs.Wedge((0.07, 0.17), 0.45, 90, 270,
                                       facecolor=variant_clrs['WT'],
                                       alpha=0.41, clip_on=False,
                                       transform=base_diag_ax.transData))
    base_diag_ax.text(0.05, 0.17,
                      "{}\nWild-Type\nw/o overlap\n({} samps)".format(
                          mut_str, np.sum(~vals_df.cStat & ~all_stat)),
                      size=9, ha='right', va='center')

    base_diag_ax.add_patch(ptchs.Wedge((0.19, 0.67), 0.2, 270, 90,
                                       facecolor=variant_clrs['Point'],
                                       edgecolor='0.31', clip_on=False,
                                       hatch='\\', alpha=0.25, linewidth=1.7,
                                       transform=base_diag_ax.transData))

    base_diag_ax.text(0.2, 0.67,
                      "{}\nMutant\nw/ overlap\n({} samps)".format(
                          mut_str, np.sum(vals_df.cStat & all_stat)),
                      size=5, alpha=0.41, ha='left', va='center')

    base_diag_ax.add_patch(ptchs.Wedge((0.46, 0.67), 0.35, 270, 90,
                                       facecolor=variant_clrs['WT'],
                                       edgecolor='0.31', clip_on=False, 
                                       hatch='\\', alpha=0.25, linewidth=1.7,
                                       transform=base_diag_ax.transData))

    base_diag_ax.text(0.47, 0.67,
                      "{}\nWild-Type\nw/ overlap\n({} samps)".format(
                          mut_str, np.sum(~vals_df.cStat & all_stat)),
                      size=7, alpha=0.41, ha='left', va='center')

    base_diag_ax.add_patch(ptchs.FancyArrow(
        0.91, 0.51, dx=0.11, dy=0, width=0.03, clip_on=False,
        length_includes_head=True, head_length=0.05, alpha=0.93,
        linewidth=1.3, facecolor='None', edgecolor='black'
        ))

    sns.violinplot(data=vals_df[~vals_df.cStat], x='cStat',
                   y='Value', hue='rStat', palette=[variant_clrs['WT']],
                   hue_order=[False, True], split=True, linewidth=0, cut=0,
                   ax=base_vio_ax)
    sns.violinplot(data=vals_df[vals_df.cStat], x='cStat',
                   y='Value', hue='rStat', palette=[variant_clrs['Point']],
                   hue_order=[False, True], split=True, linewidth=0, cut=0,
                   ax=base_vio_ax)

    for i in [0, 1, 3, 4]:
        base_vio_ax.get_children()[i].set_alpha(0.41)

    for i in [1, 4]:
        base_vio_ax.get_children()[i].set_linewidth(0.9)
        base_vio_ax.get_children()[i].set_edgecolor('0.31')
        base_vio_ax.get_children()[i].set_hatch('\\')

    base_vio_ax.text(0.25, 0.99,
                     "AUC: {:.3f}".format(
                         calc_auc(vals_df.Value[~vals_df.rStat],
                                  vals_df.cStat[~vals_df.rStat])
                        ),
                     color='red', size=7, fontstyle='italic',
                     ha='center', va='top', transform=base_vio_ax.transAxes)

    base_vio_ax.text(0.75, 0.99,
                     "AUC: {:.3f}".format(
                         calc_auc(vals_df.Value[vals_df.rStat],
                                  vals_df.cStat[vals_df.rStat])
                        ),
                     color='red', size=7, fontstyle='italic',
                     ha='center', va='top', transform=base_vio_ax.transAxes)

    base_vio_ax.set_xticks([])
    base_vio_ax.set_xticklabels([])
    base_vio_ax.set_yticklabels([])
    base_vio_ax.xaxis.label.set_visible(False)
    base_vio_ax.yaxis.label.set_visible(False)
    base_vio_ax.get_legend().remove()

    for ax, lbl in zip([pnt_ax, loss_ax, gain_ax], ['Point', 'Loss', 'Gain']):
        if lbl == 'Point':
            use_mtype = (dict(variant_mtypes)['Point'] & all_mtype) - mtype
        else:
            use_mtype = dict(variant_mtypes)[lbl]

        diag_ax = inset_axes(ax, width='100%', height='100%', loc=10,
                             borderpad=0, bbox_to_anchor=(0, 0, 0.52, 1),
                             bbox_transform=ax.transAxes)
        vio_ax = inset_axes(ax, width='100%', height='100%', loc=10,
                            borderpad=0, bbox_to_anchor=(0.57, 0, 0.6, 1),
                            bbox_transform=ax.transAxes)

        vals_df['mStat'] = np.array(cdata.train_pheno(use_mtype))
        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

        diag_ax.axhline(y=0.67, xmin=-0.22, xmax=0.1, color='red',
                        alpha=0.83, clip_on=False,
                        linestyle='--', linewidth=1.5)

        diag_ax.text(0.46, 1.18, "2) apply trained\nclassifier", color='red',
                     size=8, fontstyle='italic', ha='center', va='bottom')
        diag_ax.axhline(y=0.67, xmin=0.16, xmax=0.86, color='red',
                        linestyle=':', linewidth=1, alpha=0.57)
 
        vals_min, vals_max = vals_df.Value.quantile(q=[0, 1])
        vals_rng = (vals_max - vals_min) / 51
        vio_ax.set_ylim(vals_min - vals_rng, vals_max + 3 * vals_rng)

        diag_ax.add_patch(ptchs.Wedge((0.07, 1), 0.27, 90, 270,
                                      facecolor=variant_clrs['Point'],
                                      alpha=0.41, clip_on=False,
                                      transform=diag_ax.transData))
        diag_ax.text(0.05, 1.01,
                     "{}\nMutant\nw/o overlap\n({} samps)".format(
                         mut_str, np.sum(vals_df.cStat & ~all_stat)),
                     size=6, ha='right', va='center')

        diag_ax.add_patch(ptchs.Wedge((0.07, 0.17), 0.45, 90, 270,
                                      facecolor=variant_clrs['WT'],
                                      alpha=0.41, clip_on=False,
                                      transform=diag_ax.transData))
        diag_ax.text(0.05, 0.17,
                     "{}\nWild-Type\nw/o overlap\n({} samps)".format(
                         mut_str, np.sum(~vals_df.cStat & ~all_stat)),
                     size=9, ha='right', va='center')

        diag_ax.add_patch(ptchs.Wedge((0.19, 0.67), 0.2, 270, 90,
                                      facecolor=variant_clrs['Point'],
                                      edgecolor=variant_clrs[lbl],
                                      clip_on=False, hatch='\\', alpha=0.25,
                                      linewidth=1.7, transform=diag_ax.transData))

        diag_ax.text(0.2, 0.67,
                     "{}\nMutant\nw/ {}\n({} samps)".format(
                         mut_str, lbl, np.sum(vals_df.cStat & vals_df.mStat)),
                     size=5, alpha=0.41, ha='left', va='center')

        diag_ax.add_patch(ptchs.Wedge((0.46, 0.67), 0.35, 270, 90,
                                      facecolor=variant_clrs['WT'],
                                      edgecolor=variant_clrs[lbl], clip_on=False, 
                                      hatch='\\', alpha=0.25, linewidth=1.7,
                                      transform=diag_ax.transData))

        diag_ax.text(0.47, 0.67,
                     "{}\nWild-Type\nw/ {}\n({} samps)".format(
                         mut_str, lbl,
                         np.sum(~vals_df.cStat & vals_df.mStat)
                        ),
                     size=7, alpha=0.41, ha='left', va='center')

        diag_ax.add_patch(ptchs.FancyArrow(
            0.91, 0.51, dx=0.11, dy=0, width=0.03, clip_on=False,
            length_includes_head=True, head_length=0.05, alpha=0.93,
            linewidth=1.3, facecolor='None', edgecolor='black'
            ))

        sns.violinplot(data=vals_df[~vals_df.cStat & ~vals_df.rStat],
                       x='cStat', y='Value', hue='rStat',
                       palette=[variant_clrs['WT']], hue_order=[False, True],
                       split=True, linewidth=0, cut=0, ax=vio_ax)
        sns.violinplot(data=vals_df[vals_df.cStat & ~vals_df.rStat],
                       x='cStat', y='Value', hue='rStat',
                       palette=[variant_clrs['Point']],
                       hue_order=[False, True], split=True, linewidth=0,
                       cut=0, ax=vio_ax)

        sns.violinplot(data=vals_df[~vals_df.cStat & vals_df.mStat],
                       x='cStat', y='Value', hue='mStat',
                       palette=[variant_clrs['WT']], hue_order=[False, True],
                       split=True, linewidth=0, cut=0, ax=vio_ax)
        sns.violinplot(data=vals_df[vals_df.cStat & vals_df.mStat],
                       x='cStat', y='Value', hue='mStat',
                       palette=[variant_clrs['Point']], hue_order=[False, True],
                       split=True, linewidth=0, cut=0, ax=vio_ax)

        for i in [0, 1]:
            vio_ax.get_children()[i].set_alpha(0.41)

        for i in [2, 4]:
            vio_ax.get_children()[i].set_alpha(0.47)
            vio_ax.get_children()[i].set_linewidth(0.9)
            vio_ax.get_children()[i].set_edgecolor(variant_clrs[lbl])
            vio_ax.get_children()[i].set_hatch('\\')

        vio_ax.text(0.25, 0.99,
                    "AUC: {:.3f}".format(
                        calc_auc(vals_df.Value[~vals_df.rStat],
                                 vals_df.cStat[~vals_df.rStat])
                        ),
                     color='red', size=7, fontstyle='italic',
                     ha='center', va='top', transform=vio_ax.transAxes)

        vio_ax.text(0.75, 0.99,
                    "AUC: {:.3f}".format(
                        calc_auc(vals_df.Value[vals_df.mStat],
                                 vals_df.cStat[vals_df.mStat])
                        ),
                     color='red', size=7, fontstyle='italic',
                     ha='center', va='top', transform=vio_ax.transAxes)

        vio_ax.set_xticks([])
        vio_ax.set_xticklabels([])
        vio_ax.set_yticklabels([])
        vio_ax.xaxis.label.set_visible(False)
        vio_ax.yaxis.label.set_visible(False)
        vio_ax.get_legend().remove()

    plt.tight_layout(pad=0, w_pad=-3, h_pad=0)
    plt.savefig(os.path.join(
        plot_dir, args.cohort, "iso_projection_{}_samps-{}.svg".format(
            args.gene, args.samp_cutoff)
        ),
        dpi=300, bbox_inches='tight', format='svg')

    plt.close()


def plot_iso_similarities(use_mtype, use_vals, cdata, args):
    fig, (vio_ax, sim_ax, clr_ax) = plt.subplots(
        figsize=(11, 7), nrows=1, ncols=3,
        gridspec_kw=dict(width_ratios=[2, 23, 1])
        )

    use_mcomb = ExMcomb(cdata.train_mut, use_mtype)
    all_mtype = MuType(cdata.train_mut.allkey())
    mtype_str = str(use_mtype).split(':')[-1][2:]

    vals_df = pd.DataFrame({
        'Value': use_vals.loc[use_mcomb],
        'cStat': np.array(cdata.train_pheno(use_mcomb)),
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
    vio_ax.axhline(y=wt_mean, xmin=0, xmax=14,
                   color=variant_clrs['WT'], clip_on=False, linestyle='--',
                   linewidth=1.6, alpha=0.51)

    mut_mean = np.mean(vals_df.Value[vals_df.cStat & ~vals_df.rStat])
    vio_ax.axhline(y=mut_mean, xmin=0, xmax=14,
                   color=variant_clrs['Point'], clip_on=False, linestyle='--',
                   linewidth=1.6, alpha=0.51)

    vio_ax.text(-0.52, wt_mean, "0",
                size=12, fontstyle='italic', ha='right', va='center')
    vio_ax.text(-0.52, mut_mean, "1",
                size=12, fontstyle='italic', ha='right', va='center')

    vio_ax.text(0, vals_min - vals_rng,
                "Isolated\nClassification\n of {}\n(M1)".format(mtype_str),
                size=13, fontweight='semibold', ha='center', va='top')

    sim_mcombs = {mcomb: ExMcomb(cdata.train_mut,
                                 *[mtype & all_mtype - use_mtype
                                   for mtype in mcomb.mtypes])
                  for mcomb in use_vals.index
                  if (isinstance(mcomb, ExMcomb) and mcomb != use_mcomb
                      and all(('Copy' in mtype.get_levels()
                               and len(mtype.subkeys()) == 2)
                              or 'Copy' not in mtype.get_levels()
                              for mtype in mcomb.mtypes))}

    sim_df = pd.concat([
        pd.DataFrame({
            'Mcomb': mcomb, 'Value': use_vals.loc[
                use_mcomb, np.array(cdata.train_pheno(ex_mcomb))]
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
        sim_ax.get_children()[i * 2].set_alpha(9/11)

        mcomb_lbl = str(mcomb).replace('Point:', '').replace('Copy:', '')
        mcomb_lbl = mcomb_lbl.replace(' ', '\n')

        mcomb_lbl = mcomb_lbl.replace('(DeepGain|ShalGain)', 'gain')
        mcomb_lbl = mcomb_lbl.replace('(DeepDel|ShalDel)', 'deletion')
        mcomb_lbl = mcomb_lbl.replace('Point', 'any other\npoint mutation')

        sim_ax.text(i, mcomb_mins[mcomb] - vals_rng / 2,
                    "{}\n({} samples)".format(mcomb_lbl, mcomb_sizes[mcomb]),
                    size=10, ha='center', va='top')
        sim_ax.text(i, mcomb_maxs[mcomb] + vals_rng / 2, format(scr, '.2f'),
                    size=11, fontstyle='italic', ha='center', va='bottom')

    sim_ax.text(len(mcomb_scores) / 2, vals_min - 2 * vals_rng,
                "{} Classifier Scoring\nof Other "
                "Isolated {} Mutations\n(M2)".format(mtype_str, args.gene),
                size=13, fontweight='semibold', ha='center', va='top')

    for ax in vio_ax, sim_ax:
        ax.axis('off')
        ax.set_ylim(vals_min - vals_rng, vals_max + vals_rng)

    clr_min = clr_norm((vals_min - vals_rng - wt_mean) / (mut_mean - wt_mean))
    clr_max = clr_norm((vals_max + vals_rng - wt_mean) / (mut_mean - wt_mean))
    clr_ext = min(0.2, -clr_min, clr_max - 1)

    clr_bar = ColorbarBase(ax=clr_ax, cmap=simil_cmap, norm=clr_norm,
                           extend='both', extendfrac=clr_ext,
                           ticks=[-0.5, 0, 0.5, 1.0, 1.5])

    clr_bar.set_ticklabels(
        ['M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'])
    clr_ax.set_ylim(clr_min, clr_max)
    clr_ax.tick_params(labelsize=12)

    plt.tight_layout(pad=0, h_pad=0, w_pad=-2)
    plt.savefig(os.path.join(
        plot_dir, args.cohort, "iso_similarities_{}_samps-{}.svg".format(
            args.gene, args.samp_cutoff)
        ),
        dpi=300, bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot an example diagram showing how overlap with other types of "
        "mutations can affect a mutation classification task."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)

    # load cohort expression and mutation data, get list of inference
    # experiments that have been run for the given gene in this cohort
    cdata = load_cohort_data(base_dir, args.cohort, args.gene, 'Protein')
    out_path = Path(base_dir, 'output', args.cohort, args.gene)
    out_glob = "*/samps_{}/Protein/out__task-0.p".format(args.samp_cutoff)
    out_dirs = [pth.parent for pth in out_path.glob(out_glob)]

    # get the classifiers associated with experiments that ran to completion
    use_clfs = [out_dir.parent.parent.stem for out_dir in out_dirs
                if (len(tuple(out_dir.glob('out__task-*.p')))
                    == len(tuple(out_dir.glob('slurm/fit-*.txt*'))))]

    # load data from classifiers whose inference experiments have finished
    infer_dicts = {
        clf: load_infer_output(os.path.join(
            base_dir, 'output', args.cohort, args.gene, clf,
            'samps_{}'.format(args.samp_cutoff), 'Protein'
            ))
        for clf in use_clfs
        }

    # check the integrity of inference outputs and average inferred
    # values across cross-validation runs
    check_output(infer_dicts.values())
    infer_dicts = {clf: {smps: vals.applymap(np.mean)
                         for smps, vals in infer_dict.items()}
                   for clf, infer_dict in infer_dicts.items()}

    # get list of hotspot mutations that appear in enough samples by
    # themselves to have had inferred values calculated for them
    base_mtypes = [{'All': mtype, 'Ex': ExMcomb(cdata.train_mut, mtype)}
                   for mtype in infer_dicts[use_clfs[0]]['All'].index
                   if (isinstance(mtype, MuType)
                       and 'Protein' in mtype.get_levels()
                       and (ExMcomb(cdata.train_mut, mtype)
                            in infer_dicts[use_clfs[0]]['Iso'].index))]

    # for each hotspot mutation, get the presence of the mutation with and
    # without overlapping mutations in the samples belonging to the cohort
    mcomb_stats = {mcomb: np.array(cdata.train_pheno(mcomb))
                   for mtps in base_mtypes for mcomb in mtps.values()}
    all_stat = np.array(cdata.train_pheno(MuType(cdata.train_mut.allkey())))

    # get the samples included in each inference experiment
    mcomb_masks = {
        mtps['All']: {'All': {mtp_lbl: np.array([True] * len(cdata.samples))
                              for mtp_lbl in mtps},
                      'Iso': {mtp_lbl: ~(all_stat & ~mcomb_stats[mtp])
                              for mtp_lbl, mtp in mtps.items()}}
        for mtps in base_mtypes
        }

    # calculate the performance of the classifier in predicting the presence
    # of the given mutation type in each experiment
    auc_dict = {
        (use_clf, mtps['All']): pd.DataFrame.from_dict({
            mtp_lbl: {
                smps: calc_auc(
                    infer_dicts[use_clf][smps].loc[
                        mcomb, mcomb_masks[mtps['All']][smps][mtp_lbl]],
                    mcomb_stats[mcomb][
                        mcomb_masks[mtps['All']][smps][mtp_lbl]]
                    )
                for smps in ['All', 'Iso']
                }
            for mtp_lbl, mcomb in mtps.items()
            })
        for use_clf, mtps in product(use_clfs, base_mtypes)
        }

    # find experiments where the classifier performed well and also with an
    # improvement when samples with overlapping mutations were removed
    good_exs = {k for k, aucs in auc_dict.items()
                if (aucs['All']['All'] > 0.7
                    and aucs['Ex']['Iso'] > aucs['All']['All'])}
    off_diags = {k: auc_dict[k].values[~np.equal(*np.indices((2, 2)))]
                 for k in good_exs}

    use_clf, use_mtype = sorted(
        [(k, max(auc_dict[k]['All']['All'] - np.min(off_diags[k]),
                 np.max(off_diags[k]) - auc_dict[k]['Ex']['Iso']))
         for k in good_exs],
        key=itemgetter(1)
        )[0][0]

    plot_base_classification(use_mtype, infer_dicts.copy()[use_clf]['All'],
                             cdata, args)
    plot_iso_classification(use_mtype, infer_dicts.copy()[use_clf],
                            cdata, args)

    plot_iso_projection(use_mtype, infer_dicts.copy()[use_clf]['Iso'],
                        cdata, args)
    plot_iso_similarities(use_mtype, infer_dicts.copy()[use_clf]['Iso'],
                          cdata, args)


if __name__ == '__main__':
    main()

