
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
    load_infer_output, calc_auc)
from HetMan.experiments.subvariant_infer import (
    variant_mtypes, variant_clrs, MuType)

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


def plot_base_classification(mtype, use_vals, cdata, args):
    fig, ((coh_ax1, diag_ax1, clf_ax1),
          (coh_ax2, diag_ax2, clf_ax2)) = plt.subplots(
              figsize=(7, 8), nrows=2, ncols=3,
              gridspec_kw=dict(width_ratios=[2, 4, 5])
            )

    mtype_str = ":".join([args.gene, str(mtype).split(':')[-1][2:]])
    mut_str = mtype_str.split(':')[-1]
    rest_mtype = MuType(cdata.train_mut.allkey()) - mtype

    use_df = pd.DataFrame({'Value': use_vals.loc[mtype],
                           'cStat': np.array(cdata.train_pheno(mtype)),
                           'rStat': np.array(cdata.train_pheno(rest_mtype))})
    mut_prop = np.sum(use_df.cStat) / len(cdata.samples)
    ovlp_prop = np.mean(~use_df.rStat[~use_df.cStat]) * (1 - mut_prop)

    coh_ax1.text(-0.21, -0.02,
                 "TCGA-{}\n({} samples)".format(
                     args.cohort, len(cdata.samples)),
                 size=11, ha='center', va='center', rotation=90)
    coh_ax1.add_patch(ptchs.FancyArrowPatch(
        posA=(-0.04, -0.038), posB=(-0.02, -0.038), clip_on=False,
        arrowstyle=ptchs.ArrowStyle('-[', lengthB=4.3, widthB=149)
        ))

    coh_ax2.add_patch(ptchs.Rectangle(
        (0.04, 0.49), 0.23, (1 - mut_prop) * 1.1, clip_on=False,
        facecolor=variant_clrs['WT'], alpha=0.41
        ))
    coh_ax2.add_patch(ptchs.Rectangle(
        (0.04, 0.49 + (1 - mut_prop) * 1.1), 0.23, mut_prop * 1.1,
        clip_on=False, alpha=0.41, linewidth=0.3,
        edgecolor=variant_clrs['Point'], facecolor=variant_clrs['Point']
        ))

    coh_ax1.text(0.25, 0.62, "{}\nmutated status".format(mtype_str),
                 size=8, rotation=315, ha='right', va='center')

    coh_ax2.add_patch(ptchs.Rectangle((0.33, 0.49 + ovlp_prop * 1.1),
                                      0.23, np.mean(use_df.rStat) * 1.1,
                                      clip_on=False, alpha=0.83,
                                      facecolor=variant_clrs['Point']))
    coh_ax2.add_patch(ptchs.Rectangle(
        (0.33, 0.49 + ovlp_prop * 1.1), 0.23, np.mean(use_df.rStat) * 1.1,
        clip_on=False, alpha=0.83, linewidth=0.9, edgecolor='black',
        facecolor='None'
        ))

    coh_ax2.add_patch(ptchs.Rectangle((0.04, 0.49 + ovlp_prop * 1.1),
                                      0.23, np.mean(use_df.rStat) * 1.1,
                                      clip_on=False, edgecolor='black',
                                      facecolor='None', linewidth=0.9))

    coh_ax2.text(0.6, 0.52 + ovlp_prop * 1.1,
                 "{} mutations\nother than {}\n({} samples)".format(
                     args.gene, mut_str, np.sum(use_df.rStat)),
                 color=variant_clrs['Point'], size=8, fontstyle='italic',
                 ha='left', va='bottom')

    for coh_ax in coh_ax1, coh_ax2:
        coh_ax.axis('off')
        coh_ax.set_xlim(0, 1)
        coh_ax.set_ylim(0, 1)

    for diag_ax in diag_ax1, diag_ax2:
        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

        diag_ax.add_patch(ptchs.FancyArrow(
            0.92, 0.51, dx=0.17, dy=0, width=0.04, length_includes_head=True,
            head_length=0.08, clip_on=False, alpha=0.91, linewidth=1.9,
            facecolor='white', edgecolor='black'
            ))

    diag_ax1.add_patch(ptchs.Circle(
        (0.4, 0.95), radius=0.25, facecolor=variant_clrs['Point'], alpha=0.41,
        clip_on=False, transform=diag_ax1.transData
        ))
    diag_ax1.text(0.4, 0.95,
                  "{}\nMutant\n({} samples)".format(
                      mut_str, np.sum(use_df.cStat)),
                  size=10, ha='center', va='center')

    diag_ax1.add_patch(ptchs.Circle(
        (0.4, 0.22), radius=0.42, facecolor=variant_clrs['WT'], alpha=0.41,
        clip_on=False, transform=diag_ax1.transData
        ))
    diag_ax1.text(0.4, 0.22,
                  "{}\nWild-Type\n({} samples)".format(
                      mut_str, np.sum(~use_df.cStat)),
                  size=13, ha='center', va='center')

    diag_ax1.text(0.02, 0.67, "classify\nmutations", color='red',
                  size=11, fontstyle='italic', ha='right', va='center')
    diag_ax1.axhline(y=0.67, xmin=0.03, xmax=0.83, color='red',
                     linestyle='--', linewidth=2.1, alpha=0.81)

    diag_ax1.text(0.82, 0.68, "{} (+)".format(np.sum(use_df.cStat)),
                  color='red', size=8, fontstyle='italic', 
                  ha='right', va='bottom')
    diag_ax1.text(0.82, 0.65, "{} (\u2212)".format(np.sum(~use_df.cStat)),
                  color='red', size=8, fontstyle='italic',
                  ha='right', va='top')

    sns.violinplot(data=use_df[~use_df.cStat], y='Value', ax=clf_ax1,
                   palette=[variant_clrs['WT']], linewidth=0, cut=0)
    sns.violinplot(data=use_df[use_df.cStat], y='Value', ax=clf_ax1,
                   palette=[variant_clrs['Point']], linewidth=0, cut=0)

    clf_ax1.text(0.5, 0.99,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value, use_df.cStat)),
                 color='red', size=11, fontstyle='italic',
                 ha='center', va='top', transform=clf_ax1.transAxes)

    diag_ax2.add_patch(ptchs.Wedge((0.38, 0.95), 0.25, 90, 270,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, clip_on=False,
                                   transform=diag_ax2.transData))

    diag_ax2.add_patch(ptchs.Wedge((0.42, 0.95), 0.25, 270, 90,
                                   facecolor=variant_clrs['Point'],
                                   alpha=0.41, clip_on=False,
                                   transform=diag_ax2.transData))
    diag_ax2.add_patch(ptchs.Wedge((0.42, 0.95), 0.25, 270, 90,
                                   facecolor='None', edgecolor='black',
                                   clip_on=False, linewidth=0.8,
                                   transform=diag_ax2.transData))

    diag_ax2.text(0.02, 0.67, "same classifier\nresults", color='red',
                  size=8, fontstyle='italic', ha='right', va='center')
    diag_ax2.axhline(y=0.67, xmin=0.03, xmax=0.83, color='red',
                     linestyle='--', linewidth=0.8, alpha=0.67)

    diag_ax2.add_patch(ptchs.Wedge((0.38, 0.22), 0.42, 90, 270,
                                   facecolor=variant_clrs['WT'],
                                   alpha=0.41, clip_on=False,
                                   transform=diag_ax2.transData))

    diag_ax2.add_patch(ptchs.Wedge((0.42, 0.22), 0.42, 270, 90,
                                   facecolor=variant_clrs['WT'],
                                   alpha=0.41, clip_on=False,
                                   transform=diag_ax2.transData))
    diag_ax2.add_patch(ptchs.Wedge((0.42, 0.22), 0.42, 270, 90,
                                   facecolor='None', edgecolor='black',
                                   clip_on=False, linewidth=0.8,
                                   transform=diag_ax2.transData))

    diag_ax2.text(0.36, 0.95,
                  "{}\nMutant\nw/o overlap\n({} samps)".format(
                      mut_str, np.sum(use_df.cStat & ~use_df.rStat)),
                  size=8, ha='right', va='center')
    diag_ax2.text(0.44, 0.95,
                  "{}\nMutant\nw/ overlap\n({} samps)".format(
                      mut_str, np.sum(use_df.cStat & use_df.rStat)),
                  size=8, ha='left', va='center')

    diag_ax2.text(0.36, 0.22,
                  "{}\nWild-Type\nw/o overlap\n({} samps)".format(
                      mut_str, np.sum(~use_df.cStat & ~use_df.rStat)),
                  size=12, ha='right', va='center')
    diag_ax2.text(0.44, 0.22,
                  "{}\nWild-Type\nw/ overlap\n({} samps)".format(
                      mut_str, np.sum(~use_df.cStat & use_df.rStat)),
                  size=12, ha='left', va='center')

    sns.violinplot(data=use_df[~use_df.cStat], x='cStat', y='Value',
                   hue='rStat', palette=[variant_clrs['WT']],
                   hue_order=[False, True], split=True, linewidth=0,
                   cut=0, ax=clf_ax2)
    sns.violinplot(data=use_df[use_df.cStat], x='cStat', y='Value',
                   hue='rStat', palette=[variant_clrs['Point']],
                   hue_order=[False, True], split=True, linewidth=0,
                   cut=0, ax=clf_ax2)

    vals_min, vals_max = use_df.Value.quantile(q=[0, 1])
    vals_rng = (vals_max - vals_min) / 51

    clf_ax2.get_legend().remove()
    diag_ax2.axvline(x=0.4, ymin=-0.22, ymax=1.22, clip_on=False,
                     color=variant_clrs['Point'], linewidth=1.1, alpha=0.81,
                     linestyle=':')

    diag_ax2.text(0.4, -0.25,
                  "partition scored samples according to\noverlap with "
                  "PIK3CA mutations\nthat are not {}".format(mut_str),
                  color=variant_clrs['Point'], size=10,
                  fontstyle='italic', ha='center', va='top')

    for clf_ax in clf_ax1, clf_ax2:
        clf_ax.set_xticks([])
        clf_ax.set_xticklabels([])
        clf_ax.set_yticklabels([])

        clf_ax.xaxis.label.set_visible(False)
        clf_ax.yaxis.label.set_visible(False)
        clf_ax.set_ylim(vals_min - vals_rng, vals_max + 3 * vals_rng)

    clf_ax1.get_children()[0].set_alpha(0.41)
    clf_ax1.get_children()[2].set_alpha(0.41)
    clf_ax2.get_children()[0].set_alpha(0.41)
    clf_ax2.get_children()[3].set_alpha(0.41)

    for i in [1, 4]:
        clr_face = clf_ax2.get_children()[i].get_facecolor()[0]
        clr_face[-1] = 0.41
        clf_ax2.get_children()[i].set_linewidth(0.7)
        clf_ax2.get_children()[i].set_facecolor(clr_face)

    clf_ax2.text(0.23, 0.96, "{} w/o overlap".format(mut_str),
                 color=variant_clrs['Point'], size=9,
                 fontstyle='italic', ha='center', va='bottom',
                 transform=clf_ax2.transAxes)
    clf_ax2.text(0.23, 0.95,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value[~use_df.rStat],
                                               use_df.cStat[~use_df.rStat])),
                 color='red', size=11, fontstyle='italic',
                 ha='center', va='top', transform=clf_ax2.transAxes)

    clf_ax2.text(0.77, 0.96, "{} w/ overlap".format(mut_str),
                 color=variant_clrs['Point'], size=9,
                 fontstyle='italic', ha='center', va='bottom',
                 transform=clf_ax2.transAxes)
    clf_ax2.text(0.77, 0.95,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value[use_df.rStat],
                                               use_df.cStat[use_df.rStat])),
                 color='red', size=11, fontstyle='italic',
                 ha='center', va='top', transform=clf_ax2.transAxes)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
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
                    edgecolor='black', clip_on=False, linewidth=0.7,
                    transform=diag_ax.transData
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
                    edgecolor='black', clip_on=False, linewidth=0.7,
                    transform=diag_ax.transData
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

    base_diag_ax.text(-0.1, 0.67, "1) classify\nmutations", color='red',
                      size=8, fontstyle='italic', ha='right', va='center')
    base_diag_ax.axhline(y=0.67, xmin=-0.09, xmax=0.12, color='red',
                         alpha=0.83, clip_on=False,
                         linestyle='--', linewidth=1.5)

    base_diag_ax.text(0.46, 1.18, "2) apply trained\nclassifier", color='red',
                      size=8, fontstyle='italic', ha='center', va='bottom')
    base_diag_ax.axhline(y=0.67, xmin=0.16, xmax=0.89, color='red',
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
                                       edgecolor='black', clip_on=False,
                                       alpha=0.19, linewidth=1.3,
                                       transform=base_diag_ax.transData))

    base_diag_ax.text(0.2, 0.67,
                      "{}\nMutant\nw/ overlap\n({} samps)".format(
                          mut_str, np.sum(vals_df.cStat & all_stat)),
                      size=5, alpha=0.41, ha='left', va='center')

    base_diag_ax.add_patch(ptchs.Wedge((0.46, 0.67), 0.35, 270, 90,
                                       facecolor=variant_clrs['WT'],
                                       edgecolor='black', clip_on=False, 
                                       alpha=0.19, linewidth=1.3,
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

    base_vio_ax.get_children()[0].set_alpha(0.41)
    base_vio_ax.get_children()[3].set_alpha(0.41)
    base_vio_ax.get_children()[1].set_alpha(0.29)
    base_vio_ax.get_children()[4].set_alpha(0.29)
    base_vio_ax.get_children()[1].set_linewidth(1.1)
    base_vio_ax.get_children()[4].set_linewidth(1.1)

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

        diag_ax.text(-0.1, 0.67, "1) classify\nmutations", color='red',
                     size=8, fontstyle='italic', ha='right', va='center')
        diag_ax.axhline(y=0.67, xmin=-0.09, xmax=0.12, color='red',
                        alpha=0.83, clip_on=False,
                        linestyle='--', linewidth=1.5)

        diag_ax.text(0.46, 1.18, "2) apply trained\nclassifier", color='red',
                     size=8, fontstyle='italic', ha='center', va='bottom')
        diag_ax.axhline(y=0.67, xmin=0.16, xmax=0.89, color='red',
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
                                      facecolor=variant_clrs[lbl],
                                      edgecolor='black', clip_on=False,
                                      alpha=0.21, linewidth=1.3,
                                      transform=diag_ax.transData))

        diag_ax.text(0.2, 0.67,
                     "{}\nMutant\nw/ {}\n({} samps)".format(
                         mut_str, lbl, np.sum(vals_df.cStat & vals_df.mStat)),
                     size=5, alpha=0.41, ha='left', va='center')

        diag_ax.add_patch(ptchs.Wedge((0.46, 0.67), 0.35, 270, 90,
                                      facecolor=variant_clrs['WT'],
                                      edgecolor='black', clip_on=False, 
                                      alpha=0.21, linewidth=1.3,
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
                       palette=[variant_clrs[lbl]], hue_order=[False, True],
                       split=True, linewidth=0, cut=0, ax=vio_ax)

        vio_ax.get_children()[0].set_alpha(0.41)
        vio_ax.get_children()[2].set_alpha(0.41)
        vio_ax.get_children()[1].set_alpha(0.29)
        vio_ax.get_children()[4].set_alpha(0.53)
        vio_ax.get_children()[2].set_linewidth(1.3)
        vio_ax.get_children()[4].set_linewidth(1.3)

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


def main():
    parser = argparse.ArgumentParser(
        "Plot an example diagram showing how overlap with other types of "
        "mutations can affect a mutation classification task."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('--samp_cutoff', default=20)

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)

    cdata = load_cohort_data(base_dir, args.cohort, args.gene, 'Protein')
    all_mtype = MuType(cdata.train_mut.allkey())
    all_stat = np.array(cdata.train_pheno(all_mtype))

    out_path = Path(base_dir, 'output', args.cohort, args.gene)
    out_dirs = [
        pth.parent for pth in out_path.glob(
            '*/samps_{}/Protein/out__task-0.p'.format(args.samp_cutoff))
        ]

    use_clfs = [out_dir.parent.parent.stem for out_dir in out_dirs
                if (len(tuple(out_dir.glob('out__task-*.p')))
                    == len(tuple(out_dir.glob('slurm/fit-*.txt*'))))]

    infer_dicts = {
        clf: load_infer_output(os.path.join(
            base_dir, 'output', args.cohort, args.gene, clf,
            'samps_{}'.format(args.samp_cutoff), 'Protein'
            ))
        for clf in use_clfs
        }

    assert all(set(infer_dict['Iso'].index) == set(infer_dict['All'].index)
               for infer_dict in infer_dicts.values())
    assert len(set(frozenset(infer_dict['Iso'].index)
                   for infer_dict in infer_dicts.values())) == 1

    infer_dicts = {clf: {smps: vals.applymap(np.mean)
                         for smps, vals in infer_dict.items()}
                   for clf, infer_dict in infer_dicts.items()}

    base_mtypes = sorted(
        [{'All': mtype, 'Ex': ExMcomb(cdata.train_mut, mtype)}
         for mtype in infer_dicts[use_clfs[0]]['All'].index
         if (isinstance(mtype, MuType) and 'Protein' in mtype.get_levels()
             and (ExMcomb(cdata.train_mut, mtype)
                  in infer_dicts[use_clfs[0]]['Iso'].index))],
        key=itemgetter('All')
        )

    mcomb_stats = {mcomb: np.array(cdata.train_pheno(mcomb))
                   for mtps in base_mtypes for mcomb in mtps.values()}

    mcomb_masks = {
        mtps['All']: {'All': {mtp_lbl: np.array([True]
                                                * len(cdata.train_samps))
                              for mtp_lbl in mtps},
                      'Iso': {mtp_lbl: ~(all_stat & ~mcomb_stats[mtp])
                              for mtp_lbl, mtp in mtps.items()}}
        for mtps in base_mtypes
        }

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


if __name__ == '__main__':
    main()

