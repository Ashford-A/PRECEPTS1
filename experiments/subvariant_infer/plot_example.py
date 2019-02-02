
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
from HetMan.experiments.subvariant_infer import variant_clrs, MuType

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
from matplotlib.collections import PolyCollection


def plot_base_classification(mtype, use_df, cdata, args):
    fig, ((diag_ax1, clf_ax1), (diag_ax2, clf_ax2)) = plt.subplots(
        figsize=(7, 9), nrows=2, ncols=2)

    all_mtype = MuType(cdata.train_mut.allkey())
    mtype_str = ":".join([args.gene, str(mtype).split(':')[-1][2:]])

    use_df = pd.DataFrame({'Value': use_df.loc[mtype],
                           'cStat': np.array(cdata.train_pheno(mtype)),
                           'rStat': np.array(cdata.train_pheno(all_mtype
                                                               - mtype))})

    for diag_ax in diag_ax1, diag_ax2:
        diag_ax.axis('off')
        diag_ax.set_aspect('equal')

        diag_ax.add_patch(ptchs.FancyArrow(
            0.92, 0.54, dx=0.17, dy=0, width=0.04, length_includes_head=True,
            head_length=0.08, clip_on=False, alpha=0.77, linewidth=1.9,
            facecolor='white', edgecolor='black'
            ))

    diag_ax1.add_patch(ptchs.Circle(
        (0.4, 0.95), radius=0.25, facecolor=variant_clrs['Point'], alpha=0.41,
        clip_on=False, transform=diag_ax1.transData
        ))
    diag_ax1.text(0.4, 0.95,
                  "TCGA-{}\n{}\nMutant\n({} samples)".format(
                      args.cohort, mtype_str, np.sum(use_df.cStat)),
                  size=11, ha='center', va='center')

    diag_ax1.add_patch(ptchs.Circle(
        (0.4, 0.22), radius=0.42, facecolor=variant_clrs['WT'], alpha=0.41,
        clip_on=False, transform=diag_ax1.transData
        ))
    diag_ax1.text(0.4, 0.22,
                  "TCGA-{}\n{}\nWild-Type\n({} samples)".format(
                      args.cohort, mtype_str, np.sum(~use_df.cStat)),
                  size=14, ha='center', va='center')

    diag_ax1.text(0.01, 0.67, "classify\nmutations", color='red',
                  size=11, fontstyle='italic', ha='right', va='center')
    diag_ax1.axhline(y=0.67, xmin=0.03, xmax=0.79, color='red',
                     linestyle='--', linewidth=2.3, alpha=0.81)

    diag_ax1.text(0.78, 0.685, "{} (+)".format(np.sum(use_df.cStat)),
                  color='red', size=9, fontstyle='italic', 
                  ha='right', va='bottom')
    diag_ax1.text(0.78, 0.655, "{} (\u2212)".format(np.sum(~use_df.cStat)),
                  color='red', size=9, fontstyle='italic',
                  ha='right', va='top')

    sns.violinplot(data=use_df[~use_df.cStat], y='Value', ax=clf_ax1,
                   palette=[variant_clrs['WT']], linewidth=0, cut=0)
    sns.violinplot(data=use_df[use_df.cStat], y='Value', ax=clf_ax1,
                   palette=[variant_clrs['Point']], linewidth=0, cut=0)

    clf_ax1.text(0.5, 0.98,
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

    diag_ax1.add_patch(ptchs.Wedge((0.38, 0.22), 0.42, 90, 270,
                                    facecolor=variant_clrs['WT'],
                                    alpha=0.41, clip_on=False,
                                    transform=diag_ax2.transData))
    diag_ax1.add_patch(ptchs.Wedge((0.42, 0.22), 0.42, 270, 90,
                                    facecolor=variant_clrs['WT'],
                                    alpha=0.41, clip_on=False,
                                    transform=diag_ax2.transData))

    diag_ax2.text(0.36, 0.95,
                  "{}\nMutant\nw/o overlap\n({} samps)".format(
                      mtype_str.split(':')[-1],
                      np.sum(use_df.cStat & ~use_df.rStat)
                    ),
                  size=8, ha='right', va='center')
    diag_ax2.text(0.44, 0.95,
                  "{}\nMutant\nw/ overlap\n({} samps)".format(
                      mtype_str.split(':')[-1],
                      np.sum(use_df.cStat & use_df.rStat)
                    ),
                  size=8, ha='left', va='center')

    diag_ax2.text(0.36, 0.22,
                  "{}\nWild-Type\nw/o overlap\n({} samps)".format(
                      mtype_str.split(':')[-1],
                      np.sum(~use_df.cStat & ~use_df.rStat)
                    ),
                  size=12, ha='right', va='center')
    diag_ax2.text(0.44, 0.22,
                  "{}\nWild-Type\nw/ overlap\n({} samps)".format(
                      mtype_str.split(':')[-1],
                      np.sum(~use_df.cStat & use_df.rStat)
                    ),
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
    vals_rng = (vals_max - vals_min) / 27

    clf_ax2.get_legend().remove()
    diag_ax2.axvline(x=0.4, ymin=-0.22, ymax=1.22, clip_on=False,
                     linestyle=':', color='green', linewidth=1.7, alpha=0.81)
    clf_ax2.axvline(x=0, ymin=-1, ymax=2, linestyle=':',
                    color='black', linewidth=1.3, alpha=0.61)

    diag_ax2.text(0.05, 0.67, "(same classifier)", color='red',
                  size=10, fontstyle='italic', ha='right', va='center')
    diag_ax2.text(0.4, -0.24,
                  "partition scored samples according to overlap\nwith "
                  "PIK3CA mutations that are not {}".format(
                      mtype_str.split(':')[-1]),
                  color='green', size=11, fontstyle='italic',
                  ha='center', va='top')

    for clf_ax in clf_ax1, clf_ax2:
        clf_ax.set_xticks([])
        clf_ax.set_xticklabels([])
        clf_ax.set_yticklabels([])

        clf_ax.xaxis.label.set_visible(False)
        clf_ax.yaxis.label.set_visible(False)
        clf_ax.set_ylim(vals_min - vals_rng, vals_max + 2 * vals_rng)

        for art in clf_ax.get_children():
            if isinstance(art, PolyCollection):
                art.set_alpha(0.41)

    clf_ax2.text(0.23, 0.96,
                 "{} w/o overlap".format(mtype_str.split(':')[-1]),
                 color='green', size=9, fontstyle='italic',
                 ha='center', va='bottom', transform=clf_ax2.transAxes)
    clf_ax2.text(0.23, 0.95,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value[~use_df.rStat],
                                               use_df.cStat[~use_df.rStat])),
                 color='red', size=11, fontstyle='italic',
                 ha='center', va='top', transform=clf_ax2.transAxes)

    clf_ax2.text(0.77, 0.96,
                 "{} w/ overlap".format(mtype_str.split(':')[-1]),
                 color='green', size=9, fontstyle='italic',
                 ha='center', va='bottom', transform=clf_ax2.transAxes)
    clf_ax2.text(0.77, 0.95,
                 "AUC: {:.3f}".format(calc_auc(use_df.Value[use_df.rStat],
                                               use_df.cStat[use_df.rStat])),
                 color='red', size=11, fontstyle='italic',
                 ha='center', va='top', transform=clf_ax2.transAxes)

    plt.tight_layout(w_pad=3.1, h_pad=2.3)
    plt.savefig(os.path.join(
        plot_dir, args.cohort, "base_classification_{}_samps-{}.svg".format(
            args.gene, args.samp_cutoff)
        ),
        dpi=300, bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot an example diagram showing how overlap with other types of
        mutations can affect a mutation classification task."
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


if __name__ == '__main__':
    main()

