"""
This module produces plots comparing AUCs of classification tasks involving
actual subgroupings of mutation types against AUCs of tasks constructed using
randomly chosen sets of samples designed to establish a null background of
task performance.

Example usages:
    python -m dryads-research.experiments.subgrouping_test.plot_random \
        microarray METABRIC_LumA Ridge
    python -m dryads-research.experiments.subgrouping_test.plot_random \
        Firehose HNSC_HPV- Ridge
    python -m dryads-research.experiments.subgrouping_test.plot_random \
        Firehose LUSC Ridge

"""

from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_cohort_label

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

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'random')


def plot_cohort_comparison(auc_vals, pheno_dict, args):
    fig, (viol_ax, sctr_ax) = plt.subplots(
        figsize=(11, 7), nrows=1, ncols=2,
        gridspec_kw=dict(width_ratios=[1, 1.51])
        )

    mtype_genes = pd.Series([get_label(mtype) for mtype in auc_vals.index
                             if (not isinstance(mtype, RandomType)
                                 and (get_subtype(mtype)
                                      & copy_mtype).is_empty())])

    sbgp_genes = mtype_genes.value_counts()[
        mtype_genes.value_counts() > 1].index
    lbl_order = ['Random-Cohort',
                 'Point w/o Sub', 'Point w/ Sub', 'Subgroupings']

    gene_stat = pd.Series({
        mtype: ('Random-Cohort' if (isinstance(mtype, RandomType)
                                    and mtype.base_mtype is None)
                else 'RandomGene' if isinstance(mtype, RandomType)
                else 'Point w/ Sub'
                if (get_subtype(mtype) == pnt_mtype
                    and get_label(mtype) in sbgp_genes)
                else 'Point w/o Sub'
                if (get_subtype(mtype) == pnt_mtype
                    and not get_label(mtype) in sbgp_genes)
                else 'Copy' if not (get_subtype(mtype)
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
    viol_ax.get_children()[4].set_linewidth(1.7)
    viol_ax.get_children()[2].set_facecolor('white')
    viol_ax.get_children()[2].set_edgecolor(variant_clrs['Point'])

    viol_xlims = viol_ax.get_xlim()
    viol_ylims = viol_ax.get_ylim()

    if 'Point w/o Sub' in gene_stat:
        viol_ax.get_children()[4].set_edgecolor(variant_clrs['Point'])

    for i, lbl in enumerate(lbl_order):
        viol_ax.get_children()[i * 2].set_alpha(0.41)

        viol_ax.text(i - 0.13, viol_ax.get_ylim()[1],
                     "n={}".format((gene_stat == lbl).sum()),
                     size=15, rotation=37, ha='left', va='bottom')

    plt.text(0.91, -0.17, get_cohort_label(args.cohort), size=23,
             style='italic', ha='left', va='top',
             transform=viol_ax.transAxes)

    viol_ax.grid(axis='y', linewidth=0.83, alpha=0.41)
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

    sctr_ax.grid(linewidth=0.83, alpha=0.41)
    sctr_ax.plot(viol_ylims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    sctr_ax.plot([0.5, 0.5], viol_ylims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    sctr_ax.plot(viol_ylims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    sctr_ax.plot([1, 1], viol_ylims, color='black', linewidth=1.9, alpha=0.89)
    sctr_ax.plot(viol_ylims, viol_ylims,
            color='#550000', linewidth=1.7, linestyle='--', alpha=0.41)

    sctr_ax.set_xlabel("AUC of Oncogene Mutation", size=19, weight='semibold')
    sctr_ax.set_ylabel("Best AUC of Size-Matched\nGene-Specific Randoms",
                       size=19, weight='semibold')

    sctr_ax.set_xlim(viol_ylims)
    sctr_ax.set_ylim(viol_ylims)

    plt.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "cohort-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_gene_comparisons(auc_df, pheno_dict, args):
    base_aucs = auc_df['mean'][[mtype for mtype in auc_df.index
                                if (not isinstance(mtype, RandomType)
                                    and get_subtype(mtype) != pnt_mtype
                                    and (get_subtype(mtype)
                                         & copy_mtype).is_empty())]]

    rand_aucs = auc_df['mean'][[mtype for mtype in auc_df.index
                                if (isinstance(mtype, RandomType)
                                    and mtype.base_mtype is not None)]]

    base_grps = base_aucs[base_aucs >= 0.7].groupby(get_label)
    fig, axarr = plt.subplots(figsize=(0.5 + 1.5 * len(base_grps), 7),
                              nrows=1, ncols=len(base_grps))
    plt_min = auc_df['mean'].min()
 
    for i, (gene, auc_vec) in enumerate(sorted(base_grps,
                                               key=lambda x: x[1].max(),
                                               reverse=True)):
        base_mtype = MuType({('Gene', gene): pnt_mtype})
        axarr[i].set_title(gene, size=19, weight='semibold')

        plt_df = pd.concat([
            pd.DataFrame({
                'AUC': base_aucs[[mtype for mtype in base_aucs.index
                                  if get_label(mtype) == gene]],
                'Type': 'Orig'
                }),

            pd.DataFrame({
                'AUC': rand_aucs[[
                    mtype for mtype in rand_aucs.index
                    if (get_label(mtype.base_mtype) == gene
                        and mtype.size_dist < pheno_dict[base_mtype].sum())
                    ]],
                'Type': 'Rand'
                })
            ])

        plt_df['SigCV'] = [(np.array(auc_df['CV'][mtype])
                            > np.array(auc_df['CV'][base_mtype])).all()
                           for mtype in plt_df.index]

        if (plt_df.Type == 'Orig').sum() > 10:
            sns.violinplot(x=plt_df.Type, y=plt_df.AUC, ax=axarr[i],
                           order=['Orig', 'Rand'],
                           palette=[choose_label_colour(gene), '0.47'],
                           cut=0, linewidth=0, width=0.93)

        else:
            sctr_x = np.random.randn(plt_df.shape[0])

            for j in range(plt_df.shape[0]):
                if plt_df.Type.iloc[j] == 'Orig':
                    plt_clr = choose_label_colour(gene)
                    plt_x = sctr_x[j] / 7

                else:
                    plt_clr = '0.47'
                    plt_x = 1 + sctr_x[j] / 7

                axarr[i].scatter(plt_x, plt_df.AUC.iloc[j], s=37, alpha=0.29,
                                 facecolor=plt_clr, edgecolor='none')

        axarr[i].plot([-0.6, 1.6], [1, 1],
                      color='black', linewidth=1.7, alpha=0.79)
        axarr[i].plot([-0.6, 1.6], [0.5, 0.5],
                      color='black', linewidth=1.3, linestyle=':', alpha=0.61)

        axarr[i].plot([-0.6, 1.6], [auc_df['mean'][base_mtype]] * 2,
                      color=variant_clrs['Point'],
                      linewidth=2.3, linestyle='--', alpha=0.71)

        axarr[i].get_children()[0].set_alpha(0.53)
        axarr[i].get_children()[2].set_alpha(0.53)

        axarr[i].set_xlabel('')
        axarr[i].set_xticklabels([])
        axarr[i].grid(axis='x', linewidth=0)
        axarr[i].grid(axis='y', linewidth=0.5)

        axarr[i].text(0.37, 0, "n={}".format((plt_df.Type == 'Orig').sum()),
                      size=12, rotation=45, ha='right', va='center',
                      transform=axarr[i].transAxes)
        axarr[i].text(5 / 6, 0, "n={}".format((plt_df.Type == 'Rand').sum()),
                      size=12, rotation=45, ha='right', va='center',
                      transform=axarr[i].transAxes)

        best_aucs = plt_df.groupby('Type')['AUC'].max()
        best_types = plt_df.groupby('Type')['AUC'].idxmax()
        sig_stats = plt_df.groupby('Type')['SigCV'].any()

        for j, (type_lbl, best_auc) in enumerate(best_aucs.iteritems()):
            if sig_stats[type_lbl]:
                lbl_pos = min(0.963, best_auc - (1 - plt_min) / 83)

                axarr[i].text(j, lbl_pos, '*', size=19,
                              color=variant_clrs['Point'],
                              ha='center', va='bottom', weight='semibold')

        tour_stat = np.array([
            (np.array(auc_df['CV'][best_types['Orig']])
             < np.array(auc_df['CV'][mtype])).all()
            for mtype in plt_df[plt_df.Type == 'Rand'].index
            ]).any()

        if tour_stat:
            axarr[i].text(0.5, 0.977, '*',
                          size=27, ha='center', va='top', weight='semibold',
                          transform=axarr[i].transAxes)

        if i == 0:
            axarr[i].set_ylabel('AUC', size=21, weight='semibold')
        else:
            axarr[i].set_yticklabels([])
            axarr[i].set_ylabel('')

        axarr[i].set_xlim([-0.6, 1.6])
        axarr[i].set_ylim([plt_min - 0.07, 1.007])

    axarr[-1].text(0.9, 0.07, get_cohort_label(args.cohort),
                   size=20, style='italic', ha='right', va='bottom',
                   transform=axarr[-1].transAxes)

    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]),
                     "gene-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_random',
        description="Plots task AUCs versus null background task AUCs."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_list = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                              'Levels': '__'.join(out_data[1].split(
                                  'out-trnsf__')[1].split('__')[:-1])}
                             for out_data in out_datas])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby('Levels')['Samps'].min()
    if 'Consequence__Exon' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    phn_dict = dict()
    auc_df = pd.DataFrame()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_df = auc_df.append(pickle.load(f))

    # create the plots
    plot_cohort_comparison(auc_df['mean'], phn_dict, args)
    plot_gene_comparisons(auc_df, phn_dict, args)


if __name__ == '__main__':
    main()

