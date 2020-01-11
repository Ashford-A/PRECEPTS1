
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'cohort')

from HetMan.experiments.subvariant_test import (
    type_file, metabric_dir, pnt_mtype, copy_mtype)
from HetMan.experiments.subvariant_test.merge_test import merge_cohort_data
from HetMan.experiments.subvariant_test.plot_aucs import choose_gene_colour
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.utilities.transformers import OmicUMAP4
from HetMan.features.cohorts.metabric import (
    load_metabric_samps, choose_subtypes)

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from functools import reduce
from operator import and_

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_umap_clustering(trans_expr, type_data, cdata, args):
    fig, axarr = plt.subplots(figsize=(14, 13), nrows=4, ncols=4)

    trans_expr = trans_expr[:, :4]
    type_stat = np.array([type_data.SUBTYPE[type_data.index.get_loc(samp)]
                          if samp in type_data.index else 'Not Available'
                          for samp in cdata.train_data(None)[0].index])

    type_clrs = sns.color_palette('bright', n_colors=len(set(type_stat)))
    lgnd_lbls = []
    lgnd_marks = []

    for sub_type, type_clr in zip(sorted(set(type_stat)), type_clrs):
        type_indx = type_stat == sub_type

        for i, j in combn(range(4), 2):
            axarr[i, j].plot(
                trans_expr[type_indx, i], trans_expr[type_indx, j],
                marker='o', linewidth=0, markersize=5, alpha=0.23,
                mfc=type_clr, mec='none'
                )

        lgnd_lbls += ["{} ({})".format(sub_type, np.sum(type_indx))]
        lgnd_marks += [Line2D([], [], marker='o', linestyle='None',
                              markersize=19, alpha=0.43,
                              markerfacecolor=type_clr,
                              markeredgecolor='none')]

    for i in range(4):
        axarr[i, i].axis('off')
        axarr[i, i].text(0.5, 0.5, "UMAP Component {}".format(i + 1),
                         size=17, weight='semibold', ha='center', va='center')

    for i, j in combn(range(4), 2):
        axarr[i, j].set_xticklabels([])
        axarr[i, j].set_yticklabels([])
        axarr[j, i].set_xticklabels([])
        axarr[j, i].set_yticklabels([])

    plt.legend(lgnd_marks, lgnd_lbls,
               bbox_to_anchor=(0.5, 1 / 29), bbox_transform=fig.transFigure,
               frameon=False, fontsize=21, ncol=3, loc=9, handletextpad=0.3)

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__UMAP-clustering.svg".format(args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_mutation_counts(auc_dict, pheno_dict, cdata, args):
    fig, ax = plt.subplots(figsize=(9, 2))

    auc_vals = pd.concat(auc_dict.values())
    auc_vals = auc_vals[[(not isinstance(mtype, RandomType)
                          and (mtype.subtype_list()[0][1] != pnt_mtype)
                          and (mtype.subtype_list()[0][1]
                               & copy_mtype).is_empty())
                         for mtype in auc_vals.index]]

    mtype_df = pd.DataFrame({
        'AUC': auc_vals, 'Levels': 'Other',
        'Gene': [mtype.get_labels()[0] for mtype in auc_vals.index]
        })

    ax.axis('off')
    for lvls, auc_list in auc_dict.items():
        mtype_df.loc[mtype_df.index.isin(auc_list.index), 'Levels'] = lvls

    gene_counts = mtype_df.Gene.value_counts()
    mtype_df.loc[~mtype_df.Gene.isin(gene_counts.index[:10]),
                 'Gene'] = 'Other'

    mtype_df.loc[[mtype.subtype_list()[0][1] == pnt_mtype
                  for mtype in mtype_df.index], 'Levels'] = 'Point'
    count_tbl = pd.crosstab(mtype_df.Levels, mtype_df.Gene, margins=True)

    for i, gene in enumerate(gene_counts.index[:10]):
        for j, lvls in enumerate(auc_dict):
            ax.text(0.41 + (i / 15), 0.95 - (j / 7),
                    count_tbl.loc[lvls, gene], size=10, ha='center', va='top',
                    transform=ax.transAxes)

        ax.text(0.41 + (i / 15), 0.95 - (4 / 7), count_tbl.loc['All', gene],
                size=11, ha='center', va='top', transform=ax.transAxes)

    ax.text(0.41 + (10 / 15), 0.95 - (4 / 7), count_tbl.loc['All', 'Other'],
            size=11, ha='center', va='top', transform=ax.transAxes)
    ax.text(0.41 + (11 / 15), 0.95 - (4 / 7), count_tbl.loc['All', 'All'],
            size=12, ha='center', va='top', transform=ax.transAxes)

    for i, gene in enumerate(gene_counts.index[:10]):
        ax.text(0.4 + (i / 15), 0.99, gene, size=13, rotation=45,
                ha='left', va='bottom', transform=ax.transAxes)

    ax.text(0.4 + (10 / 15), 0.99, 'Other', size=13, rotation=45,
            ha='left', va='bottom', transform=ax.transAxes)
    ax.text(0.4 + (11 / 15), 0.99, 'All', size=13, rotation=45,
            ha='left', va='bottom', transform=ax.transAxes)

    ax.text(0, 0.95 - (4 / 7), 'Total',
            size=13, ha='left', va='top', transform=ax.transAxes)
    for j, lvls in enumerate(auc_dict):
        ax.text(0, 0.95 - (j / 7), lvls.replace('__', '->'),
                size=13, ha='left', va='top', transform=ax.transAxes)

        ax.text(0.41 + (10 / 15), 0.95 - (j / 7),
                count_tbl.loc[lvls, 'Other'], size=10,
                ha='center', va='top', transform=ax.transAxes)
        ax.text(0.41 + (11 / 15), 0.95 - (j / 7), count_tbl.loc[lvls, 'All'],
                size=12, ha='center', va='top', transform=ax.transAxes)

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__mutation-counts.svg".format(args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_classif_performance(auc_dfs, time_dfs, cdata, args):
    use_mtypes = reduce(and_, [auc_df.index[[not isinstance(mtype, RandomType)
                                             for mtype in auc_df.index]]
                               for auc_df in auc_dfs.values()])

    plt_times = sorted(
        [(clf, (time_df.loc[use_mtypes, 'avg'].apply(np.mean).values
                + time_df.loc[use_mtypes, 'std'].apply(
                    np.mean).values).mean())
          for clf, time_df in time_dfs.items()],
        key=lambda x: x[1]
        )

    auc_df = pd.concat(
        [pd.DataFrame({'AUC': auc_vals[list(set(use_mtypes))],
                       'Classif': clf})
         for clf, auc_vals in auc_dfs.items()]
        )

    fig, ax = plt.subplots(figsize=(1 + 2 * len(plt_times), 9))
    sns.violinplot(x='Classif', y='AUC', data=auc_df, ax=ax, width=0.93,
                   palette=[choose_gene_colour(clf, clr_sat=0.67)
                            for clf, _ in plt_times],
                   order=[clf for clf, _ in plt_times], cut=0)

    for i in range(len(plt_times)):
        ax.get_children()[i * 2].set_alpha(0.71)

    ax.set_xticklabels(["{}\n({:.1f}s)".format(clf, clf_time)
                        for clf, clf_time in plt_times], size=25)
    ax.tick_params(axis='y', labelsize=23)

    ax.axhline(y=0.5, c='black', linewidth=2.7, linestyle=':', alpha=0.89)
    ax.axhline(y=1, c='black', linewidth=2.1, alpha=0.89)
    ax.set_xlabel('')
    ax.set_ylabel('AUC', size=29, weight='semibold')

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__classif_performance.svg".format(
                                 args.cohort)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the results of the subvariant enumeration experiment "
        "across all classifiers tested on a given cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)
    np.random.seed(9087)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/trnsf-vals__*__*.p.gz".format(
                args.expr_source, args.cohort)
            )
        ]
 
    out_list = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'trnsf-vals__')[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split('.p.gz')[0]}
        for out_data in out_datas
        ])
 
    if out_list.shape[0] == 0:
        raise ValueError("No {} cohorts have been loaded yet "
                         "for expression source {} !".format(
                             args.cohort, args.expr_source))

    out_use = out_list.groupby('Classif').filter(
        lambda outs: ('Exon__Location__Protein' in set(outs.Levels))).groupby(
            ['Levels', 'Classif'])['Samps'].min()

    out_dir = os.path.join(base_dir, "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, out_use.min()))
    cdata = merge_cohort_data(out_dir, use_seed=8713)

    if args.expr_source == 'Firehose' or 'toil_' in args.expr_source:
        type_data = pd.read_csv(type_file, sep='\t', index_col=0, comment='#')

        if '_' in cdata.cohort:
            use_cohort = cdata.cohort.split('_')[0]
        else:
            use_cohort = cdata.cohort

        if use_cohort in type_data.DISEASE.values:
            type_data = type_data[type_data.DISEASE == use_cohort]
        else:
            type_data = pd.DataFrame({'SUBTYPE': 'Not Available'},
                                     index=cdata.get_samples())

    elif args.expr_source == 'microarray':
        type_data = pd.DataFrame({'SUBTYPE': 'Other'},
                                 index=cdata.get_samples())

        samp_data = load_metabric_samps(metabric_dir)
        samp_data = samp_data.loc[samp_data.index.isin(cdata.get_samples())]

        for tp in ['Basal', 'Her2', 'LumA', 'LumB']:
            type_data.SUBTYPE[samp_data.index.isin(
                choose_subtypes(samp_data, tp))] = tp

    trans_expr = OmicUMAP4().fit_transform_coh(cdata)
    plot_umap_clustering(trans_expr.copy(), type_data, cdata, args)

    use_clfs = set(out_use.index.get_level_values('Classif'))
    phn_dict = dict()
    time_dicts = {clf: dict() for clf in use_clfs}
    auc_dicts = {clf: dict() for clf in use_clfs}

    for (lvls, clf), ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(
                base_dir, out_tag, "out-pheno__{}__{}.p.gz".format(lvls, clf)
                ), 'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(
                base_dir, out_tag, "out-tune__{}__{}.p.gz".format(lvls, clf)
                ), 'r') as f:
            time_dicts[clf][lvls] = pickle.load(f)[1]

        with bz2.BZ2File(os.path.join(
                base_dir, out_tag, "out-aucs__{}__{}.p.gz".format(lvls, clf)
                ), 'r') as f:
            auc_dicts[clf][lvls] = pickle.load(f)['mean']

    time_dfs = {clf: pd.concat(time_dict.values())
                for clf, time_dict in time_dicts.items()}
    auc_dfs = {clf: pd.concat(auc_dict.values())
               for clf, auc_dict in auc_dicts.items()}

    plot_mutation_counts(tuple(auc_dicts.values())[0], phn_dict, cdata, args)
    plot_classif_performance(auc_dfs, time_dfs, cdata, args)


if __name__ == "__main__":
    main()

