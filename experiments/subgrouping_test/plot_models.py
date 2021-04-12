"""
This module produces plots of mutation classification models' feature
coefficients, that is, the weights each classifier assigned to individual
genes whose expression it was using to predict the presence of a mutation.

Example usages:
    python -m dryads-research.experiments.subgrouping_test.plot_models \
        microarray METABRIC_LumA Ridge
    python -m dryads-research.experiments.subgrouping_test.plot_models \
        microarray METABRIC_LumA Ridge -a 0.75
    python -m dryads-research.experiments.subgrouping_test.plot_models \
        Firehose LUSC Ridge --plot_all
    python -m dryads-research.experiments.subgrouping_test.plot_models \
        toil__gns beatAML Ridge -g FLT3 TP53 IDH1 MEGF10

"""

from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.misc import get_label, get_subtype
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from functools import reduce
from operator import or_, and_

from scipy.stats import spearmanr
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colorbar import ColorbarBase
from matplotlib import colors

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'models')


def calculate_jaccard(*phns):
    return reduce(and_, phns).sum() / reduce(or_, phns).sum()


def plot_all_heatmap(coef_mat, args):
    fig, ax = plt.subplots(figsize=(23, coef_mat.shape[0] / 5))

    plot_df = coef_mat.loc[:, coef_mat.abs().sum() > 0]
    plt_max = np.abs(np.percentile(plot_df.values.flatten(),
                                   q=[1, 99])).max()

    plot_df = plot_df.iloc[
        dendrogram(linkage(distance.pdist(plot_df, metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves'],
        dendrogram(linkage(distance.pdist(plot_df.transpose(),
                                          metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves']
        ]

    ylabs = [get_fancy_label(get_subtype(mtype))
             for mtype in plot_df.index]
    coef_cmap = sns.diverging_palette(13, 131, s=91, l=41, sep=3,
                                      as_cmap=True)

    sns.heatmap(plot_df, cmap=coef_cmap, vmin=-plt_max, vmax=plt_max,
                xticklabels=False, yticklabels=ylabs)
    ax.set_yticklabels(ylabs, size=8, ha='right', rotation=0)

    use_gene = {get_label(mtype) for mtype in coef_mat.index}
    use_gene = tuple(use_gene)[0]

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_all-heatmap_{}.svg".format(use_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_top_heatmap(coef_mat, auc_vals, pheno_dict, args):
    use_gene = {get_label(mtype) for mtype in auc_vals.index}
    assert len(use_gene) == 1
    use_gene = tuple(use_gene)[0]

    plot_df = (coef_mat.transpose() / coef_mat.abs().max(axis=1)).transpose()

    if args.auc_cutoff == -1:
        min_auc = auc_vals[MuType({('Gene', use_gene): pnt_mtype})]
    else:
        min_auc = args.auc_cutoff

    plt_mtypes = {mtype for mtype, auc_val in auc_vals.iteritems()
                  if (not isinstance(mtype, RandomType)
                      and auc_val >= min_auc
                      and (get_subtype(mtype) & copy_mtype).is_empty())}

    plt_genes = set()
    for mtype in plt_mtypes:
        plt_genes |= set(plot_df.loc[mtype].abs().sort_values()[-10:].index)

    fig, ax = plt.subplots(figsize=(4 + len(plt_genes) / 4,
                                    1.3 + len(plt_mtypes) / 5.3))

    plot_df = plot_df.loc[plt_mtypes, plt_genes]
    plot_df = plot_df.iloc[
        dendrogram(linkage(distance.pdist(plot_df, metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves'],
        dendrogram(linkage(distance.pdist(plot_df.transpose(),
                                          metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves']
        ]

    coef_cmap = sns.diverging_palette(13, 131, s=91, l=41, sep=3,
                                      as_cmap=True)

    sns.heatmap(plot_df, cmap=coef_cmap, center=0,
                xticklabels=False, yticklabels=False)

    for i, mtype in enumerate(plot_df.index):
        if mtype == MuType({('Gene', use_gene): pnt_mtype}):
            lbl_wgt = 'bold'
        else:
            lbl_wgt = 'normal'

        ax.text(-0.29 / plot_df.shape[1], 1 - ((i + 0.53) / plot_df.shape[0]),
                get_fancy_label(get_subtype(mtype)),
                size=9, weight=lbl_wgt, ha='right', va='center',
                transform=ax.transAxes)

    for i, gene in enumerate(plot_df.columns):
        ax.text((i + 1) / plot_df.shape[1], -0.29 / plot_df.shape[0], gene,
                size=12, ha='right', va='top', rotation=47,
                transform=ax.transAxes, clip_on=False)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_top-heatmap_{}.svg".format(
                         use_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_auto_heatmap(coef_mat, auc_df, pheno_dict, args):
    plt_mtypes = {mtype for mtype, auc_val in auc_df['mean'].iteritems()
                  if (not isinstance(mtype, RandomType)
                      and (auc_val >= args.auc_cutoff
                           or get_subtype(mtype) == pnt_mtype)
                      and (get_subtype(mtype) & copy_mtype).is_empty())}

    clust_genes = set()
    for mtype in plt_mtypes:
        clust_genes |= set(
            coef_mat.loc[mtype].abs().sort_values()[-5:].index)

    plot_df = coef_mat.loc[plt_mtypes, clust_genes]
    mtype_linkg = linkage(distance.pdist(plot_df,
                                         lambda u, v: 1 - spearmanr(u, v)[0]))

    use_gene = {get_label(mtype) for mtype in auc_df.index}
    assert len(use_gene) == 1
    use_gene = tuple(use_gene)[0]
    base_mtype = MuType({('Gene', use_gene): pnt_mtype})

    mtype_clust = fcluster(mtype_linkg, 4, criterion='maxclust')
    clust_df = pd.DataFrame({'Clust': mtype_clust,
                             'AUC': auc_df['mean'][plot_df.index]},
                            index=plt_mtypes)

    clust_dict = dict()
    for clust, all_df in clust_df.groupby(by='Clust'):
        clust_mtypes = list(all_df.sort_values(by='AUC').index)
        clust_dict[clust] = [clust_mtypes.pop()]

        while clust_mtypes:
            cur_mtype = clust_mtypes.pop()

            if (cur_mtype != base_mtype
                    and not any(
                        calculate_jaccard(pheno_dict[cur_mtype],
                                          pheno_dict[old_mtype]) >= 0.9
                        for old_mtype in clust_dict[clust]
                        )):
                clust_dict[clust] += [cur_mtype]

    use_mtypes = reduce(or_, [set(mtypes) for mtypes in clust_dict.values()])
    use_mtypes |= {base_mtype}
    top_mtypes = {mtypes[0] for mtypes in clust_dict.values()}
    mtype_dendr = dendrogram(mtype_linkg, no_plot=True)

    plot_df = (plot_df.transpose() / plot_df.abs().max(axis=1)).transpose()
    plt_genes = set()
    for mtype in use_mtypes:
        plt_genes |= set(plot_df.loc[mtype].abs().sort_values()[-5:].index)

    plot_df = plot_df.loc[:, plt_genes]
    plot_df = plot_df.iloc[
        mtype_dendr['leaves'],
        dendrogram(linkage(distance.pdist(plot_df.transpose(),
                                          metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves']
        ]

    plot_df = plot_df.loc[[mtype in use_mtypes for mtype in plot_df.index]]
    fig_wdth = 13 + len(plt_genes) / 5.3

    fig, (clust_ax, heat_ax, lgnd_ax) = plt.subplots(
        figsize=(fig_wdth, 1.3 + len(use_mtypes) / 5.3), nrows=1, ncols=3,
        gridspec_kw=dict(width_ratios=[1, fig_wdth - 2, 1])
        )

    clust_indx = {i: list() for i in clust_df.Clust.unique()}
    for i, mtype in enumerate(plot_df.index[:-1]):
        new_clust = clust_df.loc[plot_df.index[i + 1], 'Clust']

        if i == 0:
            clust_indx[clust_df.loc[mtype, 'Clust']] += [-1]

        if clust_df.loc[mtype, 'Clust'] != new_clust:
            clust_indx[clust_df.loc[mtype, 'Clust']] += [i]
            clust_indx[new_clust] += [i]

            heat_ax.plot([0, plot_df.shape[1]], [i + 1, i + 1],
                         color='black', linewidth=0.61)

    coef_cmap = sns.diverging_palette(13, 131, s=91, l=41, sep=3,
                                      as_cmap=True)

    sns.heatmap(plot_df, cmap=coef_cmap, center=0, ax=heat_ax, cbar=False,
                xticklabels=False, yticklabels=False)

    max_lbl = 1
    for i, mtype in enumerate(plot_df.index):
        if mtype in {base_mtype} | top_mtypes:
            lbl_wgt = 'bold'
            lbl_gap = ''
        else:
            lbl_wgt = 'normal'
            lbl_gap = ' '

        mtype_lbl = lbl_gap.join([
            get_fancy_label(get_subtype(mtype)),
            "({})".format(sum(pheno_dict[mtype])).rjust(7)
            ])
        max_lbl = max(max_lbl, len(mtype_lbl))

        heat_ax.text(-0.29 / plot_df.shape[1],
                     1 - ((i + 0.53) / plot_df.shape[0]),
                     mtype_lbl, size=11, weight=lbl_wgt,
                     ha='right', va='center', transform=heat_ax.transAxes)

        auc_lbl = ' {:.2f}'.format(auc_df.loc[mtype, 'mean'])
        if (np.array(auc_df['CV'][mtype])
                > np.array(auc_df['CV'][base_mtype])).all():
            auc_lbl += '*'

        heat_ax.text(1, 1 - ((i + 0.53) / plot_df.shape[0]), auc_lbl,
                     size=12, style='italic', ha='left', va='center',
                     transform=heat_ax.transAxes)

    heat_ax.text(
        -0.29 / plot_df.shape[1], 1.007,
        "subgroupings of {}       \nin {}       ".format(
            use_gene, get_cohort_label(args.cohort)),
        size=16, weight='bold', ha='right', va='bottom',
        transform=heat_ax.transAxes
        )

    heat_ax.text(-0.1 / plot_df.shape[1], 1.007, 'samp\ncount',
                 size=13, weight='bold',
                 ha='right', va='bottom', transform=heat_ax.transAxes)
    heat_ax.text(1, 1.007, 'AUC', size=17, weight='bold',
                 ha='left', va='bottom', transform=heat_ax.transAxes)

    clust_ax.set_ylim(heat_ax.get_ylim())
    for clst_i, clst_edgs in clust_indx.items():
        if len(clst_edgs) == 2:
            clst_st, clst_en = clst_edgs

        else:
            clst_st = clst_edgs[0]
            clst_en = plot_df.shape[0] - 1

        clust_sz = np.sum(clust_df.Clust == clst_i)
        if clust_sz == 1:
            clust_lbl = "(1 subgp)"
        else:
            clust_lbl = "({} subgps)".format(clust_sz)

        clust_ax.text(max_lbl / -11, (clst_st + clst_en) / 2 + 1, clust_lbl,
                      ha='right', va='center', size=15, style='italic')

    for i, gene in enumerate(plot_df.columns):
        heat_ax.text(i + 0.67, plot_df.shape[0] + 0.23, gene, size=12,
                     ha='right', va='top', rotation=47, clip_on=False)

    clr_ax = lgnd_ax.inset_axes(bounds=(0, 0.17, 1, 0.66),
                                clip_on=False)
    clr_bar = ColorbarBase(ax=clr_ax, cmap=coef_cmap,
                           norm=colors.Normalize(vmin=-1, vmax=1),
                           ticks=[-1, -0.5, 0., 0.5, 1])

    clr_bar.ax.set_title("Mean\nCoefs", weight='bold', size=19)
    clr_bar.ax.set_yticklabels(['-1', '-0.5', '0', '+0.5', '+1'],
                               size=1.7 * sum(plot_df.shape) ** 0.53,
                               fontweight='bold')

    lgnd_ax.axis('off')
    clust_ax.axis('off')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_auto-heatmap_{}.svg".format(use_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_divergence_scatter(coef_vals, args):
    assert get_label(coef_vals.index[0]) == get_label(coef_vals.index[1])
    fig, ax = plt.subplots(figsize=(10, 9))

    ax.scatter(coef_vals.iloc[0, :], coef_vals.iloc[1, :],
               facecolor='0.41', s=7, alpha=0.19, edgecolors='none')

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    ax.plot(x_lims, [0, 0],
            color='black', linewidth=1.6, linestyle=':', alpha=0.53)
    ax.plot([0, 0], y_lims,
            color='black', linewidth=1.6, linestyle=':', alpha=0.53)

    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.grid(alpha=0.37, linewidth=0.9)

    mtype_lbls = [get_fancy_label(get_subtype(mtype),
                                  pnt_link='\nor ', phrase_link=' ')
                  for mtype in coef_vals.index]

    top_genes = set(coef_vals.iloc[0, :].abs().sort_values()[-10:].index)
    top_genes |= set(coef_vals.iloc[1, :].abs().sort_values()[-10:].index)
    top_genes |= set(coef_vals.iloc[0, :].sort_values()[-10:].index)
    top_genes |= set(coef_vals.iloc[1, :].sort_values()[-10:].index)
    top_genes |= set(coef_vals.iloc[0, :].sort_values()[:10].index)
    top_genes |= set(coef_vals.iloc[1, :].sort_values()[:10].index)

    top_genes |= set((coef_vals.iloc[0, :]
                      - coef_vals.iloc[1, :]).abs().sort_values()[-25:].index)

    if args.genes:
        txt_genes = set(args.genes)
    else:
        txt_genes = set()

    plot_dict = dict()
    font_dict = dict()

    for gene, (coef_val1, coef_val2) in coef_vals.iteritems():
        if gene in top_genes or gene in txt_genes:
            plot_dict[coef_val1, coef_val2] = [175034 ** -1, (gene, '')]
        if gene in txt_genes:
            font_dict[coef_val1, coef_val2] = dict(weight='bold')

    lbl_pos = place_scatter_labels(plot_dict, ax,
                                   font_size=11, font_dict=font_dict,
                                   c='black', linewidth=0.47, alpha=0.43)

    use_gene = get_label(coef_vals.index[0])
    ax.text(0.99, 0.03,
            "{} in\n{}".format(use_gene, get_cohort_label(args.cohort)),
            size=22, style='italic', ha='right', va='bottom',
            transform=ax.transAxes)

    plt.xlabel('\n'.join(["Coefficients for:", mtype_lbls[0]]),
               fontsize=21, weight='semibold')
    plt.ylabel('\n'.join(["Coefficients for:", mtype_lbls[1]]),
               fontsize=21, weight='semibold')

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.savefig(
        os.path.join(
            plot_dir, '__'.join([args.expr_source, args.cohort]),
            "{}__{}__divergence-scatter_{}.svg".format(
                use_gene, get_subtype(coef_vals.index[1]).get_filelabel(),
                args.classif
                )
            ),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--auc_cutoff', '-a', type=float,
                        help="min AUC for tasks shown in heatmaps",
                        nargs='?', default=0.7, const=-1)

    parser.add_argument('--plot_all', action='store_true',
                        help="create plot using all genes? (time-costly)")
    parser.add_argument('--genes', '-g', nargs='+',
                        help="genes to create special annotations for")

    # parse command line arguments, create directory where plots will be saved
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
        raise ValueError("Cannot compare coefficients until this experiment "
                         "is run with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    cdata = None
    phn_dict = dict()
    auc_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "cohort-data__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            new_cdata = pickle.load(f)

        if cdata is None:
            cdata = new_cdata
        else:
            cdata.merge(new_cdata)

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

    auc_df = pd.concat(auc_dict.values())
    coef_mat = pd.read_csv(os.path.join(
        base_dir, "summaries", args.classif,
        "{}__{}__coef-means_Base.csv".format(args.expr_source, args.cohort)
        ), index_col=0)

    use_aucs = auc_df[[not isinstance(mtype, RandomType)
                       and (get_subtype(mtype) & copy_mtype).is_empty()
                       for mtype in auc_df.index]]

    for gene, auc_vals in use_aucs.groupby(get_label):
        if (auc_vals['mean'] >= args.auc_cutoff).sum() >= 5:
            use_coefs = coef_mat.iloc[
                coef_mat.index.str.contains(gene),
                [(cdata.gene_annot[expr_gene]['Chr']
                  != cdata.gene_annot[gene]['Chr'])
                 for expr_gene in coef_mat.columns]
                ]
            use_coefs.index = sorted(auc_vals.index)

            if args.plot_all:
                plot_all_heatmap(use_coefs, args)

            plot_top_heatmap(use_coefs, auc_vals['mean'], phn_dict, args)
            plot_auto_heatmap(use_coefs, auc_vals, phn_dict, args)

            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_coefs = use_coefs.loc[base_mtype]

            for mtype, coefs in use_coefs.iterrows():
                if (np.array(auc_df['CV'][mtype])
                        > np.array(auc_df['CV'][base_mtype])).all():
                    coef_vals = pd.concat(
                        [base_coefs, coefs], axis=1).transpose()
                    plot_divergence_scatter(coef_vals, args)


if __name__ == '__main__':
    main()

