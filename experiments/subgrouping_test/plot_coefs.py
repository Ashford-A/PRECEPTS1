
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, dup_mtype, loss_mtype, RandomType)
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from .utils import filter_mtype
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_fancy_label

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram

from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from ...features.data.genes_NCBI_9606_ProteinCoding import GENEID2NT
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'coefs')


def choose_subtype_colour(mtype):
    if isinstance(mtype, RandomType):
        use_clr = '0.53'

    elif (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty():
        use_clr = variant_clrs['Point']
    elif not (tuple(mtype.subtype_iter())[0][1] & dup_mtype).is_empty():
        use_clr = variant_clrs['Gain']
    elif not (tuple(mtype.subtype_iter())[0][1] & loss_mtype).is_empty():
        use_clr = variant_clrs['Loss']

    else:
        raise ValueError("Unrecognized mutation type `{}`!".format(mtype))

    return use_clr


def plot_task_characteristics(coef_df, auc_vals, pheno_dict, pred_df, args):
    fig, axarr = plt.subplots(figsize=(11, 18), nrows=3, ncols=1)

    coef_magns = coef_df.abs().mean(axis=1)
    for mtype, coef_list in coef_df.iterrows():
        use_clr = choose_subtype_colour(mtype)

        coef_grps = coef_list.groupby(level=0)
        mtype_coefs = np.array([coef_grps.nth(i) for i in range(40)])

        pcorr_val = np.mean([pearsonr(mtype_coefs[i], mtype_coefs[j])[0]
                             for i, j in combn(range(40), 2)])
        scorr_val = np.mean([spearmanr(mtype_coefs[i], mtype_coefs[j])[0]
                             for i, j in combn(range(40), 2)])

        for ax, val in zip(axarr, [pcorr_val, scorr_val, coef_magns[mtype]]):
            ax.scatter(auc_vals[mtype], val, facecolor=[use_clr],
                       s=751 * np.mean(pheno_dict[mtype]),
                       alpha=0.31, edgecolors='none')

    for ax in axarr:
        x_lims = ax.get_xlim()
        y_lims = [-ax.get_ylim()[1] / 91, ax.get_ylim()[1]]

        ax.plot(x_lims, [0, 0], color='black', linewidth=1.6, alpha=0.71)
        ax.plot([0.5, 0.5], [0, y_lims[1]],
                color='black', linewidth=1.4, linestyle=':', alpha=0.61)

        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.grid(alpha=0.37, linewidth=0.9)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)

    axarr[-1].set_xlabel("Mean AUC Across CVs", size=23, weight='semibold')
    for ax, ylbl in zip(axarr, ["Mean Pearson Corr\nBetween CVs",
                                "Mean Spearman Corr\nBetween CVs",
                                "Mean Signature\nMagnitude"]):
        ax.set_ylabel(ylbl, size=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_task-characteristics_{}.svg".format(
                         args.gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_coef_divergence(coef_df, auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    coef_means = coef_df.groupby(level=0, axis=1).mean()
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    for mtype, coef_vals in coef_means.iterrows():
        use_clr = choose_subtype_colour(mtype)
        corr_val = spearmanr(coef_means.loc[base_mtype], coef_vals)[0]

        ax.scatter(auc_vals[mtype], 1 - corr_val, facecolor=[use_clr],
                   s=751 * np.mean(pheno_dict[mtype]),
                   alpha=0.31, edgecolors='none')

    x_lims = ax.get_xlim()
    y_lims = [-ax.get_ylim()[1] / 91, ax.get_ylim()[1]]

    ax.plot(x_lims, [0, 0], color='black', linewidth=1.6, alpha=0.71)
    ax.plot([0.5, 0.5], [0, y_lims[1]],
            color='black', linewidth=1.4, linestyle=':', alpha=0.61)

    ax.tick_params(axis='both', which='major', labelsize=17)
    plt.grid(alpha=0.37, linewidth=0.9)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.xlabel("Task AUC", fontsize=23, weight='semibold')
    plt.ylabel("Signature Divergence\nfrom Gene-Wide Task",
               fontsize=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_coef-divergence_{}.svg".format(
                         args.gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_top_heatmap(coef_df, auc_vals, pheno_dict, args):
    coef_vals = coef_df.groupby(level=0, axis=1).mean()

    if args.auc_cutoff == -1:
        min_auc = auc_vals[MuType({('Gene', args.gene): pnt_mtype})]
    else:
        min_auc = args.auc_cutoff

    plt_mtypes = {mtype for mtype, auc_val in auc_vals.iteritems()
                  if (not isinstance(mtype, RandomType)
                      and auc_val >= min_auc
                      and (tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty())}
    plt_genes = set()

    for mtype in plt_mtypes:
        plt_genes |= set(coef_vals.loc[mtype].abs().sort_values()[-10:].index)

    fig, ax = plt.subplots(figsize=(4 + len(plt_genes) / 11,
                                    0.53 + len(plt_mtypes) / 5))

    plot_df = coef_vals.loc[plt_mtypes, plt_genes]
    plot_df = plot_df.iloc[
        dendrogram(linkage(distance.pdist(plot_df, metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves'],
        dendrogram(linkage(distance.pdist(plot_df.transpose(),
                                          metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves']
        ]

    xlabs = [gene for gene in plot_df.columns]
    ylabs = [get_fancy_label(tuple(mtype.subtype_iter())[0][1])
             for mtype in plot_df.index]

    coef_cmap = sns.diverging_palette(13, 131, s=91, l=41, sep=3,
                                      as_cmap=True)

    sns.heatmap(plot_df, cmap=coef_cmap, center=0,
                xticklabels=xlabs, yticklabels=ylabs)

    ax.set_xticklabels(xlabs, size=5, ha='right', rotation=47)
    ax.set_yticklabels(ylabs, size=9, ha='right', rotation=0)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_top-heatmap_{}.svg".format(
                         args.gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_all_heatmap(coef_df, auc_vals, pheno_dict, args):
    coef_vals = coef_df.groupby(level=0, axis=1).mean()

    plt_mtypes = {mtype for mtype in coef_vals.index
                  if (not isinstance(mtype, RandomType)
                      and (tuple(mtype.subtype_iter())[0][1]
                           & copy_mtype).is_empty())}

    fig, ax = plt.subplots(figsize=(23, len(plt_mtypes) / 5))
    plot_df = coef_vals.loc[plt_mtypes]

    plot_df = plot_df.loc[:, plot_df.abs().sum() > 0]
    plt_max = np.abs(np.percentile(plot_df.values.flatten(),
                                   q=[1, 99])).max()

    plot_df = plot_df.iloc[
        dendrogram(linkage(distance.pdist(plot_df, metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves'],
        dendrogram(linkage(distance.pdist(plot_df.transpose(),
                                          metric='euclidean'),
                           method='centroid'), no_plot=True)['leaves']
        ]

    ylabs = [get_fancy_label(tuple(mtype.subtype_iter())[0][1])
             for mtype in plot_df.index]
    coef_cmap = sns.diverging_palette(13, 131, s=91, l=41, sep=3,
                                      as_cmap=True)

    sns.heatmap(plot_df, cmap=coef_cmap, vmin=-plt_max, vmax=plt_max,
                xticklabels=False, yticklabels=ylabs)
    ax.set_yticklabels(ylabs, size=8, ha='right', rotation=0)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_all-heatmap_{}.svg".format(
                         args.gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_go_enrichment(coef_df, auc_vals, pheno_dict, args, mode='abs'):
    obo_fl = os.path.join(args.go_dir, "go-basic.obo")
    download_go_basic_obo(obo_fl)
    obodag = GODag(obo_fl)

    assoc_fl = os.path.join(args.go_dir, "gene2go")
    download_ncbi_associations(assoc_fl)
    objanno = Gene2GoReader(assoc_fl, taxids=[9606])
    ns2assoc = objanno.get_ns2assc()

    ncbi_map = {info.Symbol: ncbi_id for ncbi_id, info in GENEID2NT.items()}
    use_genes = set(coef_df.columns) & set(ncbi_map)
    bgrd_ids = [ncbi_map[gn] for gn in use_genes]

    goeaobj = GOEnrichmentStudyNS(bgrd_ids, ns2assoc, obodag,
                                  propagate_counts=False, alpha=0.05,
                                  methods=['fdr_bh'])

    plot_dict = dict()
    use_gos = set()
    coef_mat = coef_df.loc[:, [gene in use_genes for gene in coef_df.columns]]

    if mode == 'bayes':
        coef_means = coef_mat.groupby(level=0, axis=1).mean()
        coef_stds = coef_mat.groupby(level=0, axis=1).std()
    else:
        coef_mat = coef_mat.groupby(level=0, axis=1).mean()

    for mtype, coefs in coef_mat.iterrows():
        if not isinstance(mtype, RandomType):
            if mode == 'abs':
                fgrd_ctf = coefs.abs().quantile(0.95)
                fgrd_genes = coefs.index[coefs.abs() > fgrd_ctf]
                use_clr = 3.17

            elif mode == 'high':
                fgrd_ctf = coefs.quantile(0.95)
                fgrd_genes = coefs.index[coefs > fgrd_ctf]
                use_clr = 2.03
            elif mode == 'low':
                fgrd_ctf = coefs.quantile(0.05)
                fgrd_genes = coefs.index[coefs < fgrd_ctf]
                use_clr = 1.03

            elif mode == 'bayes':
                gene_scrs = coef_means.loc[mtype].abs() - coef_stds.loc[mtype]
                fgrd_genes = gene_scrs.index[gene_scrs > 0]
                use_clr = 3.17

            else:
                raise ValueError(
                    "Unrecognized `mode` argument <{}>!".format(mode))

            fgrd_ids = [ncbi_map[gn] for gn in fgrd_genes]
            goea_out = goeaobj.run_study(fgrd_ids, prt=None)

            plot_dict[mtype] = {
                rs.name: np.log10(rs.p_fdr_bh) for rs in goea_out
                if rs.enrichment == 'e' and rs.p_fdr_bh < 0.05
                }

    plot_df = pd.DataFrame(plot_dict, columns=plot_dict.keys())
    if plot_df.shape[0] == 0:
        print("Could not find any enriched GO terms across {} "
              "subgroupings!".format(plot_df.shape[1]))
        return None

    fig, ax = plt.subplots(figsize=(4.7 + plot_df.shape[0] / 2.3,
                                    2 + plot_df.shape[1] / 5.3))

    if plot_df.shape[0] > 2:
        plot_df = plot_df.iloc[
            dendrogram(linkage(distance.pdist(plot_df.fillna(0.0),
                                              metric='cityblock'),
                               method='centroid'), no_plot=True)['leaves']
            ].transpose()
    else:
        plot_df = plot_df.transpose()

    xlabs = [rs_nm for rs_nm in plot_df.columns]
    ylabs = [get_fancy_label(tuple(mtype.subtype_iter())[0][1])
             for mtype in plot_df.index]

    pval_cmap = sns.cubehelix_palette(start=use_clr, rot=0, dark=0, light=1,
                                      reverse=True, as_cmap=True)

    sns.heatmap(plot_df, cmap=pval_cmap, vmin=-5, vmax=0,
                linewidths=0.23, linecolor='0.73',
                xticklabels=xlabs, yticklabels=ylabs)

    ax.set_xticklabels(xlabs, size=15, ha='right', rotation=31)
    ax.set_yticklabels(ylabs, size=9, ha='right', rotation=0)
    ax.set_xlim((plot_df.shape[1] / -83, plot_df.shape[1] * 1.009))
    ax.set_ylim((plot_df.shape[0] * 1.009, plot_df.shape[0] / -83))

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_go-{}-enrichment_{}.svg".format(
                         args.gene, mode, args.classif)),
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
    parser.add_argument('gene', help="a mutated gene")

    parser.add_argument('--auc_cutoff', '-a', type=float,
                        help="min AUC for tasks shown in heatmaps",
                        nargs='?', default=0.7, const=-1)
    parser.add_argument('--go_dir', help="where are GO files to be stored?",
                        default=None)

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
    pred_dict = dict()
    phn_dict = dict()
    coef_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

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
            cdata.merge(new_cdata, use_genes=[args.gene])

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_data = pickle.load(f)

        pred_dict[lvls] = pred_data.loc[[mtype for mtype in pred_data.index
                                         if filter_mtype(mtype, args.gene)]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_data = pickle.load(f)

        phn_dict.update({mtype: phn for mtype, phn in phn_data.items()
                         if filter_mtype(mtype, args.gene)})

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-coef__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            coef_data = pickle.load(f)

        coef_data = coef_data.iloc[:, [(cdata.gene_annot[gene]['Chr']
                                        != cdata.gene_annot[args.gene]['Chr'])
                                       for gene in coef_data.columns]]

        coef_dict[lvls] = coef_data.loc[[mtype for mtype in coef_data.index
                                         if filter_mtype(mtype, args.gene)]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_data = pickle.load(f)

        auc_dict[lvls] = auc_data.loc[[mtype for mtype in auc_data.index
                                       if filter_mtype(mtype, args.gene)]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_data = pickle.load(f)

        conf_dict[lvls] = conf_data.loc[[mtype for mtype in conf_data.index
                                         if filter_mtype(mtype, args.gene)]]

    pred_df = pd.concat(pred_dict.values())
    coef_df = pd.concat(coef_dict.values())
    assert coef_df.index.isin(phn_dict).all()
    auc_df = pd.concat(auc_dict.values())
    conf_list = pd.concat(conf_dict.values())

    plot_task_characteristics(coef_df, auc_df['mean'],
                              phn_dict, pred_df, args)
    plot_coef_divergence(coef_df, auc_df['mean'], phn_dict, args)

    plot_top_heatmap(coef_df, auc_df['mean'], phn_dict, args)
    plot_all_heatmap(coef_df, auc_df['mean'], phn_dict, args)

    if args.go_dir:
        for use_mode in ['high', 'low', 'abs', 'bayes']:
            plot_go_enrichment(coef_df, auc_df['mean'],
                               phn_dict, args, use_mode)


if __name__ == '__main__':
    main()

