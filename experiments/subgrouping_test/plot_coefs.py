
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, dup_mtype, loss_mtype, RandomType)
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.colour_maps import variant_clrs

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from itertools import combinations as combn

import matplotlib as mpl
import matplotlib.pyplot as plt

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


def filter_mtype(mtype, gene):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is None:
            filter_stat = False
        else:
            filter_stat = tuple(mtype.base_mtype.label_iter())[0] == gene

    else:
        filter_stat = tuple(mtype.label_iter())[0] == gene

    return filter_stat


def plot_task_characteristics(coef_mat, auc_vals, pheno_dict, pred_df, args):
    fig, axarr = plt.subplots(figsize=(11, 18), nrows=3, ncols=1)

    coef_magns = coef_mat.abs().mean(axis=1)
    for mtype, coef_list in coef_mat.iterrows():
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


def plot_coef_divergence(coef_mat, auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    coef_means = coef_mat.groupby(level=0, axis=1).mean()
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


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('gene', help="a mutated gene")

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
    coef_mat = pd.concat(coef_dict.values())
    assert coef_mat.index.isin(phn_dict).all()
    auc_df = pd.concat(auc_dict.values())
    conf_list = pd.concat(conf_dict.values())

    plot_task_characteristics(coef_mat, auc_df['mean'],
                              phn_dict, pred_df, args)
    plot_coef_divergence(coef_mat, auc_df['mean'], phn_dict, args)


if __name__ == '__main__':
    main()

