
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from .utils import filter_mtype
from ..utilities.colour_maps import variant_clrs, form_clrs
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.labels import get_fancy_label, get_cohort_label

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'projection')


def plot_projection_scores(use_mtypes, pred_df, auc_vals,
                           pheno_dict, cdata, use_type, args):
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    type_lvls = 'Gene', 'Scale', use_type
    cdata.add_mut_lvls(type_lvls)
    type_tree = cdata.mtrees[type_lvls][args.gene]['Point']

    type_mtypes = {
        type_lbl: MuType({('Gene', args.gene): {('Scale', 'Point'): {
            (use_type, type_lbl): None}}})
        for type_lbl, _ in type_tree
        }

    plt_mtypes = [base_mtype] + list(use_mtypes)
    mtype_lbls = [get_fancy_label(get_subtype(plt_mtype),
                                  pnt_link='\nor ', phrase_link=' ')
                  for plt_mtype in plt_mtypes]
    lbl_wdth = max(len(s) for lbl in mtype_lbls for s in lbl.split('\n'))

    fig, axarr = plt.subplots(
        figsize=(0.59 + lbl_wdth / 9.7 + (len(type_mtypes) + 1) * 0.47,
                 1.9 + len(plt_mtypes) * 2.1),
        nrows=len(plt_mtypes), ncols=1
        )

    pred_vals = pd.DataFrame({mtype: pred_df.loc[mtype].apply(np.mean)
                              for mtype in plt_mtypes})
    type_clrs = {'Wild-Type': variant_clrs['WT']}
    type_phns = {'Wild-Type': ~pheno_dict[base_mtype]}

    for type_lbl, type_mtype in type_mtypes.items():
        type_phns[type_lbl] = cdata.train_pheno(type_mtype)

        if use_type == 'Form_base':
            type_clrs[type_lbl] = form_clrs[type_lbl]
        else:
            type_clrs[type_lbl] = choose_label_colour(
                type_lbl, clr_lum=0.47, clr_sat=0.95)

    type_ordr = sorted(
        type_phns.items(),
        key=lambda tp: pred_vals[plt_mtypes].loc[tp[1]].quantile(0.5).mean()
        )

    for i, (ax, plt_mtype) in enumerate(zip(axarr, plt_mtypes)):
        plt_phn = pheno_dict[plt_mtype]

        plt_df = pd.concat([
            pd.DataFrame({'Type': type_lbl,
                          'Vals': pred_vals[plt_mtype][type_phn]})
            for type_lbl, type_phn in type_ordr
            ])

        plt_lims = plt_df.Vals.quantile([0, 1]).tolist()
        plt_rng = plt_lims[1] - plt_lims[0]
        vio_ordr = [type_lbl if np.sum(type_phn) > 10 else None
                    for type_lbl, type_phn in type_ordr]

        sns.violinplot(x=plt_df.Type, y=plt_df.Vals, order=vio_ordr,
                       palette=[type_clrs[type_lbl]
                                for type_lbl, _ in type_ordr],
                       ax=ax, inner=None, orient='v',
                       linewidth=0, cut=0, width=0.97)

        vio_list = [(type_lbl, type_phns[type_lbl]) for type_lbl in vio_ordr
                    if type_lbl is not None]
        for j in range(len(vio_list)):
            ax.get_children()[j].set_alpha(0.53)

        sns.violinplot(x=plt_df.Type, y=plt_df.Vals,
                       order=vio_ordr, ax=ax, inner=None, orient='v',
                       linewidth=1.1, cut=0, width=0.97)

        for j, (_, type_phn) in enumerate(vio_list):
            if np.sum(type_phn & plt_phn) < np.sum(type_phn):
                ax.get_children()[j + len(vio_list)].set_visible(False)
            else:
                ax.get_children()[j + len(vio_list)].set_facecolor('none')

        for j, (type_lbl, type_phn) in enumerate(type_ordr):
            plt_stat = type_phn & plt_phn

            if np.sum(type_phn) > 10:
                if np.sum(plt_stat) < np.sum(type_phn):
                    ax.scatter(j + np.random.randn(np.sum(plt_stat)) / 8.3,
                               pred_vals[plt_mtype][plt_stat],
                               facecolor=type_clrs[type_lbl], s=8, alpha=0.37,
                               edgecolors='black', linewidth=0.4)

            else:
                tp_stat = type_phn & ~plt_phn

                if plt_stat.any():
                    ax.scatter(j + np.random.randn(plt_stat.sum()) / 8.3,
                               pred_vals[plt_mtype][plt_stat],
                               facecolor=type_clrs[type_lbl], s=31,
                               alpha=0.53, edgecolors='black', linewidth=0.9)

                if tp_stat.any():
                    ax.scatter(j + np.random.randn(tp_stat.sum()) / 8.3,
                               pred_vals[plt_mtype][tp_stat],
                               facecolor=type_clrs[type_lbl], s=31,
                               alpha=0.53, edgecolors='none')

        for j, (type_lbl, type_phn) in enumerate(type_ordr):
            if i == 0:
                ax.text(j - 1 / 13, plt_lims[1],
                        "n={}".format(np.sum(type_phn)),
                        size=13, ha='left', va='bottom', rotation=45)

            elif i == (len(plt_mtypes) - 1):
                ax.text(j + 1 / 13, plt_lims[0] - plt_rng * 0.03,
                        type_lbl.replace('_', '\n'),
                        size=11, ha='right', va='top', rotation=53)

                if j == 0:
                    ax.text(0.5, -0.61, "{} Status".format(args.gene),
                            size=21, ha='center', va='top', weight='semibold',
                            transform=ax.transAxes)

        ax.text(1.03, 0.61, mtype_lbls[i], size=10, ha='left', va='bottom',
                transform=ax.transAxes)
        ax.text(1.03, 0.48, "n={}".format(np.sum(pheno_dict[plt_mtype])),
                size=12, ha='left', va='top', transform=ax.transAxes)
        ax.text(1.03, 0.32, "AUC: {:.3f}".format(auc_vals[plt_mtype]),
                size=12, ha='left', va='top', transform=ax.transAxes)

        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.set_xlim(np.array(ax.get_xlim()) + [-0.07, 0.07])
        ax.set_ylim([plt_lims[0] - plt_rng / 23, plt_lims[1] + plt_rng / 23])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.text(0.02, 0.5, "Mutation Classifier Score", size=21,
             ha='right', va='center', rotation=90, weight='semibold')

    plt.tight_layout(h_pad=1.7)
    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]), args.gene,
                     "{}__proj-scores_{}.svg".format(use_type, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_orthogonal_scores(pred_vals, auc_vals, pheno_dict, cdata, args):
    fig, ax = plt.subplots(figsize=(9.17, 10))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    plt_mtype1, plt_mtype2 = pred_vals.index.tolist()

    mtype_lbl1 = get_fancy_label(get_subtype(plt_mtype1),
                                 pnt_link='\nor ', phrase_link=' ')
    mtype_lbl2 = get_fancy_label(get_subtype(plt_mtype2),
                                 pnt_link='\nor ', phrase_link=' ')

    plot_df = pd.DataFrame({'Sub1': pred_vals.iloc[0].apply(np.mean),
                            'Sub2': pred_vals.iloc[1].apply(np.mean)})

    ax.plot(plot_df.Sub1[~pheno_dict[base_mtype]],
            plot_df.Sub2[~pheno_dict[base_mtype]],
            marker='o', markersize=6, linewidth=0, alpha=0.17,
            mfc=variant_clrs['WT'], mec='none')

    ax.plot(plot_df.Sub1[pheno_dict[plt_mtype1] & ~pheno_dict[plt_mtype2]],
            plot_df.Sub2[pheno_dict[plt_mtype1] & ~pheno_dict[plt_mtype2]],
            marker='o', markersize=9, linewidth=0, alpha=0.29,
            mfc='#D97400', mec='none')

    ax.plot(plot_df.Sub1[pheno_dict[plt_mtype2] & ~pheno_dict[plt_mtype1]],
            plot_df.Sub2[pheno_dict[plt_mtype2] & ~pheno_dict[plt_mtype1]],
            marker='o', markersize=9, linewidth=0, alpha=0.29,
            mfc=variant_clrs['Point'], mec='none')

    rest_stat = pheno_dict[base_mtype] & ~pheno_dict[plt_mtype1]
    rest_stat &= ~pheno_dict[plt_mtype2]
    ax.plot(plot_df.Sub1[rest_stat], plot_df.Sub2[rest_stat],
            marker='o', markersize=11, linewidth=0, alpha=0.31,
            mfc='none', mec='black', mew=3.1)

    both_stat = pheno_dict[plt_mtype1] & pheno_dict[plt_mtype2]
    if both_stat.any():
        ax.plot(plot_df.Sub1[both_stat], plot_df.Sub2[both_stat],
                marker='o', markersize=11, linewidth=0, alpha=0.31,
                mfc=variant_clrs['Point'], mec='#D97400', mew=3.1)

    ax.text(0.98, 0.03, "{}\nmutants".format(mtype_lbl1),
            size=14, c='#D97400', weight='bold',
            ha='right', va='bottom', transform=ax.transAxes)
    ax.text(0.03, 0.98, "{}\nmutants".format(mtype_lbl2),
            size=14, c=variant_clrs['Point'], weight='bold',
            ha='left', va='top', transform=ax.transAxes)

    ax.text(0.5, 1.037, get_cohort_label(args.cohort), size=31,
            style='italic', ha='center', va='bottom',
            transform=ax.transAxes)

    if args.legends:
        lgnd_lbls = [
            "wild-type for {}".format(args.gene),
            "point mutant of {}\nin neither subgrouping".format(args.gene),
            "mutant for both\nsubgroupings"
            ]

        lgnd_mrks = [
            Line2D([], [], marker='o', linestyle='None',
                   markersize=19, alpha=0.43,
                   markerfacecolor=variant_clrs['WT'],
                   markeredgecolor='none'),
            Line2D([], [], marker='o', linestyle='None',
                   markersize=27, alpha=0.43,
                   markerfacecolor='none', markeredgecolor='black', mew=3.1),
            Line2D([], [], marker='o', linestyle='None',
                   markersize=27, alpha=0.43,
                   markerfacecolor=variant_clrs['Point'],
                   markeredgecolor='#D97400', mew=3.1)
            ]

        ax.legend(lgnd_mrks, lgnd_lbls, fontsize=17,
                  bbox_to_anchor=(1.07, 0.85), ncol=1, loc=1,
                  frameon=False, handletextpad=0.3)

    xlims = ax.get_xlim()
    xgap = (xlims[1] - xlims[0]) / 7
    ax.set_xlim([xlims[0], xlims[1] + xgap])
    ylims = ax.get_ylim()
    ygap = (ylims[1] - ylims[0]) / 7
    ax.set_ylim([ylims[0], ylims[1] + ygap])

    ax.grid(linewidth=0.83, alpha=0.41)
    score_lbl = "Subgrouping Task\nMean Predicted Label"
    ax.set_xlabel(score_lbl, size=25, weight='semibold')
    ax.set_ylabel(score_lbl, size=25, weight='semibold')

    plt.savefig(
        os.path.join(
            plot_dir, '__'.join([args.expr_source, args.cohort]), args.gene,
            "ortho-scores__{}__{}__{}.svg".format(
                get_subtype(plt_mtype1).get_filelabel(),
                get_subtype(plt_mtype2).get_filelabel(), args.classif
                )
            ),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the relationships between the scores inferred by a classifier "
        "for the subgroupings enumerated for a particular gene in a cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('gene', help="a mutated gene", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.7,
                        help="min AUC for tasks shown in plots")
    parser.add_argument('--types', '-t', nargs='*', default=tuple(),
                        help='a list of mutated genes', type=str)
    parser.add_argument('--legends', action='store_true',
                        help="add plot legends where applicable?")

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
            cdata.merge(new_cdata, use_genes=[args.gene])

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_data = pickle.load(f)

        pred_dict[lvls] = pred_data.loc[[
            mtype for mtype in pred_data.index
            if (not isinstance(mtype, RandomType)
                and filter_mtype(mtype, args.gene))
            ]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_data = pickle.load(f)

        phn_dict.update({mtype: phn for mtype, phn in phn_data.items()
                         if filter_mtype(mtype, args.gene)})

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_data = pickle.load(f)['mean']

        auc_dict[lvls] = auc_data[[filter_mtype(mtype, args.gene)
                                   for mtype in auc_data.index]]

    pred_df = pd.concat(pred_dict.values())
    auc_vals = pd.concat(auc_dict.values())

    if pred_df.shape[0] == 0:
        raise ValueError(
            "No classification tasks found for gene `{}`!".format(args.gene))

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort]),
                             args.gene),
                exist_ok=True)

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    wt_phn = ~phn_dict[base_mtype]
    wt_preds = pred_df.loc[base_mtype][wt_phn].apply(np.mean)

    use_mtypes = {mtype for mtype, auc_val in auc_vals.iteritems()
                  if (not isinstance(mtype, RandomType)
                      and auc_val >= args.auc_cutoff and mtype != base_mtype
                      and (get_subtype(mtype) & copy_mtype).is_empty())}

    if use_mtypes:
        corr_dict = {
            mtype: spearmanr(pred_df.loc[mtype][wt_phn].apply(np.mean),
                             wt_preds)[0]
            for mtype in use_mtypes
            }

        divg_list = pd.Series({mtype: (auc_vals[mtype] - 0.5) * (1 - corr_val)
                               for mtype, corr_val in corr_dict.items()})

        for plt_type in args.types:
            plot_projection_scores(divg_list.sort_values().index[-3:],
                                   pred_df, auc_vals, phn_dict,
                                   cdata, plt_type, args)

        wt_phns = {mtype: pred_df.loc[mtype][wt_phn].apply(np.mean)
                   for mtype in use_mtypes}

        ortho_vals = pd.Series({
            (mtype2, mtype1): spearmanr(wt_phns[mtype1],
                                        wt_phns[mtype2]).correlation
            for mtype1, mtype2 in combn(use_mtypes, 2)
            }).sort_values()

        for mtype1, mtype2 in ortho_vals.index[:100]:
            plot_orthogonal_scores(pred_df.loc[[mtype1, mtype2]],
                                   auc_vals, phn_dict, cdata, args)


if __name__ == '__main__':
    main()

