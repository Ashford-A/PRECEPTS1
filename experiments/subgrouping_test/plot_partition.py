
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'partition')

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test import pnt_mtype, copy_mtype
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_test.merge_test import merge_cohort_data
from HetMan.experiments.subvariant_test.plot_copy import select_mtype
from HetMan.experiments.subvariant_test.plot_aucs import place_labels
from HetMan.experiments.subvariant_test.utils import (
    get_fancy_label, choose_label_colour)

from HetMan.experiments.subvariant_infer.plot_ccle import load_response_data
from HetMan.experiments.subvariant_infer.setup_infer import compare_lvls
from HetMan.experiments.subvariant_test.plot_mutations import recurse_labels
from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.utilities.colour_maps import form_clrs

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_remainder_scores(plt_mtypes, pred_df, auc_vals,
                          pheno_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(0.5 + len(plt_mtypes) * 1.7, 7),
                              nrows=1, ncols=len(plt_mtypes))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    plot_df = pd.DataFrame({mtype: pred_df.loc[mtype].apply(np.mean)
                            for mtype in plt_mtypes})

    if len(plt_mtypes) == 1:
        axarr = [axarr]

    for i, (ax, plt_mtype) in enumerate(zip(axarr, plt_mtypes)):
        use_lvls = cdata.choose_mtree(plt_mtype)
        use_mtree = cdata.mtrees[use_lvls][args.gene]['Point']
        leaf_count = len(MuType(use_mtree.allkey()).subkeys())

        sns.violinplot(x=plot_df[plt_mtype][~pheno_dict[base_mtype]],
                       ax=ax, palette=[variant_clrs['WT']], inner=None,
                       orient='v', linewidth=0, cut=0, width=0.89)
        sns.violinplot(x=plot_df[plt_mtype][pheno_dict[plt_mtype]],
                       ax=ax, palette=[variant_clrs['Point']], inner=None,
                       orient='v', linewidth=0, cut=0, width=0.89)

        rest_stat = pheno_dict[base_mtype] & ~pheno_dict[plt_mtype]
        if rest_stat.sum() > 10:
            sns.violinplot(x=plot_df[plt_mtype][rest_stat],
                           ax=ax, palette=['none'], inner=None, orient='v',
                           linewidth=1.7, cut=0, width=0.89)

            ax.get_children()[0].set_alpha(0.41)
            ax.get_children()[1].set_alpha(0.41)
            ax.get_children()[2].set_facecolor((1, 1, 1, 0))
            ax.get_children()[2].set_edgecolor((0, 0, 0, 0.47))

        else:
            ax.scatter(np.random.randn(rest_stat.sum()) / 7.3,
                       plot_df[plt_mtype][rest_stat],
                       facecolor='none', s=31, alpha=0.53,
                       edgecolors='black', linewidth=0.9)

        tree_ax = inset_axes(ax, width='100%', height='100%',
                             bbox_to_anchor=(0.03, 0.89, 0.94, 0.09),
                             bbox_transform=ax.transAxes, borderpad=0)
        tree_ax.axis('off')
        tree_mtype = plt_mtype.subtype_list()[0][1].subtype_list()[0][1]

        tree_ax = recurse_labels(tree_ax, use_mtree, (0, leaf_count),
                                 len(use_lvls) - 2, leaf_count,
                                 clr_mtype=tree_mtype, add_lbls=False,
                                 mut_clr=variant_clrs['Point'])

        mtype_lbl = '\n'.join(get_fancy_label(plt_mtype).split('\n')[1:])
        ax.text(0.5, 1.01, mtype_lbl,
                size=9, ha='center', va='bottom', transform=ax.transAxes)

        ylims = ax.get_ylim()
        ygap = (ylims[1] - ylims[0]) / 7
        ax.set_ylim([ylims[0], ylims[1] + ygap])
        ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Subgrouping Inferred Score",
                          size=21, weight='semibold')
        else:
            ax.set_ylabel('')

    plt.tight_layout(w_pad=1.1)
    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]), args.gene,
                     "remainder-scores_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_projection_scores(plt_mtypes, pred_df, auc_vals,
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

    mtype_lbls = [
        '\n'.join(get_fancy_label(plt_mtype).split('\n')[1:]).replace(
            "any point ", "any point\n")
        for plt_mtype in [base_mtype] + plt_mtypes
        ]
    lbl_wdth = max(len(s) for lbl in mtype_lbls for s in lbl.split('\n'))

    fig, axarr = plt.subplots(
        figsize=(0.59 + lbl_wdth / 9.7 + (len(type_mtypes) + 1) * 0.47,
                 1.9 + (len(plt_mtypes) + 1) * 2.1),
        nrows=len(plt_mtypes) + 1, ncols=1
        )

    pred_vals = pd.DataFrame({mtype: pred_df.loc[mtype].apply(np.mean)
                              for mtype in [base_mtype] + plt_mtypes})
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
        key=lambda tp: pred_vals[[base_mtype] + plt_mtypes].loc[
            tp[1]].quantile(0.5).mean()
        )

    for i, (ax, plt_mtype) in enumerate(zip(axarr,
                                            [base_mtype] + plt_mtypes)):
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

            elif i == len(plt_mtypes):
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


def plot_orthogonal_scores(plt_mtype1, plt_mtype2, pred_df, auc_vals,
                           pheno_dict, cdata, args):
    fig, ax = plt.subplots(figsize=(11, 10))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    mtype_lbl1 = '\n'.join(get_fancy_label(plt_mtype1).split('\n')[1:])
    mtype_lbl2 = '\n'.join(get_fancy_label(plt_mtype2).split('\n')[1:])

    plot_df = pd.DataFrame({'Sub1': pred_df.loc[plt_mtype1].apply(np.mean),
                            'Sub2': pred_df.loc[plt_mtype2].apply(np.mean)})

    ax.plot(plot_df.Sub1[~pheno_dict[base_mtype]],
            plot_df.Sub2[~pheno_dict[base_mtype]],
            marker='o', markersize=6, linewidth=0, alpha=0.19,
            mfc=variant_clrs['WT'], mec='none')

    ax.plot(plot_df.Sub1[pheno_dict[plt_mtype1] & ~pheno_dict[plt_mtype2]],
            plot_df.Sub2[pheno_dict[plt_mtype1] & ~pheno_dict[plt_mtype2]],
            marker='o', markersize=9, linewidth=0, alpha=0.23,
            mfc='#D99D00', mec='none')

    ax.plot(plot_df.Sub1[pheno_dict[plt_mtype2] & ~pheno_dict[plt_mtype1]],
            plot_df.Sub2[pheno_dict[plt_mtype2] & ~pheno_dict[plt_mtype1]],
            marker='o', markersize=9, linewidth=0, alpha=0.23,
            mfc=variant_clrs['Point'], mec='none')

    rest_stat = pheno_dict[base_mtype] & ~pheno_dict[plt_mtype1]
    rest_stat &= ~pheno_dict[plt_mtype2]
    ax.plot(plot_df.Sub1[rest_stat], plot_df.Sub2[rest_stat],
            marker='o', markersize=11, linewidth=0, alpha=0.29,
            mfc='none', mec='black')

    both_stat = pheno_dict[plt_mtype1] & pheno_dict[plt_mtype2]
    if both_stat.any():
        ax.plot(plot_df.Sub1[both_stat], plot_df.Sub2[both_stat],
                marker='o', markersize=11, linewidth=0, alpha=0.29,
                mfc=variant_clrs['Point'], mec='#D99D00')

    tree_ax1 = inset_axes(ax, width='100%', height='100%',
                          bbox_to_anchor=(0.7, 0.79, 0.28, 0.09),
                          bbox_transform=ax.transAxes, borderpad=0)
    tree_ax2 = inset_axes(ax, width='100%', height='100%',
                          bbox_to_anchor=(0.56, 0.9, 0.28, 0.09),
                          bbox_transform=ax.transAxes, borderpad=0)

    for tree_ax, plt_mtype, mut_clr in zip(
            [tree_ax1, tree_ax2], [plt_mtype1, plt_mtype2],
            ['#D99D00', variant_clrs['Point']]
            ):
        tree_ax.axis('off')
        tree_mtype = plt_mtype.subtype_list()[0][1].subtype_list()[0][1]

        use_lvls = cdata.choose_mtree(plt_mtype)
        use_mtree = cdata.mtrees[use_lvls][args.gene]['Point']
        leaf_count = len(MuType(use_mtree.allkey()).subkeys())

        tree_ax = recurse_labels(tree_ax, use_mtree, (0, leaf_count),
                                 len(use_lvls) - 2, leaf_count,
                                 clr_mtype=tree_mtype, add_lbls=False,
                                 mut_clr=mut_clr)

    ax.text(0.98, 0.03, "{}\nmutants".format(mtype_lbl1),
            size=13, c='#D99D00', ha='right', va='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.98, "{}\nmutants".format(mtype_lbl2),
            size=13, c=variant_clrs['Point'], ha='left', va='top',
            transform=ax.transAxes)

    xlims = ax.get_xlim()
    xgap = (xlims[1] - xlims[0]) / 7
    ax.set_xlim([xlims[0], xlims[1] + xgap])
    ylims = ax.get_ylim()
    ygap = (ylims[1] - ylims[0]) / 7
    ax.set_ylim([ylims[0], ylims[1] + ygap])

    ax.set_xlabel("Subgrouping Inferred Score",
                  size=21, weight='semibold')
    ax.set_ylabel("Subgrouping Inferred Score",
                  size=21, weight='semibold')

    plt.savefig(
        os.path.join(
            plot_dir, '__'.join([args.expr_source, args.cohort]), args.gene,
            "ortho-scores__{}__{}__{}.svg".format(plt_mtype1.get_filelabel(),
                                                  plt_mtype2.get_filelabel(),
                                                  args.classif)
            ),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_orthogonal_response(plt_mtype1, plt_mtype2, auc_vals, ccle_df,
                             resp_df, cdata, args):
    fig, ax = plt.subplots(figsize=(11, 10))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    mtype_lbl1 = '\n'.join(get_fancy_label(plt_mtype1).split('\n')[1:])
    mtype_lbl2 = '\n'.join(get_fancy_label(plt_mtype2).split('\n')[1:])

    pnt_dict = dict()
    clr_dict = dict()
    for drug, resp_vals in resp_df.iteritems():
        resp_stat = ~resp_vals.isna()

        if resp_stat.sum() >= 100:
            clr_dict[str(drug)] = choose_label_colour(str(drug))
            use_resp = resp_vals[resp_stat]
            use_samps = set(use_resp.index) & set(ccle_df.columns)
            drug_size = resp_stat.mean()

            corr_x = -spearmanr(ccle_df.loc[plt_mtype1, use_samps],
                                use_resp[use_samps]).correlation
            corr_y = -spearmanr(ccle_df.loc[plt_mtype2, use_samps],
                                use_resp[use_samps]).correlation

            ax.scatter(corr_x, corr_y, s=drug_size * 601,
                       c=[clr_dict[str(drug)]], alpha=0.37, edgecolors='none')

            if (isinstance(drug, str) and not drug[-1].isnumeric()
                    and (abs(corr_x) > 0.2 or abs(corr_y) > 0.2)):
                pnt_dict[corr_x, corr_y] = drug_size ** 2, (str(drug), '')

            elif str(drug) == 'Dasatinib':
                pnt_dict[corr_x, corr_y] = drug_size ** 2, (str(drug), '')

            else:
                pnt_dict[corr_x, corr_y] = drug_size ** 2, ('', '')

    plt_xlims = np.array(ax.get_xlim())
    plt_ylims = np.array(ax.get_ylim())
    plt_min = min(plt_xlims[0], plt_ylims[0]) - 0.07
    plt_max = max(plt_xlims[1], plt_ylims[1]) + 0.07

    lbl_pos = place_labels(
        pnt_dict, lims=(np.mean([plt_xlims[0], plt_ylims[0]]) - 0.04,
                        np.mean([plt_xlims[1], plt_ylims[1]]) + 0.04),
        lbl_dens=1
        )

    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=15, ha=pos[1], va='bottom')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.013 + pnt_dict[pnt_x, pnt_y][0] / 23):
            use_clr = clr_dict[str(pnt_dict[pnt_x, pnt_y][1][0])]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.text(0.98, 0.03, "{}\nmutants".format(mtype_lbl1),
            size=13, c='#D99D00', ha='right', va='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.98, "{}\nmutants".format(mtype_lbl2),
            size=13, c=variant_clrs['Point'], ha='left', va='top',
            transform=ax.transAxes)

    ax.plot([plt_min, plt_max], [0, 0],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0, 0], [plt_min, plt_max], color='black', linewidth=1.3,
            linestyle=':', alpha=0.71)
    ax.plot([plt_min, plt_max], [plt_min, plt_max],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim(plt_xlims + [-0.07, 0.07])
    ax.set_ylim(plt_ylims + [-0.07, 0.07])

    ax.set_xlabel("Correlation Between Response and Subgrouping Scores",
                  size=21, weight='semibold')
    ax.set_ylabel("Correlation Between Response and Subgrouping Scores",
                  size=21, weight='semibold')

    plt.savefig(
        os.path.join(
            plot_dir, '__'.join([args.expr_source, args.cohort]), args.gene,
            "ortho-responses__{}__{}__{}.svg".format(
                plt_mtype1.get_filelabel(), plt_mtype2.get_filelabel(),
                args.classif
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

    parser.add_argument('--types', '-t', nargs='*',
                        help='a list of mutated genes', type=str)

    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-conf__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif)
            )
        ]

    out_list = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-conf__')[1].split('__')[:-1])}
        for out_data in out_datas
        ])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby(['Levels'])['Samps'].min()
    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError(
            "Cannot plot inferred scores until this experiment is run with "
            "mutation levels `Exon__Location__Protein` which tests genes' "
            "base mutations!"
            )

    pred_dict = dict()
    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()
    trnsf_dicts = dict()
    ccle_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_data = pickle.load(f)

            pred_dict[lvls] = pred_data.loc[[
                mtype for mtype in pred_data.index
                if (not isinstance(mtype, RandomType)
                    and select_mtype(mtype, args.gene))
                ]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_data = pickle.load(f)

            phn_dict.update({mtype: phn for mtype, phn in phn_data.items()
                             if select_mtype(mtype, args.gene)})

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_data = pickle.load(f)['mean']

            auc_dict[lvls] = auc_data[[select_mtype(mtype, args.gene)
                                       for mtype in auc_data.index]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_data = pickle.load(f)['mean']

            conf_dict[lvls] = conf_data.loc[[select_mtype(mtype, args.gene)
                                             for mtype in conf_data.index]]

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-trnsf__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                trnsf_dicts[lvls] = pickle.load(f)

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "trnsf-vals__{}__{}.p.gz".format(
                                              lvls, args.classif)),
                             'r') as f:
                ccle_mat = pickle.load(f)['CCLE']

                ccle_dict[lvls] = pd.DataFrame(
                    np.vstack(ccle_mat.values), index=ccle_mat.index,
                    columns=trnsf_dicts[lvls]['CCLE']['Samps']
                    )

    pred_df = pd.concat(pred_dict.values())
    if pred_df.shape[0] == 0:
        raise ValueError(
            "No classification tasks found for gene `{}`!".format(args.gene))

    auc_vals = pd.concat(auc_dict.values())
    conf_vals = pd.concat(conf_dict.values())
    ccle_df = pd.concat(ccle_dict.values())

    out_tag = "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                        out_use.min())
    cdata = merge_cohort_data(os.path.join(base_dir, out_tag), use_seed=8713)

    resp_df = load_response_data()
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    base_auc = auc_vals[base_mtype]

    ortho_dict = {
        mtype: mtype.get_sorted_levels()
        for mtype, auc_val in auc_vals.iteritems()
        if (not isinstance(mtype, RandomType) and auc_val > (base_auc - 0.05)
            and mtype.subtype_list()[0][1] != pnt_mtype
            and (mtype.subtype_list()[0][1] & copy_mtype).is_empty())
        }

    plt_mtypes = sorted(ortho_dict, key=lambda mtype: auc_vals[mtype])[::-1]
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort]),
                             args.gene),
                exist_ok=True)

    if plt_mtypes:
        plot_remainder_scores(plt_mtypes[:10], pred_df, auc_vals, phn_dict,
                              cdata, args)

    if args.types is not None:
        for plt_type in args.types:
            plot_projection_scores(plt_mtypes[:2], pred_df, auc_vals,
                                   phn_dict, cdata, plt_type, args)

    ortho_pairs = {
        (mtype1, mtype2)
        for (mtype1, lvls1), (mtype2, lvls2) in combn(ortho_dict.items(), 2)
        if ((compare_lvls(lvls1, lvls2) and (mtype1 & mtype2).is_empty())
            or (phn_dict[mtype1] | phn_dict[mtype2]).sum() == 0)
        }

    for mtype1, mtype2 in sorted(
            ortho_pairs, key=lambda mtypes: (
                (phn_dict[mtypes[0]] | phn_dict[mtypes[1]]).sum()
                * (1 - spearmanr(
                    pred_df.loc[mtypes[0]].apply(np.mean),
                    pred_df.loc[mtypes[1]].apply(np.mean)
                    ).correlation)
                )
            )[-25:]:

        plot_orthogonal_scores(mtype1, mtype2, pred_df, auc_vals,
                               phn_dict, cdata, args)
        plot_orthogonal_response(mtype1, mtype2, auc_vals, ccle_df, resp_df,
                                 cdata, args)


if __name__ == '__main__':
    main()

