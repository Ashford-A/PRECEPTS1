
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from .utils import filter_mtype
from .plot_ccle import load_response_data
from .plot_mutations import recurse_labels

from ..utilities.colour_maps import variant_clrs
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.label_placement import place_scatter_labels

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'partition')


def plot_remainder_scores(pred_mat, auc_vals, pheno_dict, cdata, args):
    fig, axarr = plt.subplots(figsize=(0.5 + pred_mat.shape[0] * 1.7, 7),
                              nrows=1, ncols=pred_mat.shape[0])

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    if pred_mat.shape[0] == 1:
        axarr = [axarr]

    for i, (ax, plt_mtype) in enumerate(zip(axarr, pred_mat.index)):
        use_lvls = cdata.find_pheno(plt_mtype)
        use_mtree = cdata.mtrees[use_lvls][args.gene]['Point']
        leaf_count = len(MuType(use_mtree.allkey()).leaves())

        sns.violinplot(x=pred_mat.loc[plt_mtype][~pheno_dict[base_mtype]],
                       ax=ax, palette=[variant_clrs['WT']], inner=None,
                       orient='v', linewidth=0, cut=0, width=0.89)
        sns.violinplot(x=pred_mat.loc[plt_mtype][pheno_dict[plt_mtype]],
                       ax=ax, palette=[variant_clrs['Point']], inner=None,
                       orient='v', linewidth=0, cut=0, width=0.89)

        rest_stat = pheno_dict[base_mtype] & ~pheno_dict[plt_mtype]
        if rest_stat.sum() > 10:
            sns.violinplot(x=pred_mat.loc[plt_mtype][rest_stat],
                           ax=ax, palette=['none'], inner=None, orient='v',
                           linewidth=1.7, cut=0, width=0.89)

        else:
            ax.scatter(np.random.randn(rest_stat.sum()) / 7.3,
                       pred_mat.loc[plt_mtype][rest_stat],
                       facecolor='none', s=31, alpha=0.53,
                       edgecolors='black', linewidth=0.9)

        ax.get_children()[0].set_alpha(0.41)
        ax.get_children()[1].set_alpha(0.41)
        ax.get_children()[2].set_facecolor((1, 1, 1, 0))
        ax.get_children()[2].set_edgecolor((0, 0, 0, 0.47))

        tree_ax = inset_axes(ax, width='100%', height='100%',
                             bbox_to_anchor=(0.03, 0.89, 0.94, 0.09),
                             bbox_transform=ax.transAxes, borderpad=0)
        tree_ax.axis('off')
        tree_mtype = get_subtype(get_subtype(plt_mtype))

        tree_ax = recurse_labels(tree_ax, use_mtree, (0, leaf_count),
                                 len(use_lvls) - 2, leaf_count,
                                 clr_mtype=tree_mtype, add_lbls=False,
                                 mut_clr=variant_clrs['Point'])

        mtype_lbl = get_fancy_label(get_subtype(plt_mtype),
                                    pnt_link='\nor\n', phrase_link='\n')
        ax.text(0.5, 1.01, mtype_lbl,
                size=8, ha='center', va='bottom', transform=ax.transAxes)

        ylims = ax.get_ylim()
        ygap = (ylims[1] - ylims[0]) / 7
        ax.set_ylim([ylims[0], ylims[1] + ygap])
        ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Subgrouping Inferred Score",
                          size=21, weight='semibold')
        else:
            ax.set_ylabel('')

    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]), args.gene,
                     "remainder-scores_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_orthogonal_response(plt_mtype1, plt_mtype2, auc_vals, ccle_df,
                             resp_df, cdata, args):
    fig, ax = plt.subplots(figsize=(11, 10))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    mtype_lbl1 = get_fancy_label(get_subtype(plt_mtype1),
                                 pnt_link='\nor ', phrase_link=' ')
    mtype_lbl2 = get_fancy_label(get_subtype(plt_mtype2),
                                 pnt_link='\nor ', phrase_link=' ')

    pnt_dict = dict()
    line_dict = dict()
    for drug, resp_vals in resp_df.iteritems():
        resp_stat = ~resp_vals.isna()

        if resp_stat.sum() >= 100:
            use_resp = resp_vals[resp_stat]
            use_samps = set(use_resp.index) & set(ccle_df.columns)

            corr_x = -spearmanr(ccle_df.loc[plt_mtype1, use_samps],
                                use_resp[use_samps]).correlation
            corr_y = -spearmanr(ccle_df.loc[plt_mtype2, use_samps],
                                use_resp[use_samps]).correlation

            use_clr = choose_label_colour(str(drug))
            line_dict[corr_x, corr_y] = dict(c=use_clr)
            drug_size = resp_stat.mean()

            ax.scatter(corr_x, corr_y, s=drug_size * 601,
                       c=[use_clr], alpha=0.37, edgecolors='none')

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

    ax.set_xlabel("Correlation Between Response and Subgrouping Scores",
                  size=21, weight='semibold')
    ax.set_ylabel("Correlation Between Response and Subgrouping Scores",
                  size=21, weight='semibold')

    if pnt_dict:
        lbl_pos = place_scatter_labels(
            pnt_dict, ax, plt_lims=[plt_xlims, plt_ylims],
            plt_type='scatter', font_size=11, line_dict=line_dict,
            linewidth=0.7, alpha=0.37
            )

    ax.set_xlim(plt_xlims + [-0.07, 0.07])
    ax.set_ylim(plt_ylims + [-0.07, 0.07])

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
    trnsf_dict = {lvls: dict() for lvls in out_use.index}
    trnsf_vals = dict()

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

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-trnsf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            trnsf_out = pickle.load(f)['CCLE']
            trnsf_dict[lvls]['Samps'] = trnsf_out['Samps']

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "trnsf-preds__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            trnsf_mat = pickle.load(f)['CCLE']

            trnsf_vals[lvls] = pd.DataFrame(np.vstack(trnsf_mat.values),
                                            index=trnsf_mat.index,
                                            columns=trnsf_dict[lvls]['Samps'])

    pred_df = pd.concat(pred_dict.values())
    if pred_df.shape[0] == 0:
        raise ValueError(
            "No classification tasks found for gene `{}`!".format(args.gene))

    auc_vals = pd.concat(auc_dict.values())
    trnsf_vals = pd.concat(trnsf_vals.values())
    resp_df = load_response_data()

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort]),
                             args.gene),
                exist_ok=True)

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    wt_phn = ~phn_dict[base_mtype]
    wt_preds = pred_df.loc[base_mtype][wt_phn].apply(np.mean)

    ortho_dict = {mtype: mtype.get_sorted_levels()
                  for mtype, auc_val in auc_vals.iteritems()
                  if (not isinstance(mtype, RandomType)
                      and auc_val >= 0.7 and mtype != base_mtype
                      and (get_subtype(mtype) & copy_mtype).is_empty())}

    if ortho_dict:
        pred_vals = pred_df.loc[ortho_dict].applymap(np.mean)

        corr_dict = {mtype: spearmanr(pred_vals.loc[mtype][wt_phn],
                                      wt_preds).correlation
                     for mtype in ortho_dict}

        divg_list = pd.Series({mtype: (auc_vals[mtype] - 0.5) * (1 - corr_val)
                               for mtype, corr_val in corr_dict.items()})

        plot_remainder_scores(
            pred_vals.loc[divg_list.sort_values().index[:10]],
            auc_vals, phn_dict, cdata, args
            )

        ortho_pairs = {(mtype2, mtype1)
                       for mtype1, mtype2 in combn(ortho_dict, 2)
                       if not (phn_dict[mtype1] & phn_dict[mtype2]).any()}

        for mtype1, mtype2 in sorted(
                ortho_pairs,
                key=lambda mtypes: (
                    (auc_vals[mtypes[0]] - 0.5) * (auc_vals[mtypes[1]] - 0.5)
                    * (1 - spearmanr(
                        pred_df.loc[mtypes[0]][wt_phn].apply(np.mean),
                        pred_df.loc[mtypes[1]][wt_phn].apply(np.mean)
                        ).correlation)
                    )
                )[-10:]:

            plot_orthogonal_response(mtype1, mtype2, auc_vals,
                                     trnsf_vals, resp_df, cdata, args)


if __name__ == '__main__':
    main()

