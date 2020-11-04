
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, dup_mtype, loss_mtype, RandomType)
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.metrics import calc_conf
from ..utilities.misc import choose_label_colour
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aupr_score

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'accuracy')


def plot_aupr_comparisons(auc_vals, pred_df, pheno_dict, conf_vals, args):
    fig, (base_ax, subg_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    plot_dicts = {'Base': dict(), 'Subg': dict()}
    line_dicts = {'Base': dict(), 'Subg': dict()}
    plt_max = 0.53

    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            if auc_vec[best_subtype] > 0.6:
                base_size = np.mean(pheno_dict[base_mtype])
                plt_size = 0.07 * base_size ** 0.5
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                base_infr = pred_df.loc[base_mtype].apply(np.mean)
                best_infr = pred_df.loc[best_subtype].apply(np.mean)

                base_auprs = (aupr_score(pheno_dict[base_mtype], base_infr),
                              aupr_score(pheno_dict[base_mtype], best_infr))
                subg_auprs = (aupr_score(pheno_dict[best_subtype], base_infr),
                              aupr_score(pheno_dict[best_subtype], best_infr))

                conf_sc = calc_conf(conf_vals[best_subtype],
                                    conf_vals[base_mtype])

                base_lbl = '', ''
                subg_lbl = '', ''
                min_diff = np.log2(1.25)

                mtype_lbl = get_fancy_label(
                    tuple(best_subtype.subtype_iter())[0][1],
                    pnt_link='\nor ', phrase_link=' '
                    )

                if conf_sc > 0.9:
                    base_lbl = gene, mtype_lbl
                    subg_lbl = gene, mtype_lbl

                elif (auc_vec[base_indx] > 0.75
                        or auc_vec[best_subtype] > 0.75):
                    base_lbl = gene, ''
                    subg_lbl = gene, ''

                elif auc_vec[base_indx] > 0.6 or auc_vec[best_subtype] > 0.6:
                    if abs(np.log2(base_auprs[1] / base_auprs[0])) > min_diff:
                        base_lbl = gene, ''
                    if abs(np.log2(subg_auprs[1] / subg_auprs[0])) > min_diff:
                        subg_lbl = gene, ''

                for lbl, auprs, mtype_lbl in zip(['Base', 'Subg'],
                                                 (base_auprs, subg_auprs),
                                                 [base_lbl, subg_lbl]):
                    plot_dicts[lbl][auprs] = plt_size, mtype_lbl
                    line_dicts[lbl][auprs] = dict(c=choose_label_colour(gene))

                for ax, lbl, (base_aupr, subg_aupr) in zip(
                        [base_ax, subg_ax], ['Base', 'Subg'],
                        [base_auprs, subg_auprs]
                        ):
                    plt_max = min(1.005,
                                  max(plt_max,
                                      base_aupr + 0.11, subg_aupr + 0.11))

                    auc_bbox = (base_aupr - plt_size / 2,
                                subg_aupr - plt_size / 2, plt_size, plt_size)

                    pie_ax = inset_axes(
                        ax, width='100%', height='100%',
                        bbox_to_anchor=auc_bbox, bbox_transform=ax.transData,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    use_clr = line_dicts[lbl][base_aupr, subg_aupr]['c']
                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               colors=[use_clr + (0.77, ),
                                       use_clr + (0.29, )],
                               explode=[0.29, 0], startangle=90)

    base_ax.set_title("AUPR on all point mutations",
                      size=21, weight='semibold')
    subg_ax.set_title("AUPR on best subgrouping mutations",
                      size=21, weight='semibold')

    for ax, lbl in zip([base_ax, subg_ax], ['Base', 'Subg']):
        ax.grid(linewidth=0.83, alpha=0.41)

        ax.plot([0, plt_max], [0, 0],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([0, 0], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [1, 1],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([1, 1], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [0, plt_max],
                color='#550000', linewidth=1.7, linestyle='--', alpha=0.37)

        ax.set_xlabel("using all point mutation inferred scores",
                      size=19, weight='semibold')
        ax.set_ylabel("using best found subgrouping inferred scores",
                      size=19, weight='semibold')

        if plot_dicts[lbl]:
            lbl_pos = place_scatter_labels(
                plot_dicts[lbl], ax, plt_lims=[[plt_max / 67, plt_max]] * 2,
                line_dict=line_dicts[lbl]
                )

        ax.set_xlim([-plt_max / 181, plt_max])
        ax.set_ylim([-plt_max / 181, plt_max])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "aupr-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_remainder_scores(pred_df, auc_vals, conf_vals, pheno_dict, args):
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    plt_mtypes = dict()
    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            if auc_vec[best_subtype] > 0.8:
                conf_sc = calc_conf(conf_vals[best_subtype],
                                    conf_vals[base_mtype])

                if conf_sc > 0.6:
                    plt_mtypes[gene] = best_subtype

    plt_mtypes = sorted(plt_mtypes.items(),
                        key=lambda x: auc_vals[x[1]])[::-1][:10]
    fig, axarr = plt.subplots(figsize=(0.7 + 2.1 * len(plt_mtypes), 6),
                              nrows=1, ncols=len(plt_mtypes))

    for i, (ax, (gene, plt_mtype)) in enumerate(zip(axarr, plt_mtypes)):
        pred_vals = pred_df.loc[plt_mtype].apply(np.mean)

        base_mtype = MuType({('Gene', gene): pnt_mtype})
        mtype_lbl = get_fancy_label(tuple(plt_mtype.subtype_iter())[0][1],
                                    pnt_link='\nor ', phrase_link='\n')

        ax.text(0.5, 1.01, gene, size=16, weight='semibold',
                ha='center', va='bottom', transform=ax.transAxes)
        ax.text(0.5, 0.99, mtype_lbl,
                size=9, ha='center', va='top', transform=ax.transAxes)

        sns.violinplot(x=pred_vals[~pheno_dict[base_mtype]],
                       ax=ax, palette=[variant_clrs['WT']],
                       inner=None, orient='v', linewidth=0, cut=0, width=0.83)
        sns.violinplot(x=pred_vals[pheno_dict[plt_mtype]],
                       ax=ax, palette=[variant_clrs['Point']],
                       inner=None, orient='v', linewidth=0, cut=0, width=0.83)

        rest_stat = pheno_dict[base_mtype] & ~pheno_dict[plt_mtype]
        sns.violinplot(x=pred_vals[rest_stat],
                       ax=ax, palette=['none'], inner=None, orient='v',
                       linewidth=1.7, cut=0, width=0.83)

        plt_min, plt_max = ax.get_ylim()
        ax.set_ylim([plt_min, plt_max + ((plt_max - plt_min)
                                         * (1.7 + len(mtype_lbl)) / 53)])

        ax.get_children()[0].set_alpha(0.41)
        ax.get_children()[1].set_alpha(0.41)

        if rest_stat.sum() > 1:
            ax.get_children()[2].set_facecolor((1, 1, 1, 0))
            ax.get_children()[2].set_edgecolor((0, 0, 0, 0.47))

        ax.set_yticklabels([])
        ax.set_ylabel('')

    axarr[0].text(-0.13, 0.5, "Subgrouping Classifier Score",
                  size=19, weight='semibold', ha='right', va='center',
                  rotation=90, transform=axarr[0].transAxes)

    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]),
                     "remainder-scores_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_random_comparison(auc_vals, conf_vals, pheno_dict, args):
    base_aucs = auc_vals[[
        mtype for mtype in auc_vals.index
        if (not isinstance(mtype, RandomType)
            and tuple(mtype.subtype_iter())[0][1] != pnt_mtype
            and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty())
        ]]

    rand_aucs = auc_vals[[mtype for mtype in auc_vals.index
                          if (isinstance(mtype, RandomType)
                              and mtype.base_mtype is not None)]]

    base_grps = base_aucs[base_aucs > 0.7].groupby(
        lambda mtype: tuple(mtype.label_iter())[0])
    fig, axarr = plt.subplots(figsize=(0.5 + 1.5 * len(base_grps), 7),
                              nrows=1, ncols=len(base_grps))
    plt_min = auc_vals.min()
 
    for i, (gene, auc_vec) in enumerate(sorted(base_grps,
                                               key=lambda x: x[1].max(),
                                               reverse=True)):
        base_mtype = MuType({('Gene', gene): pnt_mtype})
        axarr[i].set_title(gene, size=19, weight='semibold')

        plt_df = pd.concat([
            pd.DataFrame({
                'AUC': base_aucs[[mtype for mtype in base_aucs.index
                                  if tuple(mtype.label_iter())[0] == gene]],
                'Type': 'Orig'
                }),

            pd.DataFrame({
                'AUC': rand_aucs[[
                    mtype for mtype in rand_aucs.index
                    if (tuple(mtype.base_mtype.label_iter())[0] == gene
                        and mtype.size_dist < pheno_dict[base_mtype].sum())
                    ]],
                'Type': 'Rand'
                })
            ])

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

        axarr[i].plot([-0.6, 1.6], [auc_vals[base_mtype]] * 2,
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

        conf_sc = calc_conf(
            conf_vals[plt_df.loc[plt_df.Type == 'Orig'].AUC.idxmax()],
            conf_vals[plt_df.loc[plt_df.Type == 'Rand'].AUC.idxmax()]
            )

        if conf_sc == 1:
            conf_lbl = "1.0"
        elif 0.9995 < conf_sc < 1:
            conf_lbl = ">0.999"
        else:
            conf_lbl = "{:.3f}".format(conf_sc)

        axarr[i].text(0.5, -0.06, conf_lbl, size=17, ha='center', va='top',
                      transform=axarr[i].transAxes)

        if i == 0:
            axarr[i].set_ylabel('AUC', size=21, weight='semibold')
        else:
            axarr[i].set_yticklabels([])
            axarr[i].set_ylabel('')

        axarr[i].set_xlim([-0.6, 1.6])
        axarr[i].set_ylim([plt_min - 0.041, 1.007])

    fig.tight_layout(w_pad=1.1)
    plt.savefig(
        os.path.join(plot_dir,
                     '__'.join([args.expr_source, args.cohort]),
                     "random-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_accuracy',
        description="Plots alternative methods of measuring task performance."
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

    pred_dict = dict()
    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_dict[lvls] = pickle.load(f)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_dict[lvls] = pickle.load(f)['mean']

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_dict[lvls] = pickle.load(f)

    pred_df = pd.concat(pred_dict.values())
    auc_vals = pd.concat(auc_dict.values())
    conf_vals = pd.concat(conf_dict.values())

    # create the plots
    plot_aupr_comparisons(auc_vals, pred_df, phn_dict, conf_vals, args)
    plot_remainder_scores(pred_df, auc_vals, conf_vals, phn_dict, args)
    plot_random_comparison(auc_vals, conf_vals, phn_dict, args)


if __name__ == '__main__':
    main()

