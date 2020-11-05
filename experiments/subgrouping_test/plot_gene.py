
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir, train_cohorts
from .utils import filter_mtype, choose_cohort_colour
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.label_placement import place_scatter_labels
from ..utilities.colour_maps import auc_cmap
from ..utilities.metrics import calc_conf

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# make plots cleaner by turning off outer box, make background all white
mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'gene')


def plot_sub_comparisons(auc_dict, conf_dict, pheno_dict, use_clf, args,
                         include_copy=False):
    fig, ax = plt.subplots(figsize=(11, 11))

    gene_mtype = MuType({('Gene', args.gene): pnt_mtype})
    plot_dict = dict()
    plt_min = 0.89

    # for each cohort, check if the given gene had subgroupings that were
    # tested, and get the results for all the gene's point mutations...
    for (src, coh), auc_vals in auc_dict.items():
        use_aucs = auc_vals[[mtype for mtype in auc_vals.index
                             if not isinstance(mtype, RandomType)]]

        if not include_copy:
            use_aucs = use_aucs[[
                mtype for mtype in use_aucs.index
                if (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
                ]]

        sub_aucs = use_aucs[[
            mtype for mtype in use_aucs.index
            if not ((tuple(mtype.subtype_iter())[0][1] & pnt_mtype).is_empty()
                    or tuple(mtype.subtype_iter())[0][1].is_supertype(
                        pnt_mtype))
            ]]

        if len(sub_aucs) > 0:
            best_subtype = sub_aucs.idxmax()

            if (tuple(best_subtype.subtype_iter())[0][1]
                    & copy_mtype).is_empty():
                base_mtype = gene_mtype
            else:
                base_mtype = (best_subtype - gene_mtype) | gene_mtype

            auc_tupl = use_aucs[base_mtype], use_aucs[best_subtype]
            base_size = np.mean(pheno_dict[src, coh][base_mtype])
            plt_size = 0.07 * base_size ** 0.5

            plot_dict[auc_tupl] = [plt_size, ('', '')]
            coh_lbl = get_cohort_label(coh)
            plt_min = min(plt_min, use_aucs[base_mtype] - 0.05,
                          use_aucs[best_subtype] - 0.07)

            best_prop = np.mean(pheno_dict[src, coh][best_subtype])
            best_prop /= base_size
            conf_sc = calc_conf(conf_dict[src, coh][best_subtype],
                                conf_dict[src, coh][base_mtype])

            if conf_sc > 0.8:
                plot_dict[auc_tupl][1] = coh_lbl, get_fancy_label(
                    tuple(best_subtype.subtype_iter())[0][1],
                    pnt_link='\nor ', phrase_link=' '
                    )

            else:
                plot_dict[auc_tupl][1] = coh_lbl, ''

            auc_bbox = (auc_tupl[0] - plt_size / 2,
                        auc_tupl[1] - plt_size / 2, plt_size, plt_size)

            # create the axis in which the pie chart will be plotted
            pie_ax = inset_axes(ax, width='100%', height='100%',
                                bbox_to_anchor=auc_bbox,
                                bbox_transform=ax.transData,
                                axes_kwargs=dict(aspect='equal'), borderpad=0)

            # plot the pie chart for the AUCs of the gene in this cohort
            pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                       colors=[choose_cohort_colour(coh) + (0.83, ),
                               choose_cohort_colour(coh) + (0.23, )],
                       wedgeprops=dict(edgecolor='black', linewidth=10 / 11))

    plt_lims = plt_min, 1 + (1 - plt_min) / 113
    ax.grid(linewidth=0.83, alpha=0.41)

    ax.plot([plt_min, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [plt_min, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([plt_min + 0.003, 0.999], [plt_min + 0.003, 0.999],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=27, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=27, weight='semibold')

    # figure out where to place the annotation labels for each cohort so that
    # they don't overlap with one another or the pie charts
    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[[plt_min + 0.01, 0.99]] * 2,
                                       font_size=19, seed=args.seed,
                                       c='black', linewidth=0.83, alpha=0.61)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    if include_copy:
        sub_lbl = "sub-copy"
    else:
        sub_lbl = "sub"

    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}-comparisons_{}.svg".format(sub_lbl, use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_conf_distributions(auc_vals, conf_dict, pheno_dict, use_clf, args):
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    coh_dict = dict()
    for (src, coh), conf_vals in conf_dict.items():
        use_confs = conf_vals[[mtype for mtype in conf_vals.index
                               if (not isinstance(mtype, RandomType)
                                   and (tuple(mtype.subtype_iter())[0][1]
                                        & copy_mtype).is_empty())]]

        if len(use_confs) > 1 and base_mtype in use_confs.index:
            conf_list = use_confs.apply(
                lambda confs: np.percentile(confs, 25))

            base_indx = conf_list.index.get_loc(base_mtype)
            best_subtype = conf_list[:base_indx].append(
                conf_list[(base_indx + 1):]).idxmax()

            if conf_list[best_subtype] > 0.6:
                coh_dict[src, coh] = (
                    choose_cohort_colour(coh), best_subtype,
                    calc_conf(use_confs[best_subtype],
                              use_confs[base_mtype])
                    )

    ymin = 0.83
    fig, axarr = plt.subplots(figsize=(0.3 + 1.7 * len(coh_dict), 7),
                              nrows=1, ncols=len(coh_dict), sharey=True,
                              squeeze=False)

    for i, ((src, coh), (coh_clr, best_subtype, conf_sc)) in enumerate(
            sorted(coh_dict.items(),
                   key=lambda x: auc_vals[x[0]][x[1][1]], reverse=True)
            ):
        coh_lbl = get_cohort_label(coh).replace('(', '\n(')

        plt_df = pd.concat([
            pd.DataFrame({'Type': 'Base',
                          'Conf': conf_dict[src, coh][base_mtype]}),
            pd.DataFrame({'Type': 'Subg',
                          'Conf': conf_dict[src, coh][best_subtype]})
            ])

        sns.violinplot(x=plt_df.Type, y=plt_df.Conf, ax=axarr[0, i],
                       order=['Subg', 'Base'], palette=[coh_clr, coh_clr],
                       cut=0, linewidth=1.3, width=0.93, inner=None)

        axarr[0, i].scatter(0, auc_vals[src, coh][best_subtype], 
                         s=41, c=[coh_clr], edgecolor='0.23', alpha=0.97)
        axarr[0, i].scatter(1, auc_vals[src, coh][base_mtype],
                            s=41, c=[coh_clr], edgecolor='0.23', alpha=0.53)

        axarr[0, i].set_title(coh_lbl, size=17, weight='semibold')
        axarr[0, i].get_children()[0].set_alpha(0.83)
        axarr[0, i].get_children()[1].set_alpha(0.26)

        axarr[0, i].text(0.5, 1 / 97, "{:.3f}".format(conf_sc),
                         size=17, ha='center', va='bottom',
                         transform=axarr[0, i].transAxes)

        axarr[0, i].plot([-0.5, 1.5], [0.5, 0.5], color='black',
                         linewidth=2.3, linestyle=':', alpha=0.83)
        axarr[0, i].plot([-0.5, 1.5], [1, 1], color='black',
                         linewidth=1.7, alpha=0.83)

        axarr[0, i].set_xlabel('')
        axarr[0, i].set_xticklabels([])
        ymin = min(ymin, min(conf_dict[src, coh][base_mtype]) - 0.04,
                   min(conf_dict[src, coh][best_subtype]) - 0.04)

        if i == 0:
            axarr[0, i].set_ylabel('AUCs', size=21, weight='semibold')
        else:
            axarr[0, i].set_ylabel('')

    if 0.47 < ymin < 0.51:
        ymin = 0.445
    for i in range(len(coh_dict)):
        axarr[0, i].set_xlim([-0.59, 1.59])
        axarr[0, i].set_ylim([ymin, 1 + (1 - ymin) / 31])

    fig.tight_layout(w_pad=1.3)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "conf-distributions_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_transfer_aucs(trnsf_dict, auc_dict, conf_dict, pheno_dict,
                       use_clf, args):
    fig, axarr = plt.subplots(figsize=(13, 7), nrows=2, ncols=1)

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    conf_agg = dict()

    for (src, coh), conf_vals in conf_dict.items():
        if base_mtype in conf_vals.index:
            use_confs = conf_vals[[
                mtype for mtype in conf_vals.index
                if (mtype != base_mtype and not isinstance(mtype, RandomType)
                    and (tuple(mtype.subtype_iter())[0][1]
                         & copy_mtype).is_empty()
                    and any(mtype in auc_df.index
                            for auc_df in trnsf_dict.values()))
                ]]

            for mtype, conf_list in use_confs.iteritems():
                conf_sc = np.sum(pheno_dict[src, coh][mtype])
                conf_sc *= calc_conf(conf_list, conf_vals[base_mtype]) - 0.5

                if mtype in conf_agg:
                    conf_agg[mtype] += conf_sc
                else:
                    conf_agg[mtype] = conf_sc

    best_subtype = sorted(conf_agg.items(), key=lambda x: x[1])[-1][0]
    for ax, mtype in zip(axarr, [base_mtype, best_subtype]):
        trnsf_mat = pd.Series({cohs: auc_vals[mtype]
                               for cohs, auc_vals in trnsf_dict.items()
                               if mtype in auc_vals.index}).unstack()

        auc_cmap.set_bad('black')
        xlabs = [get_cohort_label(coh) for coh in trnsf_mat.columns]
        ylabs = [get_cohort_label(coh) for _, coh in trnsf_mat.index]

        sns.heatmap(trnsf_mat, cmap=auc_cmap, ax=ax, vmin=0, vmax=1,
                    xticklabels=xlabs, yticklabels=ylabs,
                    cbar_kws={'aspect': 7})

        plt_ylims = ax.get_ylim()
        ax.set_ylim([plt_ylims[1] - 0.5, plt_ylims[0] + 0.5])

        ax.set_title(get_fancy_label(tuple(mtype.subtype_iter())[0][1]),
                     size=19)
        ax.set_xticklabels(xlabs, size=12, ha='right', rotation=37)
        ax.set_yticklabels(ylabs, size=12, ha='right', rotation=0)

        ax.collections = [ax.collections[-1]]
        cbar = ax.collections[-1].colorbar
        cbar.ax.tick_params(labelsize=13)
        cbar.ax.set_title('AUC', size=17, weight='semibold')

    fig.text(0.5, -0.02, "Testing Cohort", fontsize=23, weight='semibold',
             ha='center', va='top')
    fig.text(0, 0.5, "Training Cohort", fontsize=23, weight='semibold',
             rotation=90, ha='right', va='center')

    fig.tight_layout(h_pad=1.7)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "transfer-aucs_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_transfer_comparisons(trnsf_dict, conf_dict, pheno_dict,
                              use_clf, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    conf_agg = dict()

    for (src, coh), conf_vals in conf_dict.items():
        if base_mtype in conf_vals.index:
            use_confs = conf_vals[[
                mtype for mtype in conf_vals.index
                if (mtype != base_mtype and not isinstance(mtype, RandomType)
                    and (tuple(mtype.subtype_iter())[0][1]
                         & copy_mtype).is_empty())
                ]]

            for mtype, conf_list in use_confs.iteritems():
                conf_sc = np.sum(pheno_dict[src, coh][mtype])
                conf_sc *= calc_conf(conf_list, conf_vals[base_mtype]) - 0.5

                if mtype in conf_agg:
                    conf_agg[mtype] += conf_sc
                else:
                    conf_agg[mtype] = conf_sc

    best_subtype = sorted(conf_agg.items(), key=lambda x: x[1])[-1][0]
    plot_dict = dict()
    plt_min = 0.83

    for (src, train_coh, trnsf_coh), auc_vals in trnsf_dict.items():
        if (base_mtype in auc_vals.index and best_subtype in auc_vals.index
                and trnsf_coh in train_cohorts):
            coh_lbl = ' \u2192 '.join([get_cohort_label(train_coh),
                                       get_cohort_label(trnsf_coh)])

            coh_clr = choose_cohort_colour(train_coh)
            auc_tupl = auc_vals.loc[base_mtype], auc_vals.loc[best_subtype]
            base_size = np.mean(pheno_dict[src, train_coh][base_mtype])
            plt_size = 0.07 * base_size ** 0.5

            best_prop = np.mean(pheno_dict[src, train_coh][best_subtype])
            best_prop /= base_size
            plt_min = min(plt_min, auc_tupl[0] - 0.05, auc_tupl[1] - 0.07)

            if auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                plot_dict[auc_tupl] = plt_size, (coh_lbl, '')
            else:
                plot_dict[auc_tupl] = plt_size, ('', '')

            auc_bbox = (auc_tupl[0] - plt_size / 2,
                        auc_tupl[1] - plt_size / 2, plt_size, plt_size)

            pie_ax = inset_axes(ax, width='100%', height='100%',
                                bbox_to_anchor=auc_bbox,
                                bbox_transform=ax.transData,
                                axes_kwargs=dict(aspect='equal'), borderpad=0)

            pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                       colors=[coh_clr + (0.83, ), coh_clr + (0.23, )],
                       wedgeprops=dict(edgecolor='black', linewidth=13 / 11))

    plt_lims = plt_min, 1 + (1 - plt_min) / 103
    ax.grid(linewidth=0.83, alpha=0.41)

    ax.plot([plt_min, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [plt_min, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([plt_min, 0.997], [plt_min, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("Transfer AUC\nusing gene-wide classifier",
                  size=27, weight='semibold')
    ax.set_ylabel("Transfer AUC\nusing best found subgrouping",
                  size=27, weight='semibold')

    # figure out where to place the labels for each point, and plot them
    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[[plt_min + 0.01, 0.99]] * 2,
                                       font_size=19, seed=args.seed,
                                       c='black', linewidth=0.83, alpha=0.61)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "transfer-comparisons_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_gene',
        description="Plots results for a given gene across all cohorts."
        )

    parser.add_argument('gene', help='a mutated gene', type=str)
    parser.add_argument(
        '--seed', default=9401, type=int,
        help="the random seed to use for setting plotting parameters"
        )

    # parse command line arguments, get list of experiments matching given
    # criteria that have run to completion
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "*__*__samps-*", "out-trnsf__*__*.p.gz"))
        ]

    # parse out input attributes of each experiment
    out_list = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "out-trnsf__")[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split(".p.gz")[0]}
        for out_data in out_datas
        ])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found!")

    # filter out experiments that have not tested gene-wide classifiers as
    # well as those that did not test the most subgroupings for a given
    # combination of input attributes
    out_use = out_list.groupby(['Source', 'Cohort', 'Classif']).filter(
        lambda outs: 'Consequence__Exon' in set(outs.Levels)
        ).groupby(['Source', 'Cohort', 'Levels', 'Classif'])['Samps'].min()

    out_use = out_use[out_use.index.get_level_values('Cohort').isin(
        train_cohorts)]

    # ensure gene-wide classifier input is read in first for each experiment
    out_lvls = set(out_use.index.get_level_values('Levels'))
    out_use = out_use.reindex(['Consequence__Exon']
                              + sorted(out_lvls - {'Consequence__Exon'}),
                              level='Levels')

    phn_dict = dict()
    auc_dict = dict()
    trnsf_aucs = dict()
    conf_dict = dict()

    for (src, coh, lvls, clf), ctf in tuple(out_use.iteritems()):
        out_tag = "{}__{}__samps-{}".format(src, coh, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, clf)),
                         'r') as f:
            phns = pickle.load(f)

        phn_vals = {mtype: phn for mtype, phn in phns.items()
                    if filter_mtype(mtype, args.gene)}

        if phn_vals:
            if (src, coh) in phn_dict:
                phn_dict[src, coh].update(phn_vals)
            else:
                phn_dict[src, coh] = phn_vals

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-aucs__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                auc_vals = pickle.load(f)['mean']

            auc_vals = auc_vals[[mtype for mtype in auc_vals.index
                                 if filter_mtype(mtype, args.gene)]]

            if (src, coh, clf) in auc_dict:
                auc_dict[src, coh, clf] = auc_dict[src, coh, clf].append(
                    auc_vals)
            else:
                auc_dict[src, coh, clf] = auc_vals

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-trnsf__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                trnsf_data = pickle.load(f)

            for trnsf_coh, trnsf_out in trnsf_data.items():
                if trnsf_out['AUC'].shape[0] > 0:
                    auc_vals = trnsf_out['AUC'].loc[[
                        filter_mtype(mtype, args.gene)
                        for mtype in trnsf_out['AUC'].index], 'mean'
                        ]

                    if (src, clf, coh, trnsf_coh) in trnsf_aucs:
                        trnsf_aucs[src, clf, coh, trnsf_coh] = pd.concat([
                            trnsf_aucs[src, clf, coh, trnsf_coh],
                            auc_vals
                            ])
                    else:
                        trnsf_aucs[src, clf, coh, trnsf_coh] = auc_vals

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-conf__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                conf_vals = pickle.load(f)

            conf_vals = conf_vals[[mtype for mtype in conf_vals.index
                                   if filter_mtype(mtype, args.gene)]]

            if (src, coh, clf) in conf_dict:
                conf_dict[src, coh, clf] = conf_dict[src, coh, clf].append(
                    conf_vals)
            else:
                conf_dict[src, coh, clf] = conf_vals

    if not phn_dict:
        raise ValueError("No experiment output found for "
                         "gene `{}`!".format(args.gene))

    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    plt_tbl = out_use.groupby(['Source', 'Cohort', 'Classif']).count()
    plt_tbl = plt_tbl[plt_tbl == 4]
    plt_clfs = plt_tbl.index.get_level_values('Classif').value_counts()

    for clf in plt_clfs[plt_clfs > 1].index:
        clf_aucs = {(src, coh): auc_data
                    for (src, coh, out_clf), auc_data in auc_dict.items()
                    if out_clf == clf}
        clf_confs = {(src, coh): conf_data
                     for (src, coh, out_clf), conf_data in conf_dict.items()
                     if out_clf == clf}

        clf_trnsf = {
            (src, coh, t_coh): trnsf_data
            for (src, out_clf, coh, t_coh), trnsf_data in trnsf_aucs.items()
            if out_clf == clf
            }

        plot_sub_comparisons(clf_aucs, clf_confs, phn_dict, clf, args,
                             include_copy=False)
        plot_sub_comparisons(clf_aucs, clf_confs, phn_dict, clf, args,
                             include_copy=True)
        plot_conf_distributions(clf_aucs, clf_confs, phn_dict, clf, args)

        plot_transfer_aucs(clf_trnsf, clf_aucs, clf_confs, phn_dict,
                           clf, args)
        plot_transfer_comparisons(clf_trnsf, clf_confs, phn_dict,
                                  clf, args)


if __name__ == '__main__':
    main()

