
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, train_cohorts)
from HetMan.experiments.subvariant_tour.utils import RandomType
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_test.utils import (
    get_fancy_label, get_cohort_label)
from HetMan.experiments.subvariant_test.plot_aucs import place_labels
from HetMan.experiments.subvariant_test.plot_copy import select_mtype
from HetMan.experiments.utilities.pcawg_colours import cohort_clrs
from HetMan.experiments.utilities import auc_cmap

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# make plots cleaner by turning off outer box, make background all white
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def choose_cohort_colour(cohort):
    coh_base = cohort.split('_')[0]

    # if using a non-TCGA cohort, match to a TCGA cohort of the same
    # disease type, using white for pan-cancer cohorts
    if coh_base == 'METABRIC':
        use_clr = cohort_clrs['BRCA']
    elif coh_base == 'beatAML':
        use_clr = cohort_clrs['LAML']
    elif coh_base == 'CCLE':
        use_clr = '#000000'

    # otherwise, choose the colour according to the PCAWG scheme
    else:
        use_clr = cohort_clrs[coh_base]

    # convert the hex colour to a [0-1] RGB tuple
    return tuple(int(use_clr.lstrip('#')[i:(i + 2)], 16) / 256
                 for i in range(0, 6, 2))


def plot_sub_comparisons(auc_dict, conf_dict, pheno_dict, use_clf, args,
                         include_copy=False):
    fig, ax = plt.subplots(figsize=(11, 11))

    gene_mtype = MuType({('Gene', args.gene): pnt_mtype})
    plt_min = 0.89
    pnt_dict = dict()

    # for each cohort, check if the given gene had subgroupings that were
    # tested, and get the results for all the gene's point mutations...
    for coh, auc_vals in auc_dict.items():
        use_aucs = auc_vals[[mtype for mtype in auc_vals.index
                             if not isinstance(mtype, RandomType)]]

        if not include_copy:
            use_aucs = use_aucs[[
                mtype for mtype in use_aucs.index
                if (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
                ]]

        sub_aucs = use_aucs[[
            mtype for mtype in use_aucs.index
            if not ((mtype.subtype_list()[0][1] & pnt_mtype).is_empty()
                    or mtype.subtype_list()[0][1].is_supertype(pnt_mtype))
            ]]

        if (len(use_aucs) > 1 and gene_mtype in use_aucs.index
                and len(sub_aucs) > 0):
            gene_indx = use_aucs.index.get_loc(gene_mtype)
            best_subtype = sub_aucs.idxmax()

            if (best_subtype.subtype_list()[0][1] & copy_mtype).is_empty():
                base_mtype = gene_mtype
            else:
                base_mtype = (best_subtype - gene_mtype) | gene_mtype

            plt_min = min(plt_min, use_aucs[base_mtype] - 0.04,
                          use_aucs[best_subtype] - 0.04)
            base_size = np.mean(pheno_dict[coh][base_mtype])
            best_prop = np.mean(pheno_dict[coh][best_subtype]) / base_size

            conf_sc = np.greater.outer(conf_dict[coh][best_subtype],
                                       conf_dict[coh][base_mtype]).mean()

            if conf_sc > 0.8:
                mtype_lbl = '\n'.join(
                    get_fancy_label(best_subtype).split('\n')[1:])
                pnt_dict[use_aucs[base_mtype], use_aucs[best_subtype]] = [
                    base_size ** 0.47, (coh, mtype_lbl)]

            else:
                pnt_dict[use_aucs[base_mtype], use_aucs[best_subtype]] = [
                    base_size ** 0.47, (coh, '')]

            # create the axis in which the pie chart will be plotted
            pie_ax = inset_axes(
                ax, width=base_size ** 0.5, height=base_size ** 0.5,
                bbox_to_anchor=(use_aucs[base_mtype], use_aucs[best_subtype]),
                bbox_transform=ax.transData, loc=10,
                axes_kwargs=dict(aspect='equal'), borderpad=0
                )

            # plot the pie chart for the AUCs of the gene in this cohort
            pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                       colors=[choose_cohort_colour(coh) + (0.83, ),
                               choose_cohort_colour(coh) + (0.23, )],
                       wedgeprops=dict(edgecolor='black', linewidth=10 / 11))

    for pnt_x, pnt_y in pnt_dict:
        pnt_dict[pnt_x, pnt_y][0] *= (1 - plt_min) * 1.31

    # figure out where to place the annotation labels for each cohort so that
    # they don't overlap with one another or the pie charts
    lbl_pos = place_labels(pnt_dict,
                           lims=(plt_min + 0.01, 1 - (1 - plt_min) / 41),
                           lbl_dens=0.67, seed=args.seed)

    for (pnt_x, pnt_y), pos in lbl_pos.items():
        coh_lbl = get_cohort_label(pnt_dict[pnt_x, pnt_y][1][0])

        ax.text(pos[0][0], pos[0][1] + 700 ** -1, coh_lbl,
                size=23, ha=pos[1], va='bottom')
        ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][1],
                size=17, ha=pos[1], va='top')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.017 + pnt_dict[pnt_x, pnt_y][0] / (83 * plt_min)):
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] * (1 - plt_min) ** 0.61
            pnt_gap /= ln_lngth * 13
            lbl_gap = (1 - plt_min) / (ln_lngth * 61)

            # ...create a line connecting the pie chart to the label
            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     (pos[0][1] + lbl_gap * y_delta
                      + 0.008 + 0.004 * np.sign(y_delta))],
                    c='black', linewidth=0.9, alpha=0.71)

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

    ax.set_xlim([plt_min, 1 + (1 - plt_min) / 103])
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 103])

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=27, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=27, weight='semibold')

    if include_copy:
        sub_lbl = "sub-copy"
    else:
        sub_lbl = "sub"

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "{}-comparisons_{}.svg".format(sub_lbl, use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_conf_distributions(auc_vals, conf_dict, pheno_dict, use_clf, args):
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    coh_dict = dict()
    for coh, conf_vals in conf_dict.items():
        use_confs = conf_vals[[
            mtype for mtype in conf_vals.index
            if (not isinstance(mtype, RandomType)
                and (mtype.subtype_list()[0][1] & copy_mtype).is_empty())
            ]]

        if len(use_confs) > 1 and base_mtype in use_confs.index:
            conf_list = use_confs.apply(
                lambda confs: np.percentile(confs, 25))

            base_indx = conf_list.index.get_loc(base_mtype)
            best_subtype = conf_list[:base_indx].append(
                conf_list[(base_indx + 1):]).idxmax()
            best_indx = conf_list.index.get_loc(best_subtype)

            if conf_list[best_indx] > 0.6:
                coh_dict[coh] = (
                    choose_cohort_colour(coh), best_subtype,
                    np.greater.outer(use_confs[best_subtype],
                                     use_confs[base_mtype]).mean()
                    )

    ymin = 0.83
    fig, axarr = plt.subplots(figsize=(0.3 + 1.7 * len(coh_dict), 7),
                              nrows=1, ncols=len(coh_dict), sharey=True,
                              squeeze=False)

    for i, (coh, (coh_clr, best_subtype, conf_sc)) in enumerate(
            sorted(coh_dict.items(),
                   key=lambda x: auc_vals[x[0]][x[1][1]], reverse=True)
            ):
        coh_lbl = get_cohort_label(coh).replace('(', '\n(')

        plt_df = pd.concat([
            pd.DataFrame({'Type': 'Base',
                          'Conf': conf_dict[coh][base_mtype]}),
            pd.DataFrame({'Type': 'Subg',
                          'Conf': conf_dict[coh][best_subtype]})
            ])

        sns.violinplot(x=plt_df.Type, y=plt_df.Conf, ax=axarr[0, i],
                       order=['Subg', 'Base'], palette=[coh_clr, coh_clr],
                       cut=0, linewidth=1.3, width=0.93, inner=None)

        axarr[0, i].scatter(0, auc_vals[coh][best_subtype], 
                         s=41, c=[coh_clr], edgecolor='0.23', alpha=0.97)
        axarr[0, i].scatter(1, auc_vals[coh][base_mtype],
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
        ymin = min(ymin, min(conf_dict[coh][base_mtype]) - 0.04,
                   min(conf_dict[coh][best_subtype]) - 0.04)

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
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "conf-distributions_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_transfer_aucs(trnsf_dict, auc_dict, conf_dict, pheno_dict,
                       use_clf, args):
    fig, axarr = plt.subplots(figsize=(13, 7), nrows=2, ncols=1)

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    conf_agg = dict()

    for coh, conf_vals in conf_dict.items():
        if base_mtype in conf_vals.index:
            use_confs = conf_vals[[
                mtype for mtype in conf_vals.index
                if (mtype != base_mtype and not isinstance(mtype, RandomType)
                    and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
                    and any(mtype in auc_df.index
                            for auc_df in trnsf_dict.values()))
                ]]

            for mtype, conf_list in use_confs.iteritems():
                conf_sc = np.sum(pheno_dict[coh][mtype]) * (np.greater.outer(
                    conf_list, conf_vals[base_mtype]).mean() - 0.5)

                if mtype in conf_agg:
                    conf_agg[mtype] += conf_sc
                else:
                    conf_agg[mtype] = conf_sc

    best_subtype = sorted(conf_agg.items(), key=lambda x: x[1])[-1][0]
    for ax, mtype in zip(axarr, [base_mtype, best_subtype]):
        trnsf_mat = pd.Series({cohs: auc_df.loc[mtype, 'mean']
                               for cohs, auc_df in trnsf_dict.items()
                               if mtype in auc_df.index}).unstack()

        auc_cmap.set_bad('black')
        sns.heatmap(trnsf_mat, cmap=auc_cmap,
                    vmin=0, vmax=1, cbar_kws={'aspect': 7}, ax=ax)

        plt_ylims = ax.get_ylim()
        ax.set_ylim([plt_ylims[1] - 0.5, plt_ylims[0] + 0.5])

        mtype_lbl = ' '.join(get_fancy_label(mtype).split('\n')[1:])
        xlabs = [get_cohort_label(coh) for coh in trnsf_mat.columns]
        ylabs = [get_cohort_label(coh) for coh in trnsf_mat.index]

        ax.set_title(mtype_lbl, size=19)
        ax.set_xticklabels(xlabs, size=12, ha='right', rotation=37)
        ax.set_yticklabels(ylabs, size=12, ha='right', rotation=0)

        ax.collections = [ax.collections[-1]]
        cbar = ax.collections[-1].colorbar
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_title('AUC', size=17, weight='semibold')

    fig.text(0.5, -0.02, "Testing Cohort", fontsize=23, weight='semibold',
             ha='center', va='top')
    fig.text(0, 0.5, "Training Cohort", fontsize=23, weight='semibold',
             rotation=90, ha='right', va='center')

    fig.tight_layout(h_pad=1.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "transfer-aucs_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_transfer_comparisons(trnsf_dict, conf_dict, pheno_dict,
                              use_clf, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    conf_agg = dict()

    for coh, conf_vals in conf_dict.items():
        if base_mtype in conf_vals.index:
            use_confs = conf_vals[[
                mtype for mtype in conf_vals.index
                if (mtype != base_mtype and not isinstance(mtype, RandomType)
                    and (mtype.subtype_list()[0][1] & copy_mtype).is_empty())
                ]]

            for mtype, conf_list in use_confs.iteritems():
                conf_sc = np.sum(pheno_dict[coh][mtype]) * (np.greater.outer(
                    conf_list, conf_vals[base_mtype]).mean() - 0.5)

                if mtype in conf_agg:
                    conf_agg[mtype] += conf_sc
                else:
                    conf_agg[mtype] = conf_sc

    best_subtype = sorted(conf_agg.items(), key=lambda x: x[1])[-1][0]
    plt_min = 0.83
    pnt_dict = dict()
    clr_dict = dict()

    for (train_coh, trnsf_coh), auc_df in trnsf_dict.items():
        if (base_mtype in auc_df.index and best_subtype in auc_df.index
                and trnsf_coh in train_cohorts):
            coh_lbl = ' \u2192 '.join([get_cohort_label(train_coh),
                                       get_cohort_label(trnsf_coh)])

            clr_dict[coh_lbl] = choose_cohort_colour(train_coh)
            base_size = np.mean(pheno_dict[train_coh][base_mtype])
            best_prop = np.mean(pheno_dict[train_coh][best_subtype])
            best_prop /= base_size
            base_size *= 7.7

            plt_x = auc_df.loc[base_mtype, 'mean']
            plt_y = auc_df.loc[best_subtype, 'mean']
            plt_min = min(plt_min, plt_x - 0.07, plt_y - 0.07)

            if plt_x > 0.7 or plt_y > 0.7:
                pnt_dict[plt_x, plt_y] = (base_size ** 0.47, (coh_lbl, ''))
            else:
                pnt_dict[plt_x, plt_y] = (base_size ** 0.47, ('', ''))

            pie_ax = inset_axes(
                ax, width=base_size ** 0.5, height=base_size ** 0.5,
                bbox_to_anchor=(plt_x, plt_y), bbox_transform=ax.transData,
                loc=10, axes_kwargs=dict(aspect='equal'), borderpad=0
                )

            pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                       colors=[clr_dict[coh_lbl] + (0.83, ),
                               clr_dict[coh_lbl] + (0.23, )],
                       wedgeprops=dict(edgecolor='black', linewidth=13 / 11))

    # figure out where to place the labels for each point, and plot them
    lbl_pos = place_labels(pnt_dict, lims=(plt_min, 1 - (1 - plt_min) / 71),
                           lbl_dens=1.04, seed=args.seed)

    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=21, ha=pos[1], va='bottom')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.017 + pnt_dict[pnt_x, pnt_y][0] / (61 * plt_min)):
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / ((1 - plt_min) * 173
                                                   * ln_lngth)
            lbl_gap = (ln_lngth ** -1) / 253

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c='black', linewidth=1.1, alpha=0.61)

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

    ax.set_xlim([plt_min, 1 + (1 - plt_min) / 103])
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 103])

    ax.set_xlabel("Transfer AUC\nusing gene-wide classifier",
                  size=27, weight='semibold')
    ax.set_ylabel("Transfer AUC\nusing best found subgrouping",
                  size=27, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "transfer-comparisons_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots how well the mutation subgroupings of a gene can be predicted "
        "across all tested cohorts for a given source of expression data."
        )

    # create positional command line arguments
    parser.add_argument('expr_source',
                        help='a source of cohort expression data', type=str)
    parser.add_argument('gene', help='a mutated gene', type=str)

    # create argument for seed used to regulate label placement in plots
    parser.add_argument(
        '--seed', default=9401, type=int,
        help="the random seed to use for setting plotting parameters"
        )

    # parse command line arguments, get list of experiments matching given
    # criteria that have run to completion
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__*__samps-*".format(args.expr_source),
            "trnsf-vals__*__*.p.gz"))
        ]

    # parse out input attributes of each experiment
    out_list = pd.DataFrame([
        {'Cohort': out_data[0].split('__')[1],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "trnsf-vals__")[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split(".p.gz")[0]}
        for out_data in out_datas
        ])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for expression "
                         "source `{}` !".format(args.expr_source))

    # filter out experiments that have not tested gene-wide classifiers as
    # well as those that did not test the most subgroupings for a given
    # combination of input attributes
    out_use = out_list.groupby(['Cohort', 'Classif']).filter(
        lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                      and outs.Levels.str.match('Domain_').any())
        ).groupby(['Cohort', 'Levels', 'Classif'])['Samps'].min()

    out_use = out_use[out_use.index.get_level_values('Cohort').isin(
        train_cohorts)]

    # ensure gene-wide classifier input is read in first for each experiment
    out_lvls = set(out_use.index.get_level_values('Levels'))
    out_use = out_use.reindex(['Exon__Location__Protein']
                              + list(out_lvls - {'Exon__Location__Protein'}),
                              level='Levels')

    phn_dict = dict()
    auc_dict = dict()
    trnsf_aucs = dict()
    conf_dict = dict()

    for (coh, lvls, clf), ctf in tuple(out_use.iteritems()):
        out_tag = "{}__{}__samps-{}".format(args.expr_source, coh, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, clf)),
                         'r') as f:
            phns = pickle.load(f)

            phn_vals = {mtype: phn for mtype, phn in phns.items()
                        if select_mtype(mtype, args.gene)}

        if phn_vals:
            if coh in phn_dict:
                phn_dict[coh].update(phn_vals)
            else:
                phn_dict[coh] = phn_vals

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-aucs__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                auc_vals = pickle.load(f)['mean']
                auc_vals = auc_vals[[mtype for mtype in auc_vals.index
                                     if select_mtype(mtype, args.gene)]]

                if (coh, clf) in auc_dict:
                    auc_dict[coh, clf] = pd.concat([auc_dict[coh, clf],
                                                    auc_vals])
                else:
                    auc_dict[coh, clf] = auc_vals

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-trnsf__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                trnsf_data = pickle.load(f)

                for trnsf_coh, trnsf_out in trnsf_data.items():
                    if trnsf_out['AUC']:
                        use_mtypes = trnsf_out['AUC']['mean'].index[[
                            select_mtype(mtype, args.gene)
                            for mtype in trnsf_out['AUC']['mean'].index
                            ]]

                        auc_df = pd.DataFrame.from_dict(
                            trnsf_out['AUC']).loc[use_mtypes]

                        if (clf, coh, trnsf_coh) in trnsf_aucs:
                            trnsf_aucs[clf, coh, trnsf_coh] = pd.concat([
                                trnsf_aucs[clf, coh, trnsf_coh], auc_df])
                        else:
                            trnsf_aucs[clf, coh, trnsf_coh] = auc_df

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-conf__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                conf_vals = pickle.load(f)['mean']
                conf_vals = conf_vals[[mtype for mtype in conf_vals.index
                                       if select_mtype(mtype, args.gene)]]

                if (coh, clf) in conf_dict:
                    conf_dict[coh, clf] = pd.concat([conf_dict[coh, clf],
                                                     conf_vals])
                else:
                    conf_dict[coh, clf] = conf_vals

    if not phn_dict:
        raise ValueError("No experiment output found for "
                         "gene `{}`!".format(args.gene))

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.gene])),
                exist_ok=True)
    plt_clfs = out_use.index.get_level_values('Classif').value_counts()

    for clf in plt_clfs[plt_clfs > 1].index:
        clf_aucs = {coh: auc_data
                    for (coh, out_clf), auc_data in auc_dict.items()
                    if out_clf == clf}
        clf_confs = {coh: conf_data
                     for (coh, out_clf), conf_data in conf_dict.items()
                     if out_clf == clf}

        clf_trnsf = {
            (coh, trnsf_coh): trnsf_data
            for (out_clf, coh, trnsf_coh), trnsf_data in trnsf_aucs.items()
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

