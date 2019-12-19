
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, train_cohorts)
from HetMan.experiments.subvariant_tour.utils import RandomType
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_test.utils import get_fancy_label
from HetMan.experiments.subvariant_test.plot_aucs import place_labels
from HetMan.experiments.subvariant_test.plot_copy import select_mtype
from HetMan.experiments.utilities.pcawg_colours import cohort_clrs

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


def get_cohort_label(coh):
    if '_' in coh:
        coh_lbl = "{}({})".format(*coh.split('_'))
        coh_lbl = coh_lbl.replace("IDHmut-non-codel", "IDHmut-nc")

    else:
        coh_lbl = str(coh)

    return coh_lbl


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


def plot_sub_comparisons(auc_dict, conf_dict, pheno_dict, use_clf, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    plt_min = 0.89
    pnt_dict = dict()
    auc_vals = dict()
    conf_vals = dict()

    # filter out results not returned by the given classifier
    for (coh, lvls, clf), auc_list in auc_dict.items():
        if clf == use_clf:
            use_mtypes = {
                mtype for mtype in auc_list.index
                if (not isinstance(mtype, RandomType)
                    and (mtype.subtype_list()[0][1] & copy_mtype).is_empty())
                }

            if coh in auc_vals:
                auc_vals[coh] = pd.concat([
                    auc_vals[coh], auc_list[use_mtypes]])
                conf_vals[coh] = pd.concat([
                    conf_vals[coh], conf_dict[coh, lvls, clf][use_mtypes]])

            else:
                auc_vals[coh] = auc_list[use_mtypes]
                conf_vals[coh] = conf_dict[coh, lvls, clf][use_mtypes]

    # for each cohort, check if the given gene had subgroupings that were
    # tested, and get the results for all the gene's point mutations...
    for coh, auc_vec in auc_vals.items():
        if len(auc_vec) > 1 and base_mtype in auc_vec.index:
            base_indx = auc_vec.index.get_loc(base_mtype)

            # ...and those for the subgrouping in the cohort with the best AUC
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            plt_min = min(plt_min, auc_vec[base_indx] - 0.07,
                          auc_vec[best_indx] - 0.07)
            base_size = np.mean(pheno_dict[coh][base_mtype])
            best_prop = np.mean(pheno_dict[coh][best_subtype]) / base_size

            conf_sc = np.greater.outer(conf_vals[coh][best_subtype],
                                       conf_vals[coh][base_mtype]).mean()

            if conf_sc > 0.9:
                pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                    base_size ** 0.53, (coh, get_fancy_label(best_subtype)))

            else:
                pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                    base_size ** 0.53, (coh, ''))

            # create the axis in which the pie chart will be plotted
            pie_ax = inset_axes(
                ax, width=base_size ** 0.5, height=base_size ** 0.5,
                bbox_to_anchor=(auc_vec[base_indx], auc_vec[best_indx]),
                bbox_transform=ax.transData, loc=10,
                axes_kwargs=dict(aspect='equal'), borderpad=0
                )

            # plot the pie chart for the AUCs of the gene in this cohort
            pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                       colors=[choose_cohort_colour(coh) + (0.83, ),
                               choose_cohort_colour(coh) + (0.23, )],
                       wedgeprops=dict(edgecolor='black', linewidth=10 / 11))

    # figure out where to place the annotation labels for each cohort so that
    # they don't overlap with one another or the pie charts
    lbl_pos = place_labels(pnt_dict,
                           lims=(plt_min + 0.03, 1 - (1 - plt_min) / 71),
                           lbl_dens=0.59, seed=args.seed)

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
        min_lngth = (1 - plt_min) * (0.02 + pnt_dict[pnt_x, pnt_y][0] / 13)
        if ln_lngth > min_lngth:
            use_clr = choose_cohort_colour(pnt_dict[pnt_x, pnt_y][1][0])

            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (100 - 53 * plt_min)
            pnt_gap /= ln_lngth
            lbl_gap = (1 - plt_min) / (ln_lngth * 97)

            # ...create a line connecting the pie chart to the label
            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     (pos[0][1] + lbl_gap * y_delta + lbl_gap * 0.015
                      + lbl_gap * 0.005 * np.sign(y_delta))],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.plot([plt_min, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [plt_min, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([plt_min + 0.01, 0.999], [plt_min + 0.01, 0.999],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim([plt_min, 1 + (1 - plt_min) / 71])
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 71])

    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found subgrouping", size=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "sub-comparisons_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_conf_distributions(auc_dict, conf_dict, pheno_dict, use_clf, args):
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    coh_dict = dict()
    auc_vals = dict()
    conf_vals = dict()

    # filter out results not returned by the given classifier
    for (coh, lvls, clf), auc_list in auc_dict.items():
        if clf == use_clf:
            use_mtypes = {
                mtype for mtype in auc_list.index
                if (not isinstance(mtype, RandomType)
                    and (mtype.subtype_list()[0][1] & copy_mtype).is_empty())
                }

            if coh in auc_vals:
                auc_vals[coh] = pd.concat([
                    auc_vals[coh], auc_list[use_mtypes]])
                conf_vals[coh] = pd.concat([
                    conf_vals[coh], conf_dict[coh, lvls, clf][use_mtypes]])

            else:
                auc_vals[coh] = auc_list[use_mtypes]
                conf_vals[coh] = conf_dict[coh, lvls, clf][use_mtypes]

    for coh, conf_vec in conf_vals.items():
        if len(conf_vec) > 1 and base_mtype in conf_vec.index:
            conf_list = conf_vec.apply(lambda confs: np.percentile(confs, 25))

            base_indx = conf_list.index.get_loc(base_mtype)
            best_subtype = conf_list[:base_indx].append(
                conf_list[(base_indx + 1):]).idxmax()
            best_indx = conf_list.index.get_loc(best_subtype)

            if conf_list[best_indx] > 0.6:
                coh_dict[coh] = (
                    choose_cohort_colour(coh), best_subtype,
                    np.greater.outer(conf_vec[best_subtype],
                                     conf_vec[base_mtype]).mean()
                    )

    ymin = 0.83
    fig, axarr = plt.subplots(figsize=(0.5 + 1.5 * len(coh_dict), 7),
                              nrows=1, ncols=len(coh_dict), sharey=True,
                              squeeze=False)

    for i, (coh, (coh_clr, best_subtype, conf_sc)) in enumerate(
            sorted(coh_dict.items(),
                   key=lambda x: auc_vals[x[0]][x[1][1]], reverse=True)
            ):
        coh_lbl = get_cohort_label(coh).replace('(', '\n(')

        plt_df = pd.concat([
            pd.DataFrame({'Type': 'Base',
                          'Conf': conf_vals[coh][base_mtype]}),
            pd.DataFrame({'Type': 'Subg',
                          'Conf': conf_vals[coh][best_subtype]})
            ])

        sns.violinplot(x=plt_df.Type, y=plt_df.Conf, ax=axarr[0, i],
                       order=['Subg', 'Base'], palette=[coh_clr, coh_clr],
                       cut=0, linewidth=0.7, width=0.93, inner=None)

        axarr[0, i].scatter(0, auc_vals[coh][best_subtype], 
                         s=41, c=[coh_clr], edgecolor='0.23', alpha=0.97)
        axarr[0, i].scatter(1, auc_vals[coh][base_mtype],
                            s=41, c=[coh_clr], edgecolor='0.23', alpha=0.53)

        axarr[0, i].set_title(coh_lbl, size=19, weight='semibold')
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
        ymin = min(ymin, min(conf_vals[coh][base_mtype]) - 0.04,
                   min(conf_vals[coh][best_subtype]) - 0.04)

        if i == 0:
            axarr[0, i].set_ylabel('AUCs', size=21, weight='semibold')
        else:
            axarr[0, i].set_ylabel('')

    if 0.47 < ymin < 0.51:
        ymin = 0.445
    for i in range(len(coh_dict)):
        axarr[0, i].set_ylim([ymin, 1 + (1 - ymin) / 31])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "conf-distributions_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots how well the mutation subgroupings of a gene can be predicted "
        "across all tested cohorts for a given source of expression data."
        )

    parser.add_argument('expr_source',
                        help='a source of cohort expression data', type=str)
    parser.add_argument('gene', help='a mutated gene', type=str)

    parser.add_argument(
        '--seed', default=9401, type=int,
        help="the random seed to use for setting plotting parameters"
        )

    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__*__samps-*".format(args.expr_source),
            "trnsf-vals__*__*.p.gz"))
        ]

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

    out_use = out_list.groupby(['Cohort', 'Classif']).filter(
        lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                      and outs.Levels.str.match('Domain_').any())
        ).groupby(['Cohort', 'Levels', 'Classif'])['Samps'].min()

    out_use = out_use[out_use.index.get_level_values('Cohort').isin(
        train_cohorts)]

    out_lvls = set(out_use.index.get_level_values('Levels'))
    out_use = out_use.reindex(['Exon__Location__Protein']
                              + list(out_lvls - {'Exon__Location__Protein'}),
                              level='Levels')

    phn_dict = dict()
    auc_dict = dict()
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

                auc_dict[coh, lvls, clf] = auc_vals[[
                    mtype for mtype in auc_vals.index
                    if select_mtype(mtype, args.gene)
                    ]]

            with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                          "out-conf__{}__{}.p.gz".format(
                                              lvls, clf)),
                             'r') as f:
                conf_vals = pickle.load(f)['mean']

                conf_dict[coh, lvls, clf] = conf_vals[[
                    mtype for mtype in conf_vals.index
                    if select_mtype(mtype, args.gene)
                    ]]

    if not phn_dict:
        raise ValueError("No experiment output found for "
                         "gene `{}`!".format(args.gene))

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.gene])),
                exist_ok=True)

    plt_clfs = out_use.index.get_level_values('Classif').value_counts()
    for clf in plt_clfs[plt_clfs > 1].index:
        plot_sub_comparisons(auc_dict, conf_dict, phn_dict, clf, args)
        plot_conf_distributions(auc_dict, conf_dict, phn_dict, clf, args)


if __name__ == '__main__':
    main()

