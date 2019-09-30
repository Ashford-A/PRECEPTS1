
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from HetMan.experiments.subvariant_tour.utils import (
    get_fancy_label, RandomType)
from HetMan.experiments.subvariant_tour.plot_aucs import place_labels
from dryadic.features.mutations import MuType

import argparse
import glob as glob
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from colorsys import hls_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# make plots cleaner by turning off outer box, make background all white
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_sub_comparisons(auc_dict, pheno_dict, use_clf, args):
    fig, ax = plt.subplots(figsize=(11, 11))
    np.random.seed(3742)

    auc_vals = dict()
    pnt_dict = dict()
    clr_dict = dict()
    base_mtype = MuType({('Gene', args.gene): pnt_mtype})

    for (coh, lvls, clf), auc_list in auc_dict.items():
        if clf == use_clf:
            if coh in auc_vals:
                auc_vals[coh] = pd.concat([auc_vals[coh], auc_list])
            else:
                auc_vals[coh] = auc_list

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for coh, auc_vec in auc_vals.items():
        clr_dict[coh] = hls_to_rgb(
            h=np.random.uniform(size=1)[0], l=0.5, s=0.8)

        # if there were subgroupings tested for the gene, find the results
        # for the mutation representing all point mutations for this gene...
        if len(auc_vec) > 1 and base_mtype in auc_vec.index:
            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            best_indx = auc_vec.index.get_loc(best_subtype)
            base_size = np.mean(pheno_dict[coh][base_mtype])
            best_prop = np.mean(pheno_dict[coh][best_subtype]) / base_size

            # ...and if it is really good then add a label with the gene
            # name and a description of the best found subgrouping
            if auc_vec[best_indx] > 0.6:
                pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                    2119 * base_size,
                    (coh, get_fancy_label(best_subtype, max_subs=3))
                    )

            else:
                pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                    2119 * base_size, ('', ''))

            pie_size = base_size ** 0.5
            pie_ax = inset_axes(ax, width=pie_size, height=pie_size,
                                bbox_to_anchor=(auc_vec[base_indx],
                                                auc_vec[best_indx]),
                                bbox_transform=ax.transData, loc=10,
                                axes_kwargs=dict(aspect='equal'),
                                borderpad=0)

            pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                       colors=[clr_dict[coh] + (0.77,),
                               clr_dict[coh] + (0.29,)])

    lbl_pos = place_labels(pnt_dict)
    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=13, ha=pos[1], va='bottom')
        ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][1],
                size=9, ha=pos[1], va='top')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        pnt_sz = (pnt_dict[pnt_x, pnt_y][0] ** 0.43) / 1077
        if ln_lngth > 0.01 + pnt_sz:
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            lbl_sz = pnt_dict[pnt_x, pnt_y][1][1].count('\n')

            pnt_gap = pnt_sz / ln_lngth
            lbl_gap = (0.02 + (1 / 117) * lbl_sz ** 0.17) / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta],
                    c=use_clr, linewidth=2.7, alpha=0.27)

    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])
    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found subgrouping", size=23, weight='semibold')

    ax.plot([0.48, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [0.48, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.gene]),
                     "sub-comparisons_{}.svg".format(use_clf)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots how well the mutation subgroupings of a gene can be predicted "
        "across all tested cohorts for a given source of expression data."
        )

    parser.add_argument('expr_source', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.gene])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__*__samps-*/out-data__*__*.p.gz".format(args.expr_source))
        ]

    out_use = pd.DataFrame([
        {'Cohort': out_data[0].split('__')[1],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "out-data__")[1].split('__')[:-1]),
         'Classif': out_data[1].split('__')[-1].split(".p.gz")[0]}
        for out_data in out_datas
        ]).groupby(['Cohort', 'Classif']).filter(
            lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                          and outs.Levels.str.match('Domain_').any())
        ).groupby(['Cohort', 'Levels', 'Classif'])['Samps'].min()

    cdata_dict = {
        coh: merge_cohort_data(
            os.path.join(base_dir, "{}__{}__samps-{}".format(
                args.expr_source, coh, ctf)),
            use_seed=8713
            )
        for coh, ctf in out_use.groupby('Cohort').min().iteritems()
        }

    use_cohs = {
        coh for coh, ctf in out_use.groupby('Cohort').min().iteritems()
        if ((cdata_dict[coh].muts.Gene[cdata_dict[coh].muts.Scale
                                       == 'Point'] == args.gene).any()
            and len(tuple(cdata_dict[coh].mtrees.values())[0][
                args.gene]['Point']) >= ctf)
        }

    out_use = out_use[out_use.index.get_level_values('Cohort').isin(use_cohs)]
    infer_dict = dict()
    phn_dict = dict()
    auc_dict = dict()

    for (coh, lvls, clf), ctf in tuple(out_use.iteritems()):
        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, coh, ctf),
                "out-data__{}__{}.p.gz".format(lvls, clf)
                ), 'r') as f:
            infer_df = pickle.load(f)['Infer']['Chrm']

            infer_dict[coh, lvls, clf] = infer_df.loc[[
                mtype for mtype in infer_df.index
                if not isinstance(mtype, RandomType)
                and mtype.get_labels()[0] == args.gene
                ]].applymap(np.mean)

        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, coh, ctf),
                "out-pheno__{}__{}.p.gz".format(lvls, clf)
                ), 'r') as f:
            phns = pickle.load(f)

            phn_vals = {mtype: phn for mtype, phn in phns.items()
                        if not isinstance(mtype, RandomType)
                        and mtype.get_labels()[0] == args.gene}

            if coh in phn_dict:
                phn_dict[coh].update(phn_vals)
            else:
                phn_dict[coh] = phn_vals

        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, coh, ctf),
                "out-aucs__{}__{}.p.gz".format(lvls, clf)
                ), 'r') as f:
            auc_vals = pickle.load(f).Chrm

            auc_dict[coh, lvls, clf] = auc_vals[[
                mtype for mtype in auc_vals.index
                if not isinstance(mtype, RandomType)
                and mtype.get_labels()[0] == args.gene
                ]]

    plt_clfs = out_use.index.get_level_values('Classif').value_counts()

    for clf in plt_clfs[plt_clfs > 1].index:
        plot_sub_comparisons(auc_dict, phn_dict, clf, args)


if __name__ == '__main__':
    main()

