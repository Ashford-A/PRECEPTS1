
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'conf')

from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_tour.utils import (
    get_fancy_label, RandomType)
from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.subvariant_tour.plot_aucs import place_labels
from dryadic.features.mutations import MuType

import argparse
from glob import glob
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_auc_comparison(auc_vals, conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10, 10))

    conf_df = pd.DataFrame.from_dict(dict(zip(
        conf_vals.index, conf_vals.applymap(
            lambda vals: np.percentile(vals, q=[0, 25, 50, 75, 100])).iloc[
                :, 0].values)), orient='index')

    assert set(auc_vals.index) == set(conf_df.index)
    conf_df.columns = ['Min', '1Q', 'Med', '3Q', 'Max']

    for mtype in auc_vals.index:
        if isinstance(mtype, RandomType):
            use_clr = '0.41'
        else:
            use_clr = variant_clrs['Point']

        ax.scatter(auc_vals[mtype].mean(), conf_df.loc[mtype, 'Med'],
                   marker='o', s=9, alpha=0.17,
                   facecolor=use_clr, edgecolor='none')

        ax.scatter(auc_vals[mtype].mean(), conf_df.loc[mtype, 'Max'],
                   marker='v', s=17, alpha=0.19,
                   facecolor='none', edgecolor=use_clr)
        ax.scatter(auc_vals[mtype].mean(), conf_df.loc[mtype, 'Min'],
                   marker='^', s=17, alpha=0.19,
                   facecolor='none', edgecolor=use_clr)

    ax.set_xlabel("AUC using all samples", size=23, weight='semibold')
    ax.set_ylabel("down-sampled AUCs", size=23, weight='semibold')
    plt_min = min(ax.get_xlim()[0], ax.get_ylim()[0])

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=0.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=0.9, alpha=0.89)
    ax.plot([plt_min, 0.997], [plt_min, 0.997],
            linewidth=1.1, linestyle='--', color='#550000', alpha=0.31)

    ax.set_xlim([plt_min, 1.01])
    ax.set_ylim([plt_min, 1.01])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "auc-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_sub_comparisons(conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(11, 11))
    np.random.seed(3742)

    conf_list = conf_vals[[not isinstance(mtype, RandomType)
                           for mtype in conf_vals.index]]
    conf_list = conf_list.applymap(
        lambda confs: np.percentile(confs, 25)).iloc[:, 0]

    pnt_dict = dict()
    clr_dict = dict()

    for gene, conf_vec in conf_list.groupby(
            lambda mtype: mtype.get_labels()[0]):
        clr_dict[gene] = hls_to_rgb(
            h=np.random.uniform(size=1)[0], l=0.5, s=0.8)

        if len(conf_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = conf_vec.index.get_loc(base_mtype)

            best_subtype = conf_vec[:base_indx].append(
                conf_vec[(base_indx + 1):]).idxmax()
            best_indx = conf_vec.index.get_loc(best_subtype)

            if conf_vec[best_indx] > 0.6:
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                if conf_vec[best_indx] > 0.7:
                    pnt_dict[conf_vec[base_indx], conf_vec[best_indx]] = (
                        2119 * base_size, (gene,
                                           get_fancy_label(best_subtype))
                        )

                else:
                    pnt_dict[conf_vec[base_indx], conf_vec[best_indx]] = (
                        2119 * base_size, ('', ''))

                pie_size = base_size ** 0.5
                pie_ax = inset_axes(ax, width=pie_size, height=pie_size,
                                    bbox_to_anchor=(conf_vec[base_indx],
                                                    conf_vec[best_indx]),
                                    bbox_transform=ax.transData, loc=10,
                                    axes_kwargs=dict(aspect='equal'),
                                    borderpad=0)

                pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                           colors=[clr_dict[gene] + (0.77,),
                                   clr_dict[gene] + (0.29,)])

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
    ax.set_xlabel("1st quartile of down-sampled AUCs"
                  "\nusing all point mutations", size=21, weight='semibold')
    ax.set_ylabel("1st quartile of down-sampled AUCs"
                  "\nof best found subgrouping", size=21, weight='semibold')

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            linewidth=2.1, linestyle='--', color='#550000', alpha=0.41)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the AUCs for a particular classifier on the mutations "
        "enumerated for a given cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-data__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif)
            )
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-data__')[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby(['Levels'])['Samps'].min()

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    pheno_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, args.cohort, ctf),
                "out-pheno__{}__{}.p.gz".format(lvls, args.classif)
                ), 'r') as f:
            pheno_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, args.cohort, ctf),
                "out-aucs__{}__{}.p.gz".format(lvls, args.classif)
                ), 'r') as f:
            auc_dict[lvls] = pickle.load(f)

        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, args.cohort, ctf),
                "out-conf__{}__{}.p.gz".format(lvls, args.classif)
                ), 'r') as f:
            conf_dict[lvls] = pickle.load(f)

    auc_vals = pd.concat([auc_df['Chrm'] for auc_df in auc_dict.values()])
    conf_vals = pd.concat([conf_df['Chrm'] for conf_df in conf_dict.values()])

    plot_auc_comparison(auc_vals, conf_vals, pheno_dict, args)
    plot_sub_comparisons(conf_vals, pheno_dict, args)


if __name__ == '__main__':
    main()

