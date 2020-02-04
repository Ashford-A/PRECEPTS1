
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'conf')

from HetMan.experiments.subvariant_test import pnt_mtype, copy_mtype
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.utils import (
    get_fancy_label, choose_label_colour)
from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.subvariant_tour.plot_aucs import place_labels
from dryadic.features.mutations import MuType

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

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_auc_comparison(auc_vals, conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10, 10))

    conf_df = pd.DataFrame.from_dict(dict(zip(
        conf_vals.index, conf_vals.apply(
            lambda vals: np.percentile(vals, q=[0, 25, 50, 75, 100])).values
        )), orient='index')

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
                     "auc-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_distr_comparisons(auc_vals, conf_vals, pheno_dict, args):
    gene_dict = dict()

    conf_list = conf_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    for gene, conf_vec in conf_list.apply(
            lambda confs: np.percentile(confs, 25)).groupby(
                lambda mtype: mtype.get_labels()[0]):

        if len(conf_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = conf_vec.index.get_loc(base_mtype)

            best_subtype = conf_vec[:base_indx].append(
                conf_vec[(base_indx + 1):]).idxmax()
            best_indx = conf_vec.index.get_loc(best_subtype)

            if conf_vec[best_indx] > 0.7:
                gene_dict[gene] = (
                    choose_label_colour(gene), base_mtype, best_subtype,
                    np.greater.outer(conf_list[best_subtype],
                                     conf_list[base_mtype]).mean()
                    )

    plt_size = min(len(gene_dict), 12)
    ymin = 0.47
    fig, axarr = plt.subplots(figsize=(0.5 + 1.5 * plt_size, 7),
                              nrows=1, ncols=plt_size, sharey=True)

    for i, (gene, (gene_clr, base_mtype, best_subtype, conf_sc)) in enumerate(
            sorted(gene_dict.items(),
                   key=lambda x: auc_vals[x[1][2]], reverse=True)[:plt_size]
            ):
        axarr[i].set_title(gene, size=21, weight='semibold')

        plt_df = pd.concat([
            pd.DataFrame({'Type': 'Base', 'Conf': conf_list[base_mtype]}),
            pd.DataFrame({'Type': 'Subg', 'Conf': conf_list[best_subtype]})
            ])

        sns.violinplot(x=plt_df.Type, y=plt_df.Conf, ax=axarr[i],
                       order=['Subg', 'Base'], palette=[gene_clr, gene_clr],
                       cut=0, linewidth=0, width=0.93)

        axarr[i].scatter(0, auc_vals[best_subtype], 
                         s=41, c=[gene_clr], edgecolor='0.23', alpha=0.97)
        axarr[i].scatter(1, auc_vals[base_mtype],
                         s=41, c=[gene_clr], edgecolor='0.23', alpha=0.53)

        axarr[i].get_children()[0].set_alpha(0.71)
        axarr[i].get_children()[2].set_alpha(0.29)

        if conf_sc == 1:
            conf_lbl = "1"
        elif 0.9995 < conf_sc < 1:
            conf_lbl = ">0.999"
        else:
            conf_lbl = "{:.3f}".format(conf_sc)

        axarr[i].text(0.5, 1 / 97, conf_lbl, size=17,
                      ha='center', va='bottom', transform=axarr[i].transAxes)

        axarr[i].plot([-0.5, 1.5], [0.5, 0.5],
                      color='black', linewidth=2.3, linestyle=':', alpha=0.83)
        axarr[i].plot([-0.5, 1.5], [1, 1],
                      color='black', linewidth=1.7, alpha=0.83)

        axarr[i].set_xlabel('')
        axarr[i].set_xticklabels([])
        ymin = min(ymin, min(conf_list[base_mtype]) - 0.04,
                   min(conf_list[best_subtype]) - 0.04)

        if i == 0:
            axarr[i].set_ylabel('AUCs', size=21, weight='semibold')
        else:
            axarr[i].set_ylabel('')

    if 0.47 < ymin < 0.51:
        ymin = 0.445
    for ax in axarr:
        ax.set_ylim([ymin, 1 + (1 - ymin) / 23])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "distr-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_sub_comparisons(conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    pnt_dict = dict()
    clr_dict = dict()

    conf_list = conf_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
        for mtype in conf_vals.index
        ]]

    conf_list = conf_list.apply(lambda confs: np.percentile(confs, 25))
    for gene, conf_vec in conf_list.groupby(
            lambda mtype: mtype.get_labels()[0]):

        if len(conf_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = conf_vec.index.get_loc(base_mtype)

            best_subtype = conf_vec[:base_indx].append(
                conf_vec[(base_indx + 1):]).idxmax()
            best_indx = conf_vec.index.get_loc(best_subtype)

            if conf_vec[best_indx] > 0.6:
                clr_dict[gene] = choose_label_colour(gene)
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                conf_sc = np.greater.outer(conf_list[best_subtype],
                                           conf_list[base_mtype]).mean()

                if conf_sc > 0.8:
                    mtype_lbl = '\n'.join(
                        get_fancy_label(best_subtype).split('\n')[1:])

                    pnt_dict[conf_vec[base_indx], conf_vec[best_indx]] = (
                        base_size ** 0.53, (gene, mtype_lbl))

                elif conf_vec[base_indx] > 0.7 or conf_vec[best_indx] > 0.7:
                    pnt_dict[conf_vec[base_indx], conf_vec[best_indx]] = (
                        base_size ** 0.53, (gene, ''))

                else:
                    pnt_dict[conf_vec[base_indx], conf_vec[best_indx]] = (
                        base_size ** 0.53, ('', ''))

                pie_ax = inset_axes(
                    ax, width=base_size ** 0.5, height=base_size ** 0.5,
                    bbox_to_anchor=(conf_vec[base_indx], conf_vec[best_indx]),
                    bbox_transform=ax.transData, loc=10,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

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
        if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            linewidth=2.1, linestyle='--', color='#550000', alpha=0.41)

    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])

    ax.set_xlabel("1st quartile of down-sampled AUCs"
                  "\nusing all point mutations", size=21, weight='semibold')
    ax.set_ylabel("1st quartile of down-sampled AUCs"
                  "\nof best found subgrouping", size=21, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the down-sampled AUCs for a particular classifier on the "
        "mutations enumerated for a given cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-conf__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif)
            )
        ]

    out_list = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                              'Levels': '__'.join(out_data[1].split(
                                  'out-conf__')[1].split('__')[:-1])}
                             for out_data in out_datas])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby('Levels')['Samps'].min()
    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    # create directory where plots will be stored
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

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
            conf_dict[lvls] = pickle.load(f)['mean']

    auc_vals = pd.concat(auc_dict.values())
    conf_vals = pd.concat(conf_dict.values())

    plot_auc_comparison(auc_vals, conf_vals, phn_dict, args)
    plot_distr_comparisons(auc_vals, conf_vals, phn_dict, args)
    plot_sub_comparisons(conf_vals, phn_dict, args)


if __name__ == '__main__':
    main()

