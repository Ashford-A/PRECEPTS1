
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.misc import choose_label_colour
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_fancy_label
from ..utilities.label_placement import place_scatter_labels

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

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'conf')


def plot_auc_comparison(auc_vals, conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10, 10))

    conf_df = pd.DataFrame.from_dict(dict(zip(
        conf_vals.index, conf_vals.apply(
            lambda vals: np.percentile(vals, q=[0, 25, 50, 75, 100])).values
        )), orient='index', columns=['Min', '1Q', 'Med', '3Q', 'Max'])
    assert set(auc_vals.index) == set(conf_df.index)

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
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in conf_vals.index
        ]]

    for gene, conf_vec in conf_list.apply(
            lambda confs: np.percentile(confs, 25)).groupby(
                lambda mtype: tuple(mtype.label_iter())[0]):
        if len(conf_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = conf_vec.index.get_loc(base_mtype)
            best_subtype = conf_vec[:base_indx].append(
                conf_vec[(base_indx + 1):]).idxmax()

            if conf_vec[best_subtype] > 0.7:
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
                         s=47, c=[gene_clr], edgecolor='0.31', alpha=0.93)
        axarr[i].scatter(1, auc_vals[base_mtype],
                         s=47, c=[gene_clr], edgecolor='0.31', alpha=0.41)

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

    fig.text(plt_size ** 0.71 / 97, 1 / 19, "conf.\nscore",
             fontsize=15, weight='semibold', ha='right', va='bottom')

    if 0.463 < ymin < 0.513:
        ymin = 0.453
    for ax in axarr:
        ax.set_ylim([ymin, 1 + (1 - ymin) / 23])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "distr-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_sub_comparisons(conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    clr_dict = dict()
    plt_min = 0.57

    conf_list = conf_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in conf_vals.index
        ]].apply(lambda confs: np.percentile(confs, 25))

    for gene, conf_vec in conf_list.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        if len(conf_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = conf_vec.index.get_loc(base_mtype)
            best_subtype = conf_vec[:base_indx].append(
                conf_vec[(base_indx + 1):]).idxmax()

            if conf_vec[best_subtype] > 0.6:
                auc_tupl = conf_vec[base_mtype], conf_vec[best_subtype]
                clr_dict[auc_tupl] = choose_label_colour(gene)

                base_size = np.mean(pheno_dict[base_mtype])
                plt_size = 0.07 * base_size ** 0.5
                plot_dict[auc_tupl] = [plt_size, ('', '')]
                plt_min = min(plt_min, conf_vec[base_indx] - 0.053,
                              conf_vec[best_subtype] - 0.029)

                best_prop = np.mean(pheno_dict[best_subtype]) / base_size
                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                if conf_sc > 0.8:
                    plot_dict[auc_tupl][1] = gene, get_fancy_label(
                        tuple(best_subtype.subtype_iter())[0][1],
                        pnt_link='\n', phrase_link=' '
                        )

                elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                    plot_dict[auc_tupl][1] = gene, ''

                auc_bbox = (auc_tupl[0] - plt_size / 2,
                            auc_tupl[1] - plt_size / 2, plt_size, plt_size)

                pie_ax = inset_axes(
                    ax, width='100%', height='100%',
                    bbox_to_anchor=auc_bbox, bbox_transform=ax.transData,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop],
                           colors=[clr_dict[auc_tupl] + (0.77,),
                                   clr_dict[auc_tupl] + (0.29,)],
                           explode=[0.29, 0], startangle=90)

    plt_lims = plt_min, 1 + (1 - plt_min) / 61
    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("1st quartile of down-sampled AUCs"
                  "\nusing all point mutations", size=21, weight='semibold')
    ax.set_ylabel("1st quartile of down-sampled AUCs"
                  "\nof best found subgrouping", size=21, weight='semibold')

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, clr_dict, fig, ax,
                                       plt_lims=[plt_lims, plt_lims])

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_conf',
        description="Plots comparisons of down-sampled AUCs in a cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

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
            conf_dict[lvls] = pickle.load(f)

    auc_vals = pd.concat(auc_dict.values())
    conf_vals = pd.concat(conf_dict.values())

    plot_auc_comparison(auc_vals, conf_vals, phn_dict, args)
    plot_distr_comparisons(auc_vals, conf_vals, phn_dict, args)
    plot_sub_comparisons(conf_vals, phn_dict, args)


if __name__ == '__main__':
    main()

