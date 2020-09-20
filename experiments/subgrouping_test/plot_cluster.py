
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'cluster')

from HetMan.experiments.tcga_cluster import *
from HetMan.experiments.subvariant_test import type_file, metabric_dir
from HetMan.features.cohorts.metabric import (
    load_metabric_samps, choose_subtypes)

from HetMan.experiments.subvariant_test.merge_test import merge_cohort_data
from HetMan.experiments.subvariant_test.setup_test import load_cohort
from HetMan.experiments.subvariant_test.utils import get_cohort_label

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_clustering(trans_expr, type_data, type_clrs,
                    cdata, args, pca_comps=(0, 1)):
    fig, ax = plt.subplots(figsize=(9, 8))

    trans_expr = trans_expr[:, np.array(pca_comps)]
    type_stat = np.array([type_data.loc[samp, 'SUBTYPE']
                          if samp in type_data.index else 'Not Available'
                          for samp in cdata.train_data(None)[0].index])

    lgnd_lbls = []
    lgnd_marks = []

    for sub_type in sorted(set(type_stat)):
        type_indx = type_stat == sub_type

        if sub_type == 'Not Available':
            type_clr = '0.53'
        else:
            type_clr = type_clrs[sub_type]

        ax.scatter(trans_expr[type_indx, 0], trans_expr[type_indx, 1],
                   marker='o', s=31, c=[type_clr],
                   alpha=0.27, edgecolor='none')

        lgnd_lbls += ["{} ({})".format(sub_type, np.sum(type_indx))]
        lgnd_marks += [Line2D([], [], marker='o', linestyle='None',
                              markersize=23, alpha=0.43,
                              markerfacecolor=type_clr,
                              markeredgecolor='none')]

    ax.set_xlabel("{} Component {}".format(args.transform, pca_comps[0] + 1),
                  size=17, weight='semibold')
    ax.set_xticklabels([])

    ax.set_ylabel("{} Component {}".format(args.transform, pca_comps[1] + 1),
                  size=17, weight='semibold')
    ax.set_yticklabels([])

    ax.legend(lgnd_marks, lgnd_lbls, bbox_to_anchor=(0.5, -0.05),
              frameon=False, fontsize=21, ncol=3, loc=9, handletextpad=0.3)
    ax.grid(alpha=0.47, linewidth=1.3)

    ax.set_title(' - '.join([get_cohort_label(args.cohort),
                             args.expr_source]),
                 size=23, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, "{}__{}__{}_comps-{}_{}.svg".format(
            args.expr_source, args.cohort, args.transform,
            pca_comps[0], pca_comps[1]
            )),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the clustering of the samples in a given cohort as "
        "performed by an unsupervised learning method."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)

    parser.add_argument('transform', type=str,
                        choices=list(clust_algs.keys()),
                        help='an unsupervised learning method')

    parser.add_argument('--seed', type=int, default=9087)
    args = parser.parse_args()
    np.random.seed(args.seed)

    use_ctf = sorted(
        int(out_file.parts[-2].split('__samps-')[1])
        for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-conf__*__*.p.gz".format(
                args.expr_source, args.cohort)
            )
        )

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-conf__*__*.p.gz".format(
                args.expr_source, args.cohort)
            )
        ]

    if use_ctf:
        out_dir = os.path.join(base_dir, "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, use_ctf[0]))
        cdata = merge_cohort_data(out_dir, use_seed=8713)

    else:
        cdata = load_cohort(args.cohort, args.expr_source)

    if args.expr_source == 'Firehose' or 'toil_' in args.expr_source:
        type_data = pd.read_csv(type_file, sep='\t', index_col=0, comment='#')

        if '_' in cdata.cohort:
            use_cohort = cdata.cohort.split('_')[0]
        else:
            use_cohort = cdata.cohort

        if use_cohort in type_data.DISEASE.values:
            type_data = type_data[type_data.DISEASE == use_cohort]
            type_list = sorted(set(type_data.SUBTYPE.values))

            type_clrs = dict(zip(type_list,
                                 sns.color_palette('bright',
                                                   n_colors=len(type_list))))

        else:
            type_data = pd.DataFrame({'SUBTYPE': 'Not Available'},
                                     index=cdata.get_samples())
            type_clrs = dict()

    elif args.cohort.split('_')[0] == 'METABRIC':
        type_data = pd.DataFrame({'SUBTYPE': 'Other'},
                                 index=cdata.get_samples())

        samp_data = load_metabric_samps(metabric_dir)
        samp_data = samp_data.loc[samp_data.index.isin(cdata.get_samples())]

        type_list = ['Basal', 'Her2', 'LumA', 'LumB']
        for tp in type_list:
            type_data.SUBTYPE[type_data.index.isin(
                choose_subtypes(samp_data, tp))] = tp

        type_clrs = dict(zip(
            type_list, sns.color_palette('bright', n_colors=len(type_list))))

    os.makedirs(plot_dir, exist_ok=True)
    trans_expr = clust_algs[args.transform].fit_transform_coh(cdata)
    plot_clustering(trans_expr.copy(), type_data, type_clrs, cdata, args)


if __name__ == "__main__":
    main()

