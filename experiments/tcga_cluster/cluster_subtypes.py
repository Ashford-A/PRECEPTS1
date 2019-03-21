
import os
import sys

if 'DATADIR' in os.environ:
    data_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'variant_baseline')
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'tcga_cluster')

else:
    data_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'subtypes')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.tcga_cluster import *
from HetMan.experiments.variant_baseline.fit_tests import load_cohort_data

import argparse
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


def plot_clustering(trans_expr, args, cdata, lum_data, pca_comps=(0, 1)):
    fig, ax = plt.subplots(figsize=(9, 8))

    trans_expr = trans_expr[:, np.array(pca_comps)]
    lum_stat = np.array([lum_data.SUBTYPE[lum_data.index.get_loc(samp)]
                         if samp in lum_data.index else 'None'
                         for samp in cdata.train_data()[0].index])

    lum_clrs = sns.color_palette('bright', n_colors=len(set(lum_stat)))
    lgnd_lbls = []
    lgnd_marks = []

    for lum, lum_clr in zip(sorted(set(lum_stat)), lum_clrs):
        lum_indx = lum_stat == lum

        ax.scatter(trans_expr[lum_indx, 0], trans_expr[lum_indx, 1],
                   marker='o', s=31, c=lum_clr, alpha=0.27, edgecolor='none')

        lgnd_lbls += ["{} ({})".format(lum, np.sum(lum_indx))]
        lgnd_marks += [Line2D([], [],
                              marker='o', linestyle='None',
                              markersize=23, alpha=0.43,
                              markerfacecolor=lum_clr,
                              markeredgecolor='none')]

    ax.set_xlabel("{} Component {}".format(args.transform, pca_comps[0] + 1),
                  size=17, weight='semibold')
    ax.set_xticklabels([])

    ax.set_ylabel("{} Component {}".format(args.transform, pca_comps[1] + 1),
                  size=17, weight='semibold')
    ax.set_yticklabels([])

    ax.legend(lgnd_marks, lgnd_lbls, bbox_to_anchor=(0.5, -0.05),
              frameon=False, fontsize=21, ncol=3, loc=9, handletextpad=0.3)

    fig.savefig(os.path.join(
        plot_dir, "{}__{}_{}_comps-{}_{}.png".format(
            args.expr_source, args.cohort, args.transform,
            pca_comps[0], pca_comps[1]
            )
        ),
        dpi=400, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the clustering done by an unsupervised learning method on a "
        "TCGA cohort with molecular subtypes highlighted."
        )

    parser.add_argument('expr_source', type=str,
                        choices=list(expr_sources.keys()),
                        help="which TCGA expression data source to use")
    parser.add_argument('cohort', type=str, help='a cohort in TCGA')

    parser.add_argument('transform', type=str,
                        choices=list(clust_algs.keys()),
                        help='an unsupervised learning method')

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(use_seed)

    cdata = load_cohort_data(data_dir, args.expr_source, args.cohort,
                             samp_cutoff=25)
    trans_expr = clust_algs[args.transform].fit_transform_coh(cdata)

    lum_data = pd.read_csv(type_file, sep='\t', index_col=0)
    lum_data = lum_data[lum_data.DISEASE == args.cohort.split('_')[0]]

    plot_clustering(trans_expr.copy(), args, cdata, lum_data)


if __name__ == "__main__":
    main()

