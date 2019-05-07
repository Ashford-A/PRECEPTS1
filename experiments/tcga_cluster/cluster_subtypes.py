
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.tcga_cluster import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
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


def plot_clustering(trans_expr, args, cdata, type_data, pca_comps=(0, 1)):
    fig, ax = plt.subplots(figsize=(9, 8))

    trans_expr = trans_expr[:, np.array(pca_comps)]
    type_stat = np.array([type_data.SUBTYPE[type_data.index.get_loc(samp)]
                          if samp in type_data.index else 'Not Available'
                          for samp in cdata.train_data()[0].index])

    type_clrs = sns.color_palette('bright', n_colors=len(set(type_stat)))
    lgnd_lbls = []
    lgnd_marks = []

    for subtype, type_clr in zip(sorted(set(type_stat)), type_clrs):
        type_indx = type_stat == subtype

        ax.scatter(trans_expr[type_indx, 0], trans_expr[type_indx, 1],
                   marker='o', s=31, c=type_clr, alpha=0.27, edgecolor='none')

        lgnd_lbls += ["{} ({})".format(sub_type, np.sum(type_indx))]
        lgnd_marks += [Line2D([], [],
                              marker='o', linestyle='None',
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

    parse_dir = args.out_dir.split(os.path.join('', 'output', ''))
    plot_dir = os.path.join(parse_dir[0], 'plots', 'cluster')
    os.makedirs(plot_dir, exist_ok=True)

    fig.savefig(
        os.path.join(plot_dir, "{}_comps-{}_{}.png".format(
            parse_dir[1], pca_comps[0], pca_comps[1])),
        dpi=400, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the clustering done by an unsupervised learning method on a "
        "TCGA cohort with molecular subtypes highlighted."
        )

    parser.add_argument('out_dir', type=str)
    parser.add_argument('transform', type=str,
                        choices=list(clust_algs.keys()),
                        help='an unsupervised learning method')
    parser.add_argument('--use_seed', type=int, default=1301)

    args = parser.parse_args()
    np.random.seed(args.use_seed)
    cdata = merge_cohort_data(args.out_dir)
    type_data = pd.read_csv(type_file, sep='\t', index_col=0, comment='#')

    if '_' in cdata.cohort:
        use_cohort = cdata.cohort.split('_')[0]
    else:
        use_cohort = cdata.cohort

    if use_cohort not in type_data.DISEASE.values:
        raise ValueError("The source of this cohort ({}) does not "
                         "match those present in the TCGA subtypes "
                         "file!".format(use_cohort))
 
    type_data = type_data[type_data.DISEASE == use_cohort]
    trans_expr = clust_algs[args.transform].fit_transform_coh(cdata)

    plot_clustering(trans_expr.copy(), args, cdata, type_data)


if __name__ == "__main__":
    main()

