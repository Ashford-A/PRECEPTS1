
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.tcga_cluster import *
import argparse
import dill as pickle
import numpy as np
import pandas as pd
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_copy_count(trans_expr, args, cdata, comps=3):
    fig_size = 1 + comps * 2.9
    pnt_size = trans_expr.shape[0] ** -0.19

    fig, axarr = plt.subplots(figsize=(fig_size, fig_size * 19/17),
                              nrows=comps + 1, ncols=comps,
                              gridspec_kw=dict(height_ratios=[
                                  27 * fig_size ** -0.97] * comps + [1]))

    copy_df = pd.DataFrame(
        0, index=trans_expr.index, columns=['Gains', 'Dels'])

    for gene, muts in cdata.train_mut:
        if 'Copy' in dict(muts):
            if 'DeepGain' in dict(muts['Copy']):
                copy_df['Gains'] += copy_df.index.isin(
                    muts['Copy']['DeepGain']).astype(int)

            if 'DeepDel' in dict(muts['Copy']):
                copy_df['Dels'] += copy_df.index.isin(
                    muts['Copy']['DeepDel']).astype(int)

    gain_max = copy_df['Gains'].max()
    gain_cmap = cmx.ScalarMappable(
        norm=colors.Normalize(vmin=0, vmax=gain_max),
        cmap=sns.cubehelix_palette(start=1.91, rot=0, hue=1,
                                   dark=0, light=1, as_cmap=True)
        ).to_rgba

    loss_max = copy_df['Dels'].max()
    loss_cmap = cmx.ScalarMappable(
        norm=colors.Normalize(vmin=0, vmax=loss_max),
        cmap=sns.cubehelix_palette(start=0.77, rot=0, hue=1,
                                   dark=0, light=1, as_cmap=True)
        ).to_rgba

    for i, j in combn(range(comps), 2):
        for samp, (gain_count, loss_count) in copy_df.iterrows():
            axarr[i, j].plot(trans_expr.loc[samp, i], trans_expr.loc[samp, j],
                             marker='o', markersize=pnt_size * 12,
                             color=gain_cmap(gain_count), alpha=0.53,
                             markeredgecolor='0.3', markeredgewidth=0.3)

            axarr[j, i].plot(trans_expr.loc[samp, i], trans_expr.loc[samp, j],
                             marker='o', markersize=pnt_size * 12,
                             color=loss_cmap(loss_count), alpha=0.53,
                             markeredgecolor='0.3', markeredgewidth=0.3)

    for i in range(comps):
        axarr[i, i].axis('off')
        axarr[i, i].text(0.5, 0.5, "Component {}".format(i + 1),
                         size=fig_size * 2.07, fontweight='semibold',
                         ha='center', va='center')

        axarr[-1, i].axis('off')
        axarr[-1, i].set_xlim(0, 1)
        axarr[-1, i].set_ylim(0, 1)

    for ax in axarr.flatten():
        ax.grid(alpha=0.41, linewidth=1.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.text(0.5, 1/14, "{} Genes\nConsidered".format(len(cdata.mut_genes)),
             size=fig_size * 2.19, ha='center', va='center')

    for i in [0, 0.5, 1]:
        axarr[-1, 0].plot(12/11, 0.97 - 0.81 * i, marker='o',
                          markersize=pnt_size * 63, clip_on=False,
                          color=loss_cmap(loss_max * i),
                          alpha=0.71, markeredgecolor='0.2')

        axarr[-1, 0].text(1, 0.95 - 0.81 * i,
                          "{:.0f} Deleted Genes".format(loss_max * i),
                          size=fig_size * 1.83, ha='right', va='center',
                          transform=axarr[-1, 0].transData)

        axarr[-1, -1].plot(-1/11, 0.97 - 0.81 * i, marker='o', clip_on=False,
                           markersize=pnt_size * 63,
                           color=gain_cmap(gain_max * i),
                           alpha=0.71, markeredgecolor='0.2')

        axarr[-1, -1].text(0, 0.95 - 0.81 * i,
                           "{:.0f} Amplified Genes".format(gain_max * i),
                           size=fig_size * 1.83, ha='left', va='center',
                           transform=axarr[-1, -1].transAxes)

    parse_dir = args.cdata_file.split(os.path.join('', 'HetMan', ''))
    dir_parts = parse_dir[1].split(os.sep)
    dir_parts[-1] = dir_parts[-1].split('.p')[0]

    plot_dir = os.path.join(parse_dir[0], 'HetMan', dir_parts[0],
                            'plots', 'cluster')
    os.makedirs(plot_dir, exist_ok=True)

    if dir_parts[1] == 'output':
        plot_lbl = "{}__copy-count.svg".format('__'.join(dir_parts[2:]))
    else:
        plot_lbl = "{}__copy-count.svg".format('__'.join(dir_parts[1:]))

    fig.tight_layout(w_pad=7/3, h_pad=7/3)
    fig.savefig(os.path.join(plot_dir, plot_lbl),
                dpi=500, bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots an unsupervised clustering of the expression of a TCGA cohort "
        "with the samples highlighted according to the number of genes with "
        "copy number alterations."
        )

    parser.add_argument('cdata_file', type=str)
    parser.add_argument('transform', type=str,
                        choices=list(clust_algs.keys()),
                        help='an unsupervised learning method')
    parser.add_argument('--use_seed', type=int, default=5009)

    args = parser.parse_args()
    np.random.seed(args.use_seed)

    with open(args.cdata_file, 'rb') as cdata_fl:
        cdata = pickle.load(cdata_fl)

    use_alg = clust_algs[args.transform]
    use_alg.set_params(**{'fit__n_components': 3, 'fit__n_neighbors': 20})
    trans_expr = pd.DataFrame(use_alg.fit_transform_coh(cdata),
                              index=sorted(cdata.train_samps))

    plot_copy_count(trans_expr.copy(), args, cdata)


if __name__ == "__main__":
    main()

