
from ..subgrouping_isolate import base_dir
from ..utilities.transformers import OmicPCA, OmicTSNE, OmicUMAP
from ..utilities.data_dirs import vep_cache_dir
from ..utilities.labels import get_cohort_label
from ...features.cohorts.utils import load_cohort, list_cohort_subtypes

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'cluster')

clust_algs = {'PCA': OmicPCA(), 'tSNE': OmicTSNE(), 'UMAP': OmicUMAP()}


def plot_clustering(trans_expr, subt_data, cdata, args, pca_comps=(0, 1)):
    fig, ((type_ax, muts_ax), (lgnd1_ax, lgnd2_ax)) = plt.subplots(
        figsize=(15, 7), nrows=2, ncols=2,
        gridspec_kw=dict(height_ratios=[4.3, 1])
        )

    plt_subts = subt_data.unique()
    subt_clrs = dict(zip(
        plt_subts, sns.color_palette('bright', n_colors=len(plt_subts))))

    trans_expr = trans_expr[:, np.array(pca_comps)]
    train_samps = cdata.train_data(None)[0].index
    subt_stat = np.array([subt_data[samp]
                          if samp in subt_data.index else 'Not Available'
                          for samp in train_samps])

    lgnd_lbls = []
    lgnd_marks = []

    for sub_type in sorted(set(subt_stat)):
        subt_indx = subt_stat == sub_type

        if sub_type == 'Not Available':
            subt_clr = '0.53'
        else:
            subt_clr = subt_clrs[sub_type]

        type_ax.scatter(trans_expr[subt_indx, 0], trans_expr[subt_indx, 1],
                        marker='o', s=31, c=[subt_clr],
                        alpha=0.27, edgecolor='none')

        lgnd_lbls += ["{} ({})".format(sub_type, np.sum(subt_indx))]
        lgnd_marks += [Line2D([], [], marker='o', linestyle='None',
                              markersize=23, alpha=0.43,
                              markerfacecolor=subt_clr,
                              markeredgecolor='none')]

    lgnd1_ax.legend(lgnd_marks, lgnd_lbls, bbox_to_anchor=(0.5, 1),
                    frameon=False, fontsize=19, ncol=2, loc=9,
                    handletextpad=0.17)

    var_data = pd.read_csv(os.path.join(args.temp_path, 'vars.txt'),
                           sep='\t', header=None,
                           names=['Chr', 'Start', 'End',
                                  'Nucleo', 'Strand', 'Sample'])

    mut_counts = np.log10(var_data.Sample.value_counts())
    mut_norm = colors.Normalize(vmin=0, vmax=mut_counts.max())
    mut_cmap = sns.diverging_palette(240, 20, l=53, s=91,
                                     center="dark", as_cmap=True)

    for i, samp in enumerate(train_samps):
        if samp in mut_counts.index:
            muts_ax.scatter(trans_expr[i, 0], trans_expr[i, 1],
                            marker='o', s=31,
                            c=[mut_cmap(mut_norm(mut_counts[samp]))],
                            alpha=0.27, edgecolor='none')

        else:
            muts_ax.scatter(trans_expr[i, 0], trans_expr[i, 1],
                            marker='o', s=34, facecolor='none',
                            edgecolor='black', alpha=0.31)

    clr_ax = lgnd2_ax.inset_axes(bounds=(1 / 13, 0.11, 11 / 13, 0.59),
                                 clip_on=False, in_layout=False)
    clr_bar = ColorbarBase(ax=clr_ax, cmap=mut_cmap, norm=mut_norm,
                           orientation='horizontal', ticklocation='bottom')

    clr_ax.set_title("Mutation Count", size=15, fontweight='bold')
    clr_ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2]))

    tcks_loc = clr_ax.get_xticks().tolist()
    tcks_loc += [mut_counts.max()]
    clr_ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(tcks_loc))
    clr_bar.ax.set_xticklabels([round(10 ** tick) for tick in tcks_loc],
                               size=13)

    for ax in (type_ax, muts_ax):
        ax.grid(alpha=0.31, linewidth=0.91)

        ax.set_xlabel("{} Component {}".format(args.transform,
                                               pca_comps[0] + 1),
                      size=17, weight='semibold')
        ax.set_xticklabels([])

        ax.set_ylabel("{} Component {}".format(args.transform,
                                               pca_comps[1] + 1),
                      size=17, weight='semibold')
        ax.set_yticklabels([])

    lgnd1_ax.axis('off')
    lgnd2_ax.axis('off')

    fig.text(0.5, 1,
             ' - '.join([get_cohort_label(args.cohort), args.expr_source]),
             size=23, weight='bold', ha='center', va='top')

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
        'plot_cluster',
        description="Plots the output of unsupervised learning on a cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour sample -omic dataset")

    parser.add_argument('transform', type=str,
                        choices=list(clust_algs.keys()),
                        help='an unsupervised learning method')

    parser.add_argument('--seed', type=int, default=9087)
    parser.add_argument('--temp_path', type=str, default=os.getcwd())
    args = parser.parse_args()
    np.random.seed(args.seed)

    cdata = load_cohort(args.cohort, args.expr_source, ('Consequence', ),
                        vep_cache_dir, temp_path=args.temp_path)

    trans_expr = clust_algs[args.transform].fit_transform_coh(cdata)
    os.makedirs(plot_dir, exist_ok=True)
    type_dict = list_cohort_subtypes(args.cohort.split('_')[0])

    if type_dict:
        subt_data = pd.concat([pd.Series(subt, index=smps)
                               for subt, smps in type_dict.items()])
    else:
        subt_data = pd.Series({smp: 'Not Available'
                               for smp in cdata.get_samples()})

    plot_clustering(trans_expr.copy(), subt_data, cdata, args)


if __name__ == "__main__":
    main()

