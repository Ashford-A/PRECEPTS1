
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'copy_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'cohort')

from HetMan.experiments.copy_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.utilities.colour_maps import cor_cmap
from HetMan.experiments.utilities.scatter_plotting import place_annot

import numpy as np
import pandas as pd

import argparse
from pathlib import Path
import dill as pickle
from functools import reduce
from operator import and_
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
from matplotlib import ticker

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
use_marks = [(0, 3, 0)]
use_marks += [(i, 0, k) for k in (0, 140) for i in (3, 4, 5)]


def plot_cor_highlights(out_dict, args, cdata_dict):
    """Plots the accuracy of each regressor for the top genes."""

    # calculates the first quartile of the testing correlation of each
    # regressor on each gene's CNAs across the cross-validation runs
    cor_quarts = pd.DataFrame.from_dict({
        mdl: out_data['Fit']['test'].Cor.quantile(q=0.25, axis=1)
        for mdl, out_data in out_dict.items()
        })

    # gets the top forty genes by the best first-quartile correlation across
    # all regressors, gets the regressors that did well on at least one type
    use_mtypes = cor_quarts.max(axis=1).sort_values()[-40:].index
    use_models = cor_quarts.loc[use_mtypes, :].max() > 0.2
    plot_df = cor_quarts.loc[use_mtypes, use_models]

    # set the size of the plot and the base label size based on the number
    # of data points that will be shown in the heatmap
    fig_width = 9.1 + plot_df.shape[1] * 0.21
    fig_height = 7.3 + plot_df.shape[0] * 0.11
    fig_size = 32 - min((fig_width * fig_height) ** 0.61, 28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # gets the 3rd quartile of fit times for each combination of regressor
    # and gene across cross-validation runs, takes the average across
    # all genes to get the computational complexity for each regressor
    time_vals = pd.Series({
        mdl: (out_data['Tune']['Time']['fit']['avg']
              + out_data['Tune']['Time']['fit']['std']).groupby(
                  axis=1, level=0).quantile(q=0.75).mean().mean()
        for mdl, out_data in out_dict.items()
        })

    time_vals = time_vals.loc[plot_df.columns]
    plot_df.columns = ['{} {}  ({:.3g}s)'.format(src, mdl, vals)
                       for (src, mdl), vals in time_vals.iteritems()]

    # creates labels denoting the best correlation for each gene's CNAs
    # across all classifiers to place on the output heatmap
    annot_values = plot_df.applymap('{:.3f}'.format)
    for mtype, cor_vals in plot_df.iterrows():
        best_stat = plot_df.columns == cor_vals.idxmax()
        annot_values.loc[mtype, ~best_stat] = ''

    # creates the heatmap of correlations for regressors x genes
    ax = sns.heatmap(plot_df, cmap=cor_cmap, vmin=0, vmax=1, center=0.5,
                     annot=annot_values, fmt='', annot_kws={'size': fig_size})

    ax.figure.axes[-1].tick_params(labelsize=fig_size * 1.73)
    ax.figure.axes[-1].set_ylabel("Correlation\n(25-fold CV 1st quartile)",
                                  size=fig_size * 1.9, weight='semibold')

    ax.figure.axes[0].tick_params(
        axis='x', length=fig_size * 0.53, width=fig_size * 0.17)
    plt.xticks(size=fig_size * 1.51, rotation=34, ha='right')
    plt.yticks(size=fig_size * 1.37, rotation=0)
    plt.xlabel('Model', size=fig_size * 2.43, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, "{}__cor-highlights.png".format(args.cohort)),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_mse_time(out_dict, args):
    fig, axarr = plt.subplots(figsize=(9, 15), nrows=2, sharex=True)

    time_quarts = np.log2(pd.Series({
        mdl: (out_data['Tune']['Time']['fit']['avg']
              + out_data['Tune']['Time']['fit']['std']).groupby(
                  axis=1, level=0).quantile(q=0.75).mean().mean()
        for mdl, out_data in out_dict.items()
        }))

    mse_vals = {mdl: out_data['Fit']['test'].MSE.quantile(q=0.75, axis=1)
                for mdl, out_data in out_dict.items()}

    mse_list = [
        pd.Series({mdl: vals.mean() for mdl, vals in mse_vals.items()}),
        pd.Series({mdl: vals.quantile(q=0.25)
                   for mdl, vals in mse_vals.items()}),
        ]

    expr_vec = time_quarts.index.get_level_values(0)
    expr_shapes = [use_marks[sorted(set(expr_vec)).index(expr)]
                   for expr in expr_vec]

    model_vec = time_quarts.index.get_level_values(1).str.split(
        '__').map(itemgetter(0))
    model_cmap = sns.color_palette(
        'Set1', n_colors=len(set(model_vec)), desat=.34)
    model_clrs = [model_cmap[sorted(set(model_vec)).index(mdl)]
                  for mdl in model_vec]

    for ax, mses in zip(axarr, mse_list):
        for time_val, mse_val, expr_shape, model_clr in zip(
                time_quarts.values, mses.values,
                expr_shapes, model_clrs
                ):
            ax.scatter(time_val, mse_val,
                       marker=expr_shape, c=model_clr, s=71, alpha=0.41)

        for annot_x, annot_y, annot, halign in place_annot(
                time_quarts.values.tolist(), mses.values.tolist(),
                size_vec=[71 for _ in time_quarts],
                annot_vec=[' '.join(tst) for tst in time_quarts.index],
                x_range=time_quarts.max() - time_quarts.min(),
                y_range=mses.max() - mses.min(), gap_adj=79
                ):
            ax.text(annot_x, annot_y, annot, size=10, ha=halign)

        ax.tick_params(axis='y', labelsize=14)

    axarr[1].xaxis.set_major_formatter(ticker.FormatStrFormatter(r'$2^{%d}$'))
    axarr[1].tick_params(axis='x', labelsize=21, pad=7)
    axarr[0].set_ylabel("Average MSE", size=23, weight='semibold')
    axarr[1].set_ylabel("First Quartile MSE", size=23, weight='semibold')

    plt.xlabel("Fitting Time (seconds)", size=23, weight='semibold')
    plt.tight_layout(h_pad=3.3)

    fig.savefig(
        os.path.join(plot_dir, "{}__mse-time.png".format(args.cohort)),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the success of all models tested in predicting the copy "
        "number scores of the genes in a given cohort."
        )

    # parse command-line arguments, create directory to store the plots
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # search for experiment output directories corresponding to this cohort
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "*__{}__samps-*/out-data__*.p".format(args.cohort))
        ]

    # get the experiment output directory for each combination of input
    # expression source and algorithm with the lowest sample incidence cutoff
    out_use = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Samps': int(out_data[0].split('__samps-')[1]),
         'Model': out_data[1].split('out-data__')[1].split('.p')[0]}
        for out_data in out_datas
        ]).groupby(['Model', 'Source'])['Samps'].min().reset_index(
            'Model').set_index('Samps', append=True)

    # load the cohort expression and copy number data for each combination
    # of expression source and sample cutoff
    cdata_dict = {
        (src, ctf): merge_cohort_data(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf)))
        for src, ctf in set(out_use.index)
        }

    # load the experiment output for each combination of source and cutoff
    out_dict = {
        (src, mdl.values[0]): pickle.load(open(
            os.path.join(base_dir,
                         "{}__{}__samps-{}".format(src, args.cohort, ctf),
                         "out-data__{}.p".format(mdl.values[0])),
            'rb'
            ))
        for (src, ctf), mdl in out_use.iterrows()
        }

    # create the plots
    plot_cor_highlights(out_dict.copy(), args, cdata_dict)
    plot_mse_time(out_dict.copy(), args)


if __name__ == "__main__":
    main()

