
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'variant_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'cohort')

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.utilities.scatter_plotting import place_annot

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

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


def plot_auc_highlights(out_dict, args, cdata_dict):
    """Plots the accuracy of each classifier for the top mutations."""

    # calculates the first quartile of the testing AUC of each classifier on
    # each mutation type across the cross-validation runs
    auc_quarts = pd.DataFrame.from_dict({
        mdl: out_data['Fit']['test']['AUC'].quantile(q=0.25, axis=1)
        for mdl, out_data in out_dict.items()
        })

    # gets the top forty mutation types by the best first-quartile AUC across
    # all classifiers, gets the classifiers that did well on at least one type
    use_mtypes = auc_quarts.max(axis=1).sort_values()[-40:].index
    use_models = auc_quarts.loc[use_mtypes, :].max() > 0.7
    plot_df = auc_quarts.loc[use_mtypes, use_models]

    # set the size of the plot and the base label size based on the number
    # of data points that will be shown in the heatmap
    fig_width = 9.1 + plot_df.shape[1] * 0.21
    fig_height = 7.3 + plot_df.shape[0] * 0.11
    fig_size = 32 - min((fig_width * fig_height) ** 0.61, 28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # gets the 3rd quartile of fit times for each combination of classifier
    # and mutation type across cross-validation runs, takes the average across
    # all types to get the computational complexity for each classifier
    time_vals = pd.Series({
        mdl: (out_data['Tune']['Time']['fit']['avg']
              + out_data['Tune']['Time']['fit']['std']).groupby(
                  axis=1, level=0).quantile(q=0.75).mean().mean()
        for mdl, out_data in out_dict.items()
        })

    time_vals = time_vals.loc[plot_df.columns]
    plot_df.columns = ['{} {}  ({:.3g}s)'.format(src, mdl, vals)
                       for (src, mdl), vals in time_vals.iteritems()]

    # creates labels denoting the best AUC for each mutation type across all
    # classifiers to place on the output heatmap
    annot_values = plot_df.applymap('{:.3f}'.format)
    for mtype, auc_vals in plot_df.iterrows():
        best_stat = plot_df.columns == auc_vals.idxmax()
        annot_values.loc[mtype, ~best_stat] = ''

    mtype_sizes = {mtype: {src: mtype.get_samples(cdata.mtree)
                           for src, cdata in cdata_dict.items()}
                   for mtype in use_mtypes}

    for mtype, samp_dict in mtype_sizes.items():
        if not auc_quarts.loc[mtype].isna().any():
            assert len(set(frozenset(samps)
                           for samps in samp_dict.values())) == 1

    mtype_sizes = {mtype: max(len(samps) for samps in samp_dict.values())
                   for mtype, samp_dict in mtype_sizes.items()}
    mtype_lbls = ["{} ({})".format(str(mtype), mtype_sizes[mtype])
                  for mtype in plot_df.index]

    # creates the heatmap of AUC values for classifiers x mutation types
    ax = sns.heatmap(plot_df, cmap=auc_cmap, vmin=0, vmax=1, center=0.5,
                     yticklabels=mtype_lbls, annot=annot_values, fmt='',
                     annot_kws={'size': fig_size})

    ax.figure.axes[-1].tick_params(labelsize=fig_size * 1.73)
    ax.figure.axes[-1].set_ylabel('AUC (25-fold CV 1st quartile)',
                                  size=fig_size * 1.9, weight='semibold')

    ax.figure.axes[0].tick_params(
        axis='x', length=fig_size * 0.53, width=fig_size * 0.17)
    plt.xticks(size=fig_size * 1.51, rotation=34, ha='right')
    plt.yticks(size=fig_size * 1.37, rotation=0)
    plt.xlabel('Model', size=fig_size * 2.43, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, '{}__auc-highlights.svg'.format(args.cohort)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_aupr_time(out_dict, args):
    fig, axarr = plt.subplots(figsize=(9, 15), nrows=2, sharex=True)

    time_quarts = np.log2(pd.Series({
        mdl: (out_data['Tune']['Time']['fit']['avg']
              + out_data['Tune']['Time']['fit']['std']).groupby(
                  axis=1, level=0).quantile(q=0.75).mean().mean()
        for mdl, out_data in out_dict.items()
        }))

    aupr_vals = {mdl: out_data['Fit']['test']['AUPR'].quantile(q=0.25, axis=1)
                 for mdl, out_data in out_dict.items()}

    aupr_list = [
        pd.Series({mdl: vals.mean() for mdl, vals in aupr_vals.items()}),
        pd.Series({mdl: vals.quantile(q=0.75)
                   for mdl, vals in aupr_vals.items()}),
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

    for ax, auprs in zip(axarr, aupr_list):
        for time_val, aupr_val, expr_shape, model_clr in zip(
                time_quarts.values, auprs.values,
                expr_shapes, model_clrs
                ):
            ax.scatter(time_val, aupr_val,
                       marker=expr_shape, c=model_clr, s=71, alpha=0.41)

        for annot_x, annot_y, annot, halign in place_annot(
                time_quarts.values.tolist(), auprs.values.tolist(),
                size_vec=[71 for _ in time_quarts],
                annot_vec=[' '.join(tst) for tst in time_quarts.index],
                x_range=time_quarts.max() - time_quarts.min(),
                y_range=auprs.max() - auprs.min(), gap_adj=79
                ):
            ax.text(annot_x, annot_y, annot, size=10, ha=halign)

        ax.tick_params(axis='y', labelsize=14)

    axarr[1].xaxis.set_major_formatter(ticker.FormatStrFormatter(r'$2^{%d}$'))
    axarr[1].tick_params(axis='x', labelsize=21, pad=7)
    axarr[0].set_ylabel('Average AUPR', size=23, weight='semibold')
    axarr[1].set_ylabel('Third Quartile AUPR', size=23, weight='semibold')

    plt.xlabel('Fitting Time (seconds)', size=23, weight='semibold')
    plt.tight_layout(h_pad=3.3)

    fig.savefig(
        os.path.join(plot_dir, '{}__aupr-time.svg'.format(args.cohort)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the success of all models tested in predicting the presence "
        "of the mutations in a given cohort."
        )

    # parse command-line arguments, create directory to store the plots
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # search for experiment output directories corresponding to this cohort
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "*__{}__samps-*/out-data__*.p.gz".format(args.cohort))
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

    # load the cohort expression and mutation data for each combination of
    # expression source and sample cutoff
    cdata_dict = {
        (src, ctf): merge_cohort_data(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf)))
        for src, ctf in set(out_use.index)
        }

    # load the experiment output for each combination of source and cutoff
    out_dict = {
        (src, mdl.values[0]): pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(src, args.cohort, ctf),
            "out-data__{}.p.gz".format(mdl.values[0])
            ), 'r'))
        for (src, ctf), mdl in out_use.iterrows()
        }

    # create the plots
    plot_auc_highlights(out_dict.copy(), args, cdata_dict)
    plot_aupr_time(out_dict.copy(), args)


if __name__ == "__main__":
    main()

