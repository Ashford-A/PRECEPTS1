
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'cohort')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.mut_baseline import *
from HetMan.experiments.mut_baseline.fit_tests import load_output
from HetMan.experiments.mut_baseline.setup_tests import get_cohort_data
from HetMan.experiments.utilities import auc_cmap

import numpy as np
import pandas as pd

import argparse
from pathlib import Path
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
from matplotlib import ticker

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
use_marks = [(0, 3, 0)]
use_marks += [(i, 0, k) for k in (0, 140) for i in (3, 4, 5)]


def plot_acc_highlights(out_dict, args):
    auc_quarts = pd.DataFrame.from_dict(
        {mdl: auc_df.applymap(itemgetter('test')).quantile(q=0.25, axis=1)
         for mdl, (auc_df, _, _, _, _) in out_dict.items()}
        )

    use_mtypes = auc_quarts.max(axis=1).sort_values()[-40:].index
    use_models = auc_quarts.loc[use_mtypes, :].max() > 0.7
    plot_df = auc_quarts.loc[use_mtypes, use_models]

    # set the size of the plot and the base label size based on the number
    # of data points that will be shown in the heatmap
    fig_width = 5.1 + plot_df.shape[1] * 0.29
    fig_height = 1.9 + plot_df.shape[0] * 0.19
    fig_size = (fig_width * fig_height) ** 0.37
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    time_vals = {mdl: time_df.quantile(q=0.75, axis=1).mean()
                 for mdl, (_, _, time_df, _, _) in out_dict.items()}
    plot_df.columns = ['{} {}  ({:.3g}s)'.format(src, mdl, vals)
                       for (src, mdl), vals in time_vals.items()]

    annot_values = plot_df.applymap('{:.3f}'.format)
    for mtype, auc_vals in plot_df.iterrows():
        best_stat = plot_df.columns == auc_vals.idxmax()
        annot_values.loc[mtype, ~best_stat] = ''
 
    ax = sns.heatmap(plot_df, cmap=auc_cmap, vmin=0, vmax=1, center=0.5,
                     yticklabels=True, annot=annot_values, fmt='',
                     annot_kws={'size': fig_size * 1.63})

    ax.figure.axes[-1].tick_params(labelsize=fig_size * 2.76)
    ax.figure.axes[-1].set_ylabel('AUC (25-fold CV 1st quartile)',
                                  size=fig_size * 3.4, weight='semibold')

    ax.figure.axes[0].tick_params(
        axis='x', length=fig_size * 0.71, width=fig_size * 0.23)
    plt.xticks(size=fig_size * 2.13, rotation=34, ha='right')
    plt.yticks(size=fig_size * 1.81, rotation=0)
    plt.xlabel('Model', size=fig_size * 3.7, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, '{}__auc-highlights.png'.format(args.cohort)),
        dpi=250, bbox_inches='tight'
        )
    plt.close()


def plot_aupr_time(out_dict, args):
    fig, ax = plt.subplots(figsize=(10, 9))

    time_quarts = np.log2(pd.Series(
        {mdl: time_df.quantile(q=0.75, axis=1).mean()
         for mdl, (_, _, time_df, _, _) in out_dict.items()}
        ))

    aupr_quarts = pd.Series(
        {mdl: aupr_df.applymap(itemgetter('test')).quantile(
            q=0.25, axis=1).mean()
         for mdl, (_, aupr_df, _, _, _) in out_dict.items()}
        )

    expr_vec = time_quarts.index.get_level_values(0)
    expr_shapes = [use_marks[sorted(set(expr_vec)).index(expr)]
                   for expr in expr_vec]

    model_vec = time_quarts.index.get_level_values(1).str.split(
        '__').map(itemgetter(0))
    model_cmap = sns.color_palette(
        'Set1', n_colors=len(set(model_vec)), desat=.34)
    model_clrs = [model_cmap[sorted(set(model_vec)).index(mdl)]
                  for mdl in model_vec]

    for time_val, aupr_val, expr_shape, model_clr in zip(
            time_quarts.values, aupr_quarts.values, expr_shapes, model_clrs):
        ax.scatter(time_val, aupr_val,
                   marker=expr_shape, c=model_clr, s=71, alpha=0.41)

    ax.tick_params(axis='x', labelsize=17, pad=7)
    plt.yticks(size=17)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(r'$2^{%d}$'))

    plt.xlabel('Fitting Time (seconds)', size=21, weight='semibold')
    plt.ylabel('Average AUPR', size=21, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, '{}__aupr-time.png'.format(args.cohort)),
        dpi=250, bbox_inches='tight'
        )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the success of all models tested in predicting the presence "
        "of the mutations in a given cohort."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    out_path = Path(os.path.join(base_dir, 'output'))
    out_dirs = [
        out_dir.parent for out_dir in out_path.glob(
            "*/{}__samps-*/*/out__cv-0_task-0.p".format(args.cohort))
        if (len(tuple(out_dir.parent.glob("out__*.p"))) > 0
            and (len(tuple(out_dir.parent.glob("out__*.p")))
                 == len(tuple(out_dir.parent.glob("slurm/fit-*.txt")))))
        ]

    parsed_dirs = [str(out_dir).split("/output/")[1].split('/')
                   for out_dir in out_dirs]

    if len(set(prs[1] for prs in parsed_dirs)) > 1:
        pass

    else:
        samp_ctfs = parsed_dirs[0][1].split('__samps-')[1]
        parsed_dirs = [[prs[0]] + prs[2:] for prs in parsed_dirs]

    out_dict = {(src, mdl): load_output(src, args.cohort, samp_ctfs, mdl)
                for src, mdl in parsed_dirs}

    plot_acc_highlights(out_dict, args)
    plot_aupr_time(out_dict, args)


if __name__ == "__main__":
    main()

