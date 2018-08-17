
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'cohort')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.mut_baseline import *
from HetMan.experiments.mut_baseline.fit_tests import load_output
from HetMan.experiments.mut_baseline.setup_tests import get_cohort_data
from HetMan.experiments.utilities import auc_cmap

import argparse
import pandas as pd
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_acc_highlights(out_dict, args, cdata):
    auc_quarts = pd.DataFrame.from_dict(
        {mdl: auc_df.applymap(itemgetter('test')).quantile(q=0.25, axis=1)
         for mdl, (auc_df, _, _, _, _) in out_dict.items()}
        )

    use_mtypes = auc_quarts.max(axis=1).sort_values()[-40:].index
    use_models = auc_quarts.loc[use_mtypes, :].max() > 0.7
    plot_df = auc_quarts.loc[use_mtypes, use_models]
    fig, ax = plt.subplots(figsize=(2 + plot_df.shape[1] * 0.7, 18))

    time_vals = {mdl: time_df.quantile(q=0.75, axis=1).mean()
                 for mdl, (_, _, time_df, _, _) in out_dict.items()}
    plot_df.columns = ['{}  ({:.3g}s)'.format(mdl, time_vals[mdl])
                       for mdl in plot_df.columns]

    annot_values = plot_df.applymap('{:.3f}'.format)
    for mtype, auc_vals in plot_df.iterrows():
        best_stat = plot_df.columns == auc_vals.idxmax()
        annot_values.loc[mtype, ~best_stat] = ''
 
    ax = sns.heatmap(plot_df, cmap=auc_cmap,
                     vmin=0, vmax=1, center=0.5, yticklabels=True,
                     annot=annot_values, fmt='', annot_kws={'size': 13})

    ax.figure.axes[-1].tick_params(labelsize=24)
    ax.figure.axes[-1].set_ylabel('AUC (25-fold CV 1st quartile)',
                                  size=31, weight='semibold')

    ax.figure.axes[0].tick_params(axis='x', length=8, width=3)
    plt.xticks(size=23, rotation=38, ha='right')
    plt.yticks(size=17, rotation=0)
    plt.xlabel('Model', size=34, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, '{}__{}'.format(args.expr_source, args.cohort),
                     "auc-highlights.png"),
        dpi=200, bbox_inches='tight'
        )
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the success of all models tested in predicting the presence "
        "of the mutations in a given cohort."
        )

    parser.add_argument('expr_source', type=str, choices=['Firehose', 'toil'],
                        help="which TCGA expression data source was used")
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    args = parser.parse_args()
    cdata = get_cohort_data(args.expr_source, args.cohort, args.samp_cutoff)

    os.makedirs(os.path.join(plot_dir,
                             '{}__{}'.format(args.expr_source, args.cohort)),
                exist_ok=True)

    out_dir = os.path.join(base_dir, 'output',
                           args.expr_source, '{}__samps-{}'.format(
                               args.cohort, args.samp_cutoff))
    out_models = os.listdir(out_dir)

    out_dict = dict()
    for out_model in out_models:

        out_fls = [
            out_fl for out_fl in os.listdir(os.path.join(out_dir, out_model))
            if 'out__' in out_fl
            ]

        log_fls = [
            log_fl for log_fl in os.listdir(os.path.join(
                out_dir, out_model, 'slurm'))
            if 'fit-' in log_fl
            ]

        if len(log_fls) > 0 and len(log_fls) == (len(out_fls) * 2):
            out_dict[out_model] = load_output(
                args.expr_source, args.cohort, args.samp_cutoff, out_model)

    plot_acc_highlights(out_dict, args, cdata)


if __name__ == "__main__":
    main()

