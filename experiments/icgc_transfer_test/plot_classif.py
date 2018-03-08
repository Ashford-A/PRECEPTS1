
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'classif')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.icgc import MutationCohort as ICGCcohort
from HetMan.features.variants import MuType

import argparse
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

icgc_data_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'


def load_cohort_info():
    info_dict = pickle.load(open(os.path.join(base_dir,
                                              "setup", "cohort_info.p"),
                                 'rb'))

    return info_dict


def load_base_accuracies(classif):
    
    out_lists = [
        pickle.load(open(os.path.join(base_dir, "setup",
                                      "{}__cv_{}.p".format(classif, cv_id)
                                        ), 'rb'))
        for cv_id in range(10)
        ]

    out_data = pd.concat([pd.DataFrame.from_dict(out_dict, orient='index')
                          for out_dict in out_lists],
                         axis=1)
    
    return out_data


def load_classif_output(classif):

    out_lists = [
        [pickle.load(
            open(os.path.join(base_dir, "output", classif,
                              "out__cv-{}_task-{}.p".format(cv_id, task_id)
                             ), 'rb')
            )['Acc'] for task_id in range(5)]
        for cv_id in range(10)
        ]

    out_data = pd.concat(
        [pd.concat(pd.DataFrame.from_dict(x, orient='index').stack()
                   for x in ols) for ols in out_lists],
        axis=1
        )

    return out_data


def plot_auc_distributions(out_data, args, cdata):
    fig, ax = plt.subplots(figsize=(out_data.shape[0] ** 0.9 / 2.1, 8))

    use_data = out_data.apply(sorted, axis=1)
    mtype_means = use_data.mean(axis=1)

    ax.plot(use_data.iloc[:, 0].values,
            '.', color='black', ms=4)
    ax.plot(use_data.quantile(q=0.5, axis=1).values,
            '_', color='#9CADB5', ms=9, mew=4.0)
    ax.plot(use_data.iloc[:, 9].values,
            '.', color='black', ms=4)

    for i in range(use_data.shape[0]):
        ax.add_patch(mpl.patches.Rectangle(
            (i - 0.15, use_data.iloc[i, 2]),
            0.3, use_data.iloc[i, 7] - use_data.iloc[i, 2],
            fill=True, color='0.3', alpha=0.2, zorder=1000
            ))

    plt.xticks(np.arange(use_data.shape[0]), use_data.index,
               fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=13)

    plt.xlim(-0.41, use_data.shape[0] - 0.59)
    plt.ylim(use_data.min().min() - 0.02, 1.02)

    plt.axhline(color='r', y=0.5, xmin=-1, xmax=use_data.shape[0] + 1,
                linewidth=0.8, linestyle='--')
    plt.ylabel('AUC', fontsize=25)

    plt.savefig(os.path.join(plot_dir,
                             "boxplot_baseline__{}.png".format(args.classif)),
                dpi=500, bbox_inches='tight')
    plt.close()


def plot_auc_heatmap(out_data, args, cohort_info):

    # get the first quartile of AUC performances across the ten cross
    # -validation folds, set the size of the figure according to the data size
    quant_values = out_data.quantile(q=0.25, axis=1).unstack().transpose()
    fig, ax = plt.subplots(figsize=(quant_values.shape[1] / 1.1,
                                    quant_values.shape[0] / 3.8))

    # create the labels for the x-axis with the cohort names and sizes, as
    # well as the color gradient map to use in the heatmap proper
    cohort_lbls = ['{} ({})'.format(coh, cohort_info[coh]['Samples'])
                   for coh in quant_values.columns]
    use_cmap = sns.diverging_palette(240, 10, s=75, l=65, sep=30, n=500,
                                     center='dark', as_cmap=True)

    # turn data frame of AUC values into a sorted list
    annot_values = quant_values.copy()
    annot_flat = annot_values.values.flatten()
    annot_flat = annot_flat[~np.isnan(annot_flat)]
    annot_flat.sort()

    # get top ten AUC values to annotate the cells in the heatmap
    annot_values = annot_values.round(3)
    annot_values[annot_values < annot_flat[-10]] = ''

    # create the heatmap
    ax = sns.heatmap(quant_values, cmap=use_cmap,
                     vmin=0, vmax=1, center=0.5, xticklabels=cohort_lbls,
                     annot=annot_values, fmt='', annot_kws={'size': 10})

    # set colorbar label properties
    ax.figure.axes[-1].tick_params(labelsize=21)
    ax.figure.axes[-1].set_ylabel('AUC (ten-fold CV 1st quartile)', size=28)

    # set heatmap x-axis tick and label properties
    ax.figure.axes[0].tick_params(axis='x', length=10, width=3)
    plt.xticks(rotation=45, ha='right', size=19)
    plt.xlabel('TCGA Cohort (# of samples)', size=34)

    # save the figure to file
    fig.savefig(os.path.join(plot_dir,
                             "heatmap_absolute__{}.png".format(args.classif)),
                dpi=500, bbox_inches='tight')

    plt.close()


def main():

    # parses command line arguments
    parser = argparse.ArgumentParser(
        description='Plot experiment results for given mutation classifier.')
    parser.add_argument('classif', help='a mutation classifier')
    args = parser.parse_args()

    # load ICGC expression and mutation data, create directory to save plots
    cdata_icgc = ICGCcohort('PACA-AU', icgc_data_dir, mut_genes=None,
                            samp_cutoff=[1/12, 11/12], cv_prop=1.0)
    os.makedirs(plot_dir, exist_ok=True)

    # load experiment data
    cohort_info = load_cohort_info()
    base_df = load_base_accuracies(args.classif)
    acc_df = load_classif_output(args.classif)

    # create the plots
    plot_auc_distributions(base_df, args, cdata_icgc)
    plot_auc_heatmap(acc_df, args, cohort_info)


if __name__ == '__main__':
    main()

