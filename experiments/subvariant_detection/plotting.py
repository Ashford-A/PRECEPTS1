
import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots')

sys.path.extend([os.path.join(base_dir, '../../..')])
import HetMan.experiments.utilities as utils

import numpy as np
import pandas as pd
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
from pylab import rcParams


def plot_auc_distribution(cohort, classif):
    rcParams['figure.figsize'] = 24, 8

    out_data = utils.test_output(os.path.join(
        base_dir, 'output', cohort, classif, 'search'))

    med_perf = out_data.quantile(q=0.5, axis=1)
    top_perf = out_data.max(axis=1) - med_perf
    bot_perf = med_perf - out_data.min(axis=1)
    perf_df = pd.DataFrame({'Med': med_perf, 'Top': top_perf, 'Bot': bot_perf})

    sort_perf = med_perf.sort_values(ascending=False)
    sort_indx = [med_perf.index.get_loc(x) for x in sort_perf.index]
    top_perf = top_perf[sort_indx]
    bot_perf = bot_perf[sort_indx]
    err_arr = np.array(pd.concat([bot_perf, top_perf], axis=1).transpose())

    plt.errorbar(x=range(out_data.shape[0]), y=sort_perf,
                 yerr=err_arr, elinewidth=0.9)

    plt.savefig(
        os.path.join(
            plot_dir, 'mtype-performance_{}_{}.png'.format(cohort, classif)),
        dpi=500, bbox_inches='tight'
        )
    plt.close()


def plot_coef_heatmap(coef_data, args, auc_cutoff=None, acc_data=None):

    if auc_cutoff is None:
        plot_file = "mtype-coefs_{}_{}.png".format(args.cohort, args.classif)
        use_mtypes = coef_data.index

    else:
        plot_file = "mtype-coefs_{}_{}_auc-cutoff-{}.png".format(
                args.cohort, args.classif, auc_cutoff)
        use_mtypes = acc_data.index[acc_data.mean(axis=1) > auc_cutoff]

    use_data = coef_data.loc[use_mtypes, coef_data.max(axis=0) > 0.01]

    sns.set_context("paper")

    sig_plot = sns.clustermap(
        use_data, method='centroid',
        cmap=sns.cubehelix_palette(light=1, as_cmap=True),
        figsize=(33, 17)
        )

    sig_plot.savefig(os.path.join(plot_dir, plot_file))


def main():
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')

    args = parser.parse_args()

    plot_auc_distribution(args.cohort, args.classif)

    acc_data, coef_data = utils.depict_output(
        os.path.join(base_dir, 'output',
                     args.cohort, args.classif, 'portray'))

    plot_coef_heatmap(coef_data, args)
    plot_coef_heatmap(coef_data, args, auc_cutoff=0.85, acc_data=acc_data)
    plot_coef_heatmap(coef_data, args, auc_cutoff=0.9, acc_data=acc_data)


if __name__ == '__main__':
    main()

