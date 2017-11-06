
import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots')

sys.path.extend([os.path.join(base_dir, '../../..')])
import HetMan.experiments.utilities as utils

from HetMan.features.variants import MuType

import numpy as np
import pandas as pd
from math import log
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
from pylab import rcParams


def plot_mtype_highlights(out_data, args, mtype_choice='Genes'):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    use_data = out_data.apply(sorted, axis=1)
    mtype_means = use_data.mean(axis=1)

    if isinstance(mtype_choice, MuType):
        use_mtypes = sorted([mtype for mtype in use_data.index
                             if mtype_choice.is_supertype(mtype)
                             and mtype != mtype_choice],
                            key=lambda x: mtype_means[x])

        use_mtypes += [mtype_choice]
        ax.add_patch(mpl.patches.Rectangle(
            (len(use_mtypes) - 1.49, 0), 0.98, 1,
            fill=True, color='blue', alpha=0.1, zorder=500
            ))

        use_mtypes += sorted([mtype for mtype in use_data.index
                              if mtype.is_supertype(mtype_choice)
                              and mtype != mtype_choice],
                             key=lambda x: mtype_means[x], reverse=True)

    elif mtype_choice == 'Genes':
        use_mtypes = [
            mtype for mtype in use_data.index
            if len(mtype) == 1 and mtype.subtype_list()[0][1] is None
            ]
        use_mtypes = sorted(use_mtypes,
                            key=lambda x: mtype_means[x], reverse=True)

    elif mtype_choice == 'Best':
        use_mtypes = mtype_means.sort_values(ascending=False)[:20].index

    else:
        raise ValueError("Unrecognized mtype_choice argument!")

    use_data = use_data.loc[use_mtypes, :]
    ax.plot(use_data.iloc[:, 0].values,
            '.', color='black', ms=4)
    ax.plot(use_data.iloc[:, 2].values,
            '_', color='#9CADB5', ms=9, mew=2.5)
    ax.plot(use_data.iloc[:, 4].values,
            '.', color='black', ms=4)

    for i in range(len(use_mtypes)):
        ax.add_patch(mpl.patches.Rectangle(
            (i - 0.15, use_data.iloc[i, 1]),
            0.3, use_data.iloc[i, 3] - use_data.iloc[i, 1],
            fill=True, color='0.3', alpha=0.2, zorder=1000
            ))

    plt.xticks(np.arange(len(use_mtypes)), use_mtypes,
               fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=13)

    plt.xlim(-0.41, len(use_mtypes) - 0.59)
    plt.ylim(use_data.min().min() - 0.02, 1.0)

    plt.axhline(color='r', y=0.5, xmin=-1, xmax=len(use_mtypes) + 1,
                linewidth=1.5, linestyle='--')
    plt.ylabel('AUC', fontsize=25)

    fig.set_size_inches(len(use_mtypes) ** 0.9 / 2.5, 9)
    plt.savefig(
        os.path.join(plot_dir, 'mtype-highlights_{}-{}_{}.png'.format(
                args.cohort, args.classif, mtype_choice)),
        dpi=600, bbox_inches='tight'
        )

    plt.close()


def plot_auc_distribution(out_data, args):
    rcParams['figure.figsize'] = 24, 8

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
            plot_dir, 'mtype-performance_{}_{}.png'.format(
                args.cohort, args.classif)),
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

    # parses command line arguments
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')
    args = parser.parse_args()

    search_data = utils.test_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'search'))

    plot_auc_distribution(search_data, args)

    plot_mtype_highlights(search_data, args, mtype_choice='Genes')
    plot_mtype_highlights(search_data, args, mtype_choice='Best')

    plot_mtype_highlights(search_data, args,
                          mtype_choice=MuType({('Gene', 'TP53'): None}))
    plot_mtype_highlights(search_data, args,
                          mtype_choice=MuType({('Gene', 'KRAS'): None}))
    plot_mtype_highlights(search_data, args,
                          mtype_choice=MuType({('Gene', 'CDKN2A'): None}))
    plot_mtype_highlights(search_data, args,
                          mtype_choice=MuType({('Gene', 'SMAD4'): None}))

    acc_data, coef_data = utils.depict_output(
        os.path.join(base_dir, 'output',
                     args.cohort, args.classif, 'portray'))

    plot_coef_heatmap(coef_data, args)
    plot_coef_heatmap(coef_data, args, auc_cutoff=0.85, acc_data=acc_data)
    plot_coef_heatmap(coef_data, args, auc_cutoff=0.9, acc_data=acc_data)


if __name__ == '__main__':
    main()

