
import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots')
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities import test_output
from HetMan.features.variants import MuType

import numpy as np
import pandas as pd
import argparse

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns


def plot_auc_distribution(out_data, args):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    med_perf = out_data.quantile(q=0.5, axis=1)
    top_perf = out_data.max(axis=1) - med_perf
    bot_perf = med_perf - out_data.min(axis=1)

    perf_df = pd.DataFrame(
        {'Med': med_perf, 'Top': top_perf, 'Bot': bot_perf})
    sort_perf = med_perf.sort_values(ascending=False)
    sort_indx = [med_perf.index.get_loc(x) for x in sort_perf.index]

    top_perf = top_perf[sort_indx]
    bot_perf = bot_perf[sort_indx]
    err_arr = np.array(pd.concat([bot_perf, top_perf], axis=1).transpose())

    ax.errorbar(x=range(out_data.shape[0]), y=sort_perf, yerr=err_arr,
                elinewidth=0.7, alpha=0.5, color='#726437')

    plt.xlim(out_data.shape[0] / -125, out_data.shape[0] * (126/125))
    plt.ylim(-0.02, 1.02)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.grid(axis='x')
    ax.grid(axis='y', which='major', linewidth=3, alpha=0.6)

    plt.ylabel('AUC', fontsize=25)
    plt.axhline(color='r', y=0.5, xmin=-10, xmax=out_data.shape[0] + 10,
                linewidth=1, linestyle='--')

    fig.set_size_inches(out_data.shape[0] ** 0.9 / 125, 9)
    plt.savefig(os.path.join(
        plot_dir, 'mtype-performance_{}-{}.png'.format(
            args.cohort, args.classif)
        ),
        dpi=600, bbox_inches='tight')
    plt.close()


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
    plt.savefig(os.path.join(
        plot_dir, 'mtype-highlights_{}-{}_{}.png'.format(
            args.cohort, args.classif, mtype_choice)
        ),
        dpi=600, bbox_inches='tight')
    plt.close()

    return use_mtypes


def main():

    # parses command line arguments
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')
    args = parser.parse_args()

    search_data = test_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'search'))

    print('Plotting distribution of mtype performance...')
    plot_auc_distribution(search_data, args)

    print('Plotting performance of single-gene mtypes...')
    gene_mtypes = plot_mtype_highlights(search_data, args,
                                        mtype_choice='Genes')

    print('Plotting performance of the best mtypes...')
    best_mtypes = plot_mtype_highlights(search_data, args,
                                        mtype_choice='Best')

    for gene_mtype in gene_mtypes:
        print('Plotting performance of the neighbourhood of mtype {} ...'\
                .format(gene_mtype))
        _ = plot_mtype_highlights(search_data, args, mtype_choice=gene_mtype)


if __name__ == '__main__':
    main()

