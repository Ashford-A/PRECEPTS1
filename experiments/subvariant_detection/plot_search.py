
"""Plots the results of searching for classifiable sub-types in a cohort.

Examples:
    plot_search.py PAAD Lasso
    plot_search.py BRCA ElasticNet
    plot_search.py OV rForest

"""

import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots', 'search')
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities import test_output
from HetMan.features.cohorts import VariantCohort
from HetMan.features.variants import MuType

import numpy as np
import pandas as pd

import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import colorsys

from functools import reduce
from operator import or_

from math import log
from scipy.stats import fisher_exact

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def plot_auc_distribution(out_data, args, mtype_choice=None, cdata=None):

    if mtype_choice is None != cdata is None:
        raise ValueError("Both or neither mtype and cdata must be specified!")

    fig, ax = plt.subplots(nrows=1, ncols=1)

    if mtype_choice is None:
        plot_fl = 'mtype-performance_{}-{}.png'.format(
            args.cohort, args.classif)

    else:
        plot_fl = 'mtype-performance_{}-{}_{}.png'.format(
            args.cohort, args.classif, mtype_choice)
        choice_samps = mtype_choice.get_samples(cdata.train_mut)

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

    for i, mtype in enumerate(sort_perf.index):
        if mtype_choice is None:
            ax.errorbar(
                x=i, y=sort_perf[i], yerr=err_arr[:, i].reshape(-1, 1),
                fmt='o', ms=2, elinewidth=0.9, alpha=0.2, color='#726437'
                )

        else:
            cur_samps = mtype.get_samples(cdata.train_mut)
            both_samps = choice_samps & cur_samps
            or_samps = choice_samps | cur_samps

            ovlp_val = fisher_exact(
                [[len(both_samps), len(cur_samps - both_samps)],
                 [len(choice_samps - both_samps),
                  len(cdata.samples - or_samps)]],
                alternative='greater'
                )[0]
            sub_val = log(len(cur_samps) / len(choice_samps), 2)

            clr_h = 300 + sub_val * 35
            clr_l = 75 - min(ovlp_val, 25) * 3
            clr_s = min(ovlp_val, 5) * 20

            plt_size = min(ovlp_val, 20) ** 0.5 / 3.5
            plt_clr = colorsys.hls_to_rgb(
                clr_h / 360, clr_l / 100, clr_s / 100)
            plt_alpha = 0.02 + min(ovlp_val, 16) / 35
            import pdb; pdb.set_trace()

            ax.errorbar(
                x=i, y=sort_perf[i], yerr=err_arr[:, i].reshape(-1, 1),
                elinewidth=plt_size, fmt='o', ms=2 * plt_size,
                alpha=plt_alpha, color=plt_clr
                )

    plt.xlim(out_data.shape[0] / -125, out_data.shape[0] * (126/125))
    plt.ylim(-0.02, 1.02)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.grid(axis='x')
    ax.grid(axis='y', which='major', linewidth=3, alpha=0.6)

    plt.ylabel('AUC', fontsize=25)
    plt.axhline(color='r', y=0.5, xmin=-10, xmax=out_data.shape[0] + 10,
                linewidth=1, linestyle='--')

    fig.set_size_inches(out_data.shape[0] ** 0.9 / 125, 9)
    plt.savefig(os.path.join(plot_dir, plot_fl),
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
    """Creates plots for the given combination of cohort and classifier."""

    # parses command line arguments
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')
    args = parser.parse_args()

    # reads in experiment data, finds the genes with at least one mutation
    # sub-type that was tested during the search
    search_data = test_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'search'))
    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                             for mtype in search_data.index])

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=list(use_genes),
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', data_dir=firehose_dir, syn=syn, cv_prop=1.0
        )

    print('Plotting distribution of sub-type performance...')
    plot_auc_distribution(search_data, args)

    print('Plotting performance of single-gene sub-types...')
    gene_mtypes = plot_mtype_highlights(search_data, args,
                                        mtype_choice='Genes')

    print('Plotting performance of the best sub-types...')
    best_mtypes = plot_mtype_highlights(search_data, args,
                                        mtype_choice='Best')

    for gene_mtype in gene_mtypes:
        print('Plotting performance of sub-types related to {} ...'\
                .format(gene_mtype))
        plot_auc_distribution(search_data, args,
                              mtype_choice=gene_mtype, cdata=cdata)
        _ = plot_mtype_highlights(search_data, args, mtype_choice=gene_mtype)


if __name__ == '__main__':
    main()

