
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
    """Plots the range of AUCs across all tested sub-types.
    """

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


def plot_mtype_highlights(out_data, args, mtype_choice='Genes', cdata=None):
    """Compares the AUCs for a given subset of tested sub-types.
    """

    fig, ax = plt.subplots(nrows=1, ncols=1)
    use_data = out_data.apply(sorted, axis=1)
    mtype_means = use_data.mean(axis=1)

    # highlights subsets and supersets of a given sub-type
    if isinstance(mtype_choice, MuType):
        choice_samps = mtype_choice.get_samples(cdata.train_mut)

        # finds the tested sub-types that are subsets of the given sub-type
        use_mtypes = sorted([mtype for mtype in use_data.index
                             if mtype_choice.is_supertype(mtype)
                             and mtype != mtype_choice],
                            key=lambda x: mtype_means[x]) + [mtype_choice]

        # subset sub-types are given a background shade corresponding to how
        # many samples they contain
        for i, mtype in enumerate(use_mtypes[:-1]):
            sub_val = (len(mtype.get_samples(cdata.train_mut))
                       / len(choice_samps))

            clr_s = sub_val ** 1.5
            plt_a = 0.07 + sub_val / 5.5
            plt_clr = colorsys.hls_to_rgb(240 / 360, 0.5, clr_s)

            ax.add_patch(mpl.patches.Rectangle(
                (i - 0.5, -0.5), 1, 2,
                fill=True, edgecolor='none', facecolor=plt_clr,
                alpha=plt_a, zorder=500
                ))

        sub_indx = len(use_mtypes)
        plt_clr = colorsys.hls_to_rgb(240 / 360, 0.5, 1.0)
        ax.add_patch(mpl.patches.Rectangle(
            (sub_indx - 1.5, -0.5), 1, 2,
            fill=True, lw=2, color=plt_clr, linestyle=':',
            alpha=0.3, zorder=500
            ))

        # finds the tested sub-types that are supersets of the given sub-type
        use_mtypes += sorted([mtype for mtype in use_data.index
                              if mtype.is_supertype(mtype_choice)
                              and mtype != mtype_choice],
                             key=lambda x: mtype_means[x], reverse=True)

        # superset sub-types are given a background shade corresponding to the
        # degree of overlap between the given sub-type and the other sub-type
        for i, mtype in enumerate(use_mtypes[sub_indx:]):
            cur_samps = (mtype - mtype_choice).get_samples(cdata.train_mut)
            both_samps = choice_samps & cur_samps
            or_samps = choice_samps | cur_samps

            ovlp_test = fisher_exact(
                [[len(both_samps), len(cur_samps - both_samps)],
                 [len(choice_samps - both_samps),
                  len(cdata.samples - or_samps)]],
                alternative='two-sided'
                )

            ovlp_val = log(max(ovlp_test[0], 20 ** -6), 20)
            ovlp_val *= 2 * (ovlp_test[1] > 0.5) - 1

            if ovlp_val > 6:
                ovlp_val = 6
            elif ovlp_val < -6:
                ovlp_val = -6

            clr_h = 240 + (6 - ovlp_val) * 10
            clr_s = abs(ovlp_val) / 6
            plt_a = 0.07 + abs(ovlp_val) / 33
            plt_clr = colorsys.hls_to_rgb(clr_h / 360, 0.5, clr_s)

            ax.add_patch(mpl.patches.Rectangle(
                (i + sub_indx - 0.5, -0.5), 1, 2,
                fill=True, edgecolor='none', facecolor=plt_clr,
                alpha=plt_a, zorder=500
                ))

    # highlights the sub-types corresponding to mutations of a single gene
    elif mtype_choice == 'Genes':
        use_mtypes = [
            mtype for mtype in use_data.index
            if len(mtype) == 1 and mtype.subtype_list()[0][1] is None
            ]
        use_mtypes = sorted(use_mtypes,
                            key=lambda x: mtype_means[x], reverse=True)

    # highlights the top twenty sub-types by average performance
    elif mtype_choice == 'Best':
        use_mtypes = mtype_means.sort_values(ascending=False)[:20].index

    else:
        raise ValueError("Unrecognized mtype_choice argument!")

    use_data = use_data.loc[use_mtypes, :]
    plot_min = use_data.min().min()
    plot_gap = (1 - plot_min) / 31

    # plots the median AUC and the top/bottom AUCs for each sub-type across
    # the cross-validation folds used in testing
    ax.plot(use_data.iloc[:, 2].values,
            '_', color='#9CADB5', ms=9, mew=2.5)
    ax.plot(use_data.iloc[:, 0].values,
            '.', color='black', ms=4)
    ax.plot(use_data.iloc[:, 4].values,
            '.', color='black', ms=4)

    # plots a box between the 20th and 80th percentiles of performance for
    # as well as the number of mutated samples for each sub-type
    for i, mtype in enumerate(use_mtypes):
        ax.add_patch(mpl.patches.Rectangle(
            (i - 0.15, use_data.iloc[i, 1]),
            0.3, use_data.iloc[i, 3] - use_data.iloc[i, 1],
            fill=True, color='0.3', alpha=0.2, zorder=1000
            ))

        ax.text(x=i, y=1,
                s='({})'.format(len(mtype.get_samples(cdata.train_mut))),
                fontsize=8, rotation=-45, ha='center')

    ax.text(x=(len(use_mtypes) - 1) / 2, y=1 + plot_gap, s='Samples',
            fontsize=11, ha='center')

    plt.xticks(np.arange(len(use_mtypes)), use_mtypes,
               fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=13)

    plt.xlim(-0.41, len(use_mtypes) - 0.59)
    plt.ylim(use_data.min().min() - plot_gap, 1 + plot_gap * 2)

    plt.axhline(color='r', y=0.5, xmin=-1, xmax=len(use_mtypes) + 1,
                linewidth=0.9, linestyle='--')
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
                                        mtype_choice='Genes', cdata=cdata)

    print('Plotting performance of the best sub-types...')
    best_mtypes = plot_mtype_highlights(search_data, args,
                                        mtype_choice='Best', cdata=cdata)

    for gene_mtype in gene_mtypes:
        print('Plotting performance of sub-types related to {} ...'\
                .format(gene_mtype))

        plot_auc_distribution(search_data, args,
                              mtype_choice=gene_mtype, cdata=cdata)
        _ = plot_mtype_highlights(search_data, args,
                                  mtype_choice=gene_mtype, cdata=cdata)


if __name__ == '__main__':
    main()

