
import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots', 'comb')
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities import comb_output
from HetMan.features.variants import MuType
from HetMan.features.cohorts import VariantCohort

import numpy as np
import pandas as pd

from math import log
from scipy.stats import fisher_exact

import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from functools import reduce
from operator import add
from itertools import product, cycle

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def plot_comb_distribution(out_data, args, cdata, use_mtypes):

    fig, axarr = plt.subplots(nrows=len(use_mtypes), ncols=len(use_mtypes))
    fig.tight_layout(w_pad=-0.5, h_pad=-0.87)

    plot_fl = 'comb-infer_{}-{}.png'.format(args.cohort, args.classif)
    train_clrs = ['#672885', '#323E8A', '#C03344', '0.5']

    for mtype in use_mtypes:
        pi = use_mtypes.index(mtype)
        axarr[pi, pi].axis('off')

        axarr[pi, pi].add_patch(mpl.patches.Polygon(
            np.array([[-0.06, -0.06], [1.06, 1.06], [-0.06, 1.06]]),
            fill=True, facecolor='#C03344', alpha=0.17, clip_on=False
            ))
        axarr[pi, pi].add_patch(mpl.patches.Polygon(
            np.array([[-0.06, -0.06], [1.06, 1.06], [1.06, -0.06]]),
            fill=True, facecolor='#323E8A', alpha=0.17, clip_on=False
            ))
        
        axarr[pi, pi].text(x=0.5, y=0.55, s=mtype, size=33, weight='bold',
                           ha='center', va='center')

        axarr[pi, pi].text(
            x=0.5, y=0.37,
            s='{} mutated samples'.format(
                len(mtype.get_samples(cdata.train_mut))),
            size=17, ha='center', va='center'
            )

    for mtypes in out_data.index:
        plot_data = []
        pos_data = []

        plot_locx = use_mtypes.index(mtypes[0])
        plot_locy = use_mtypes.index(mtypes[1])
        px = min(plot_locx, plot_locy)
        py = max(plot_locx, plot_locy)

        mtype1_pheno = np.array(cdata.train_pheno(mtypes[0]))
        mtype2_pheno = np.array(cdata.train_pheno(mtypes[1]))

        both_stat = mtype1_pheno & mtype2_pheno
        mtype1_stat = mtype1_pheno & ~mtype2_pheno
        mtype2_stat = ~mtype1_pheno & mtype2_pheno
        neith_stat = ~mtype1_pheno & ~mtype2_pheno

        for (i, train), (j, stat) in product(
                enumerate(['Both', 'Mtype1', 'Mtype2', 'Diff']),
                enumerate([both_stat, mtype1_stat, mtype2_stat, neith_stat])):

            if not isinstance(out_data.loc[mtypes, train], float):
                pos_data += [i * 2.5 + j / 2]

                if np.sum(stat) < 3:
                    plot_data += [[]]
                else:
                    plot_data += [out_data.loc[mtypes, train][stat]]

        if plot_data:
            bplot = axarr[px, py].boxplot(
                x=plot_data, positions=pos_data, patch_artist=True,
                flierprops=dict(markersize=2),
                medianprops=dict(color='0.3', linewidth=3)
                )

            for patch, color in zip(bplot['boxes'], cycle(train_clrs)):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        if (py - px) == 1:
            axarr[px, py].set_xticks([0.75, 3.25, 5.75, 8.25])
            axarr[px, py].set_xticklabels(
                ['M1 & M2', 'M1 - M2', 'M2 - M1', 'M1 ^ M2'],
                size=9, ha='center'
                )

        else:
            axarr[px, py].xaxis.set_ticklabels([])

        if py == (len(use_mtypes) - 1):
            axarr[px, py].tick_params(axis='y', labelsize=15)
            axarr[px, py].yaxis.tick_right()

        else:
            axarr[px, py].yaxis.set_ticklabels([])

        axarr[px, py].set_xlim(-0.5, 9.5)
        axarr[px, py].set_ylim(-0.02, 1.02)
            
        axarr[px, py].grid(axis='x')
        axarr[px, py].grid(axis='y', which='major', linewidth=2, alpha=0.7)

        axarr[py, px].set_xlim(-2, len(cdata.samples) + 2)
        axarr[py, px].grid(b=True, axis='x', which='major',
                           linewidth=2, alpha=0.7)

        axarr[py, px].minorticks_on()
        axarr[py, px].grid(b=True, axis='x', which='minor',
                           linewidth=1, alpha=0.5)

        axarr[py, px].grid(axis='y')
        axarr[py, px].yaxis.set_major_formatter(plt.NullFormatter())

        axarr[py, px].add_patch(mpl.patches.Rectangle(
            (np.sum(neith_stat) / 2, 0.51), np.sum(mtype1_stat), 0.1,
            fill=True, facecolor='#323E8A', alpha=0.6
            ))

        axarr[py, px].add_patch(mpl.patches.Rectangle(
            (np.sum(mtype1_stat) + np.sum(neith_stat) / 2, 0.51),
            np.sum(both_stat), 0.1,
            fill=True, facecolor='#672885', alpha=0.6
            ))

        axarr[py, px].add_patch(mpl.patches.Rectangle(
            (np.sum(mtype1_stat) + np.sum(neith_stat) / 2, 0.39),
            np.sum(both_stat), 0.1,
            fill=True, facecolor='#672885', alpha=0.6
            ))

        axarr[py, px].add_patch(mpl.patches.Rectangle(
            (np.sum(mtype1_pheno) + np.sum(neith_stat) / 2, 0.39),
            np.sum(mtype2_stat), 0.1,
            fill=True, facecolor='#C03344', alpha=0.6
            ))

        ovlp_test = fisher_exact(
            [[np.sum(both_stat), np.sum(mtype1_stat)],
             [np.sum(mtype2_stat), np.sum(neith_stat)]],
            alternative='two-sided'
            )

        axarr[py, px].text(
            x=0, y=0.9,
            s='{: <8} log2 odds ratio\n$10^{{{: <8}}}$ pval'.format(
                str(round(log(ovlp_test[0], 2.0), 2)),
                str(round(log(ovlp_test[1], 10), 1))),
            size=13, ha='left', va='center'
            )

        if py == (len(use_mtypes) - 1):
            axarr[py, px].tick_params(axis='x', labelsize=15)
        else:
            axarr[py, px].xaxis.set_ticklabels([])


    fig.set_size_inches(17, 17)
    plt.savefig(os.path.join(plot_dir, plot_fl),
                dpi=600, bbox_inches='tight')
    plt.close()


def main():
    """Creates plots for the given combination of cohort and classifier."""

    # parses command line arguments
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')

    args = parser.parse_args()
    cross_df = comb_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'comb'))

    use_mtypes = sorted(list(
        reduce(lambda x, y: set(x) | set(y), tuple(cross_df.index))))
    use_genes = [mtype.subtype_list()[0][0] for mtype in use_mtypes]

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/mgrzad"
                                "/input-data/synapse")
    syn.login()

    cdata = VariantCohort(cohort=args.cohort, mut_genes=use_genes,
                          mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
                          expr_source='Firehose', data_dir=firehose_dir,
                          syn=syn, cv_prop=1.0)

    plot_comb_distribution(cross_df, args, cdata, use_mtypes)


if __name__ == '__main__':
    main()

