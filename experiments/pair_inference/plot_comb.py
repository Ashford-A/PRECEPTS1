
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
from itertools import combinations as combn

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def plot_comb_distribution(out_data, args, cdata, use_mtypes, use_which=None):

    if isinstance(use_which, tuple):
        if use_which[0] == 'freq':
            use_mtypes = sorted(
                use_mtypes,
                key=lambda x: len(x.get_samples(cdata.train_mut)),
                reverse=True
                )[:use_which[1]]
        which_lbl = '_' + '-'.join([str(x) for x in use_which])

    elif isinstance(use_which, set):
        use_mtypes = list(use_which)
        which_lbl = '_' + '-'.join([str(x) for x in use_which])

    else:
        use_mtypes = out_data.index
        which_lbl = ''

    fig, axarr = plt.subplots(nrows=len(use_mtypes), ncols=len(use_mtypes))
    fig.tight_layout(w_pad=-0.5, h_pad=-0.87)

    train_clrs = ['#672885', '#323E8A', '#C03344', '0.5']
    plot_fl = 'comb-infer_{}-{}{}.png'.format(
        args.cohort, args.classif, which_lbl)

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

    for (i, mtype1), (j, mtype2) in combn(enumerate(use_mtypes), 2):
        plot_data = []
        pos_data = []

        if (mtype1, mtype2) in out_data.index:
            mtypes = (mtype1, mtype2)
            mtype_lbls = ['Mtype1', 'Mtype2']

        else:
            mtypes = (mtype2, mtype1)
            mtype_lbls = ['Mtype2', 'Mtype1']

        mtype1_pheno = np.array(cdata.train_pheno(mtype1))
        mtype2_pheno = np.array(cdata.train_pheno(mtype2))

        both_stat = mtype1_pheno & mtype2_pheno
        mtype1_stat = mtype1_pheno & ~mtype2_pheno
        mtype2_stat = ~mtype1_pheno & mtype2_pheno
        neith_stat = ~mtype1_pheno & ~mtype2_pheno

        for (w, train), (v, stat) in product(
                enumerate(['Both'] + mtype_lbls + ['Diff']),
                enumerate([both_stat, mtype1_stat, mtype2_stat, neith_stat])):

            if not isinstance(out_data.loc[mtypes, train], float):
                pos_data += [w * 2.5 + v / 2]

                if np.sum(stat) < 3:
                    plot_data += [[]]
                else:
                    plot_data += [out_data.loc[mtypes, train][stat]]

        if plot_data:
            bplot = axarr[i, j].boxplot(
                x=plot_data, positions=pos_data, patch_artist=True,
                flierprops=dict(markersize=2),
                medianprops=dict(color='0.3', linewidth=3)
                )

            for patch, color in zip(bplot['boxes'], cycle(train_clrs)):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        if (j - i) == 1:
            axarr[i, j].set_xticks([0.75, 3.25, 5.75, 8.25])
            axarr[i, j].set_xticklabels(
                ['M1 & M2', 'M1 - M2', 'M2 - M1', 'M1 ^ M2'],
                size=9, ha='center'
                )

        else:
            axarr[i, j].xaxis.set_ticklabels([])

        if j == (len(use_mtypes) - 1):
            axarr[i, j].tick_params(axis='y', labelsize=15)
            axarr[i, j].yaxis.tick_right()

        else:
            axarr[i, j].yaxis.set_ticklabels([])

        axarr[i, j].set_xlim(-0.5, 9.5)
        axarr[i, j].set_ylim(-0.02, 1.02)
            
        axarr[i, j].grid(axis='x')
        axarr[i, j].grid(axis='y', which='major', linewidth=2, alpha=0.7)

        axarr[j, i].set_xlim(len(cdata.samples) * -0.01,
                             len(cdata.samples) * 1.01)
        axarr[j, i].grid(b=True, axis='x', which='major',
                         linewidth=2, alpha=0.7)

        if j == (len(use_mtypes) - 1):
            axarr[j, i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            axarr[j, i].tick_params(axis='x', which='major', labelsize=13)

        else:
            axarr[j, i].xaxis.set_ticklabels([])

        axarr[j, i].grid(b=True, axis='x', which='minor',
                         linewidth=1, alpha=0.5)
        axarr[j, i].minorticks_on()

        axarr[j, i].grid(axis='y')
        axarr[j, i].yaxis.set_major_formatter(plt.NullFormatter())

        axarr[j, i].add_patch(mpl.patches.Rectangle(
            (np.sum(neith_stat) / 2, 0.51), np.sum(mtype1_stat), 0.1,
            fill=True, facecolor='#323E8A', alpha=0.6
            ))

        axarr[j, i].add_patch(mpl.patches.Rectangle(
            (np.sum(mtype1_stat) + np.sum(neith_stat) / 2, 0.51),
            np.sum(both_stat), 0.1,
            fill=True, facecolor='#672885', alpha=0.6
            ))

        axarr[j, i].add_patch(mpl.patches.Rectangle(
            (np.sum(mtype1_stat) + np.sum(neith_stat) / 2, 0.39),
            np.sum(both_stat), 0.1,
            fill=True, facecolor='#672885', alpha=0.6
            ))

        axarr[j, i].add_patch(mpl.patches.Rectangle(
            (np.sum(mtype1_pheno) + np.sum(neith_stat) / 2, 0.39),
            np.sum(mtype2_stat), 0.1,
            fill=True, facecolor='#C03344', alpha=0.6
            ))

        ovlp_test = fisher_exact(
            [[np.sum(both_stat), np.sum(mtype1_stat)],
             [np.sum(mtype2_stat), np.sum(neith_stat)]],
            alternative='two-sided'
            )

        axarr[j, i].text(
            x=0, y=0.9,
            s='{: <8} log2 odds ratio\n$10^{{{: <8}}}$ pval'.format(
                str(round(log(ovlp_test[0], 2.0), 2)),
                str(round(log(ovlp_test[1], 10), 1))),
            size=13, ha='left', va='center'
            )

    fig_inch = len(use_mtypes) * 3.4
    fig.set_size_inches(fig_inch, fig_inch)

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

    plot_comb_distribution(cross_df, args, cdata, use_mtypes, ('freq', 7))


if __name__ == '__main__':
    main()

