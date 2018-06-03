
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'position')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output

import numpy as np
import pandas as pd

import argparse
import synapseclient

from scipy.stats import ks_2samp
import re

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"
copy_dir = "/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose"


def plot_mtype_positions(prob_series, args, cdata):
    kern_bw = (np.max(prob_series) - np.min(prob_series)) / 29

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 18),
                                   sharex=True, sharey=False,
                                   gridspec_kw={'height_ratios': [1, 3.41]})

    base_mtype = MuType({('Gene', args.gene): None})
    cur_mtype = MuType({('Gene', args.gene): prob_series.name})
    base_pheno = np.array(cdata.train_pheno(base_mtype))
    cur_pheno = np.array(cdata.train_pheno(cur_mtype))

    without_phenos = {
        mtype: np.array(cdata.train_pheno(mtype))
        for mtype in cdata.train_mut.branchtypes(min_size=args.samp_cutoff)
        if (mtype & base_mtype).is_empty()
        }

    within_mtypes = {MuType({('Gene', args.gene): mtype})
                     for mtype in cdata.train_mut[args.gene].combtypes(
                         comb_sizes=(1, 2), min_type_size=args.samp_cutoff)
                     if (mtype & prob_series.name).is_empty()}

    within_phenos = {mtype: np.array(cdata.train_pheno(mtype))
                     for mtype in within_mtypes}

    cur_diff = (np.mean(prob_series[cur_pheno])
                - np.mean(prob_series[~base_pheno]))

    sns.kdeplot(prob_series[~base_pheno], ax=ax1, cut=0,
                color='0.4', alpha=0.45, linewidth=2.8,
                bw=kern_bw, gridsize=250, shade=True,
                label='{} Wild-Type'.format(args.gene))
    sns.kdeplot(prob_series[cur_pheno], ax=ax1, cut=0,
                color=(0.267, 0.137, 0.482), alpha=0.45, linewidth=2.8,
                bw=kern_bw, gridsize=250, shade=True,
                label='{} Mutant'.format(prob_series.name))
    sns.kdeplot(prob_series[base_pheno & ~cur_pheno], ax=ax1, cut=0,
                color=(0.698, 0.329, 0.616), alpha=0.3, linewidth=1.0,
                bw=kern_bw, gridsize=250, shade=True,
                label='Other {} Mutants'.format(args.gene))

    ax1.set_ylabel('Density', size=23, weight='semibold')
    ax1.yaxis.set_tick_params(labelsize=14)

    without_tests = {
        mtype: {
            'pval': ks_2samp(prob_series[~base_pheno & ~pheno],
                             prob_series[~base_pheno & pheno]).pvalue,
            'diff': (np.mean(prob_series[~base_pheno & pheno])
                     - np.mean(prob_series[~base_pheno & ~pheno]))
            }
        for mtype, pheno in without_phenos.items()
        }

    without_tests = sorted(
        [(mtype, tests) for mtype, tests in without_tests.items()
         if tests['pval'] < 0.05 and tests['diff'] > 0],
        key=lambda x: x[1]['pval']
        )[:8]

    within_tests = {
        mtype: {
            'pval': ks_2samp(
                prob_series[base_pheno & ~cur_pheno & ~pheno],
                prob_series[base_pheno & ~cur_pheno & pheno]).pvalue,
            'diff': (np.mean(prob_series[base_pheno & ~cur_pheno & pheno])
                     - np.mean(prob_series[base_pheno & ~cur_pheno & ~pheno]))
            }
        for mtype, pheno in within_phenos.items()
        }

    within_tests = sorted(
        [(mtype, tests) for mtype, tests in within_tests.items()
         if tests['pval'] < 0.1],
        key=lambda x: x[1]['pval']
        )[:8]

    subtype_df = pd.concat(
        [pd.DataFrame({'Mtype': repr(mtype).replace(' WITH ', '\n'),
                       'Type': '{} Wild-Type'.format(args.gene),
                       'Scores': prob_series[~base_pheno
                                             & without_phenos[mtype]]})
         for mtype, tests in without_tests]
        + [pd.DataFrame(
            {'Mtype': repr(mtype).replace(
                'Gene IS {}'.format(args.gene), '').replace(' WITH ', '\n'),
                'Type': '{} Mutants'.format(args.gene),
                'Scores': prob_series[base_pheno & within_phenos[mtype]]}
            )
            for mtype, tests in within_tests]
        )

    plt_order = subtype_df.groupby(
        ['Mtype'])['Scores'].mean().sort_values().index
    subtype_df['Mtype'] = subtype_df['Mtype'].astype(
        'category').cat.reorder_categories(plt_order)

    sns.violinplot(
        data=subtype_df, x='Scores', y='Mtype', hue='Type',
        palette={'{} Wild-Type'.format(args.gene): '0.5',
                 '{} Mutants'.format(args.gene): (0.812, 0.518, 0.745)},
        alpha=0.3, linewidth=1.3, bw=kern_bw, dodge=False,
        cut=0, gridsize=500, legend=False
        )

    ax2.set_ylabel('Mutation Type', size=23, weight='semibold')
    ax2.yaxis.set_tick_params(labelsize=12)

    ax2.xaxis.set_tick_params(labelsize=18)
    ax2.set_xlabel('Inferred {} Score'.format(prob_series.name),
                   size=23, weight='semibold')

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir,
                     args.cohort, args.gene,
                     "{}_positions__{}_{}__{}__levels__{}.png".format(
                         re.sub('/|\.|:', '_', str(prob_series.name)),
                         args.cohort, args.gene, args.classif,
                         args.mut_levels
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot experiment results for given mutation classifier.')

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Form_base__Exon')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.cohort, args.gene), exist_ok=True)

    prob_df = load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.gene, args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        ).applymap(np.mean)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=None, samp_cutoff=20,
        mut_levels=['Gene'] + args.mut_levels.split('__'),
        expr_source='Firehose', expr_dir=firehose_dir, syn=syn, cv_prop=1.0
        )

    singl_mtypes = [mtype for mtype in prob_df.index
                    if len(mtype.subkeys()) == 1]

    for singl_mtype in singl_mtypes:
        plot_mtype_positions(prob_df.loc[singl_mtype, :], args, cdata)


if __name__ == '__main__':
    main()

