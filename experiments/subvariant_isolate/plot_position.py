
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'position')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import iso_output

import argparse
import synapseclient

import numpy as np
import pandas as pd
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
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 18),
                                   sharex=True, sharey=False,
                                   gridspec_kw={'height_ratios': [1, 3.41]})

    base_mtype = MuType({('Gene', args.gene): None})
    base_pheno = np.array(cdata.train_pheno(base_mtype))

    without_phenos = {mtype: np.array(cdata.train_pheno(mtype))
                      for mtype in cdata.train_mut.branchtypes(min_size=20)
                      if mtype != base_mtype}

    within_mtypes = [MuType({('Gene', args.gene): mtype})
                     for mtype in cdata.train_mut[args.gene].combtypes(
                         comb_sizes=(1, 2), min_type_size=10)
                     if (mtype & prob_series.name).is_empty()]

    within_phenos = {mtype: np.array(cdata.train_pheno(mtype))
                     for mtype in within_mtypes}

    cur_pheno = np.array(cdata.train_pheno(prob_series.name))
    cur_diff = (np.mean(prob_series[cur_pheno])
                - np.mean(prob_series[~base_pheno]))

    sns.kdeplot(prob_series[~base_pheno], ax=ax1, cut=0,
                color='0.4', alpha=0.45, linewidth=2.8,
                gridsize=250, shade=True,
                label='{} Wild-Type'.format(args.gene))
    sns.kdeplot(prob_series[cur_pheno], ax=ax1, cut=0,
                color=(0.267, 0.137, 0.482), alpha=0.45, linewidth=2.8,
                bw=0.02, gridsize=250, shade=True,
                label='{} Mutant'.format(prob_series.name))
    sns.kdeplot(prob_series[base_pheno & ~cur_pheno], ax=ax1, cut=0,
                color=(0.698, 0.329, 0.616), alpha=0.3, linewidth=1.0,
                bw=0.02, gridsize=250, shade=True,
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
    without_tests = {mtype: tests for mtype, tests in without_tests.items()
                     if tests['pval'] < 0.1 and tests['diff'] > 0}

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
    within_tests = {mtype: tests for mtype, tests in within_tests.items()
                    if tests['pval'] < 0.1} 

    if len(without_tests) > 8:
        without_tests = dict(sorted(without_tests.items(),
                                    key=lambda x: x[1]['pval'])[:8])
    if len(within_tests) > 8:
        within_tests = dict(sorted(within_tests.items(),
                                   key=lambda x: x[1]['pval'])[:8])

    subtype_df = pd.concat(
        [pd.DataFrame({'Mtype': repr(mtype).replace(' WITH ', '\n'),
                       'Type': '{} Wild-Type'.format(args.gene),
                       'Scores': prob_series[~base_pheno
                                             & without_phenos[mtype]]})
         for mtype, tests in without_tests.items()]
        + [pd.DataFrame({'Mtype': repr(mtype).replace(' WITH ', '\n'),
                         'Type': '{} Mutants'.format(args.gene),
                         'Scores': prob_series[base_pheno
                                               & within_phenos[mtype]]})
           for mtype, tests in within_tests.items()]
        )

    sns.violinplot(
        data=subtype_df, x='Scores', y='Mtype', hue='Type', cut=0,
        palette={'{} Wild-Type'.format(args.gene): '0.5',
                 '{} Mutants'.format(args.gene): (0.812, 0.518, 0.745)},
        alpha=0.3, linewidth=1.3, bw=0.1, gridsize=500, legend=False
        )

    ax2.set_ylabel('Mutation Type', size=23, weight='semibold')
    ax2.yaxis.set_tick_params(labelsize=11)

    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.set_xlabel('Inferred {} Score'.format(prob_series.name),
                   size=23, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, "{}_positions__{}_{}__{}__levels__{}.png".format(
            re.sub('/|\.|:', '_', str(prob_series.name)),
            args.cohort, args.gene, args.classif, args.mut_levels,
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

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    prob_df = iso_output(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.classif,
        'samps_{}'.format(args.samp_cutoff), args.mut_levels
        )).applymap(np.mean)

    # logs into Synapse using locally-stored credentials
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

