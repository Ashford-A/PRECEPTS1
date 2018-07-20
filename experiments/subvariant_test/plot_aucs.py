
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.subvariant_test import firehose_dir, syn_root

from HetMan.experiments.utilities.isolate_mutype_test import load_acc
from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
import synapseclient
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_aucs(base_auc, out_auc, args, cdata):
    mpl.rcParams['axes.linewidth'] = 2.6
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, ax = plt.subplots(figsize=(13, 14))

    mtype_sizes = [
        131 * len(mtype.get_samples(cdata.train_mut)) / len(cdata.train_mut)
        for mtype in out_auc.index
        ]

    norm_aucs = out_auc['Base'].quantile(q=0.25, axis=1)
    iso_aucs = out_auc['Iso'].quantile(q=0.25, axis=1)
    ax.scatter(norm_aucs, iso_aucs, s=mtype_sizes, c='black', alpha=0.19)

    ax.plot([-1, 2], [-1, 2],
            linewidth=2.3, linestyle='--', color='#550000', alpha=0.53)
    ax.plot([-1, 2], [base_auc.quantile(q=0.25)] * 2,
            linewidth=2.1, linestyle=':', color='black', alpha=0.37)
    ax.plot([base_auc.quantile(q=0.25)] * 2, [-1, 2],
            linewidth=2.1, linestyle=':', color='black', alpha=0.37)

    ax.set_xlim(min(norm_aucs.min(), iso_aucs.min()) - 0.02, 1)
    ax.set_ylim(min(norm_aucs.min(), iso_aucs.min()) - 0.02, 1)
    ax.set_xlabel('Standard AUC', fontsize=23, weight='semibold')
    ax.set_ylabel('Isolated AUC', fontsize=23, weight='semibold')
    ax.set_yticks(list(ax.get_yticks()) + [base_auc.quantile(q=0.25)])
    ax.set_ylim(min(norm_aucs.min(), iso_aucs.min()) - 0.02, 1)

    fig.savefig(
        os.path.join(plot_dir, '{}-{}_{}_levels-{}__samps-{}.png'.format(
            args.cohort, args.gene, args.classif,
            args.mut_levels, args.samp_cutoff
            )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance and tuning characteristics of a model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Form_base__Exon',
                        help='a set of mutation annotation levels')
    parser.add_argument('--samp_cutoff', type=int, default=20)

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    base_auc, out_auc, _, _ = load_acc(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.classif,
        'samps_{}'.format(args.samp_cutoff), args.mut_levels
        ))

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=[args.gene],
                           mut_levels=args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    plot_aucs(base_auc.copy(), out_auc.copy(), args, cdata)


if __name__ == '__main__':
    main()

