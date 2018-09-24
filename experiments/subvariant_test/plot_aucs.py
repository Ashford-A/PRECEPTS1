
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_test import *
from HetMan.experiments.utilities.isolate_mutype_test import load_acc
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType

import numpy as np
import pandas as pd

import argparse
import synapseclient
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
type_pal = sns.color_palette('Dark2', n_colors=17)


def plot_basic(out_auc, args, cdata):
    fig, ax = plt.subplots(figsize=(3 + out_auc.shape[0] * 0.17, 9))
    flier_props = dict(marker='o', markerfacecolor='black', markersize=4,
                       markeredgecolor='none', alpha=0.4)

    out_auc.index = [
        '{}  ({})'.format(re.sub("^Copy:", "", str(mtype)),
                          len(mtype.get_samples(cdata.train_mut)))
        for mtype in out_auc.index
        ]

    sns.boxplot(data=out_auc.transpose(), palette=type_pal, linewidth=1.6,
                boxprops=dict(alpha=0.68), flierprops=flier_props)
    plt.axhline(color='#550000', y=0.5, linewidth=3.7, alpha=0.37)

    plt.ylabel('AUC', fontsize=23, weight='semibold')
    plt.xticks(rotation=31, ha='right', size=11)
    plt.yticks(np.linspace(0.3, 1, 8), size=15)
    ax.tick_params(axis='y', length=8, width=2.3)

    fig.savefig(
        os.path.join(plot_dir,
                     'basic__{}-{}_{}_levels-{}__samps-{}.png'.format(
            args.cohort, args.gene, args.classif,
            args.mut_levels, args.samp_cutoff
            )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_iso_aucs(out_auc, basic_clrs, args, cdata):
    mpl.rcParams['axes.linewidth'] = 2.6
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, ax = plt.subplots(figsize=(13, 14))

    mtype_sizes = [
        131 * len(mtype.get_samples(cdata.train_mut)) / len(cdata.train_mut)
        for mtype in out_auc.index
        ]

    type_mrks = ['s' if len(mtype.subtype_list()) == 2 else 'o'
                 for mtype in out_auc.index]
    type_clrs = [basic_clrs[mtype] if mtype in basic_clrs else 'black'
                 for mtype in out_auc.index]

    base_aucs = out_auc['Base'].quantile(q=0.25, axis=1)
    iso_aucs = out_auc['Iso'].quantile(q=0.25, axis=1)

    for base_auc, iso_auc, type_mrk, mtype_size, type_clr in zip(
            base_aucs, iso_aucs, type_mrks, mtype_sizes, type_clrs):
        ax.scatter(base_auc, iso_auc,
                   marker=type_mrk, s=mtype_size, c=type_clr, alpha=0.19)

    ax.plot([-1, 2], [-1, 2],
            linewidth=2.3, linestyle='--', color='#550000', alpha=0.53)

    ax.set_xlim(min(base_aucs.min(), iso_aucs.min()) - 0.02, 1)
    ax.set_ylim(min(base_aucs.min(), iso_aucs.min()) - 0.02, 1)
    ax.set_xlabel('Standard AUC', fontsize=23, weight='semibold')
    ax.set_ylabel('Isolated AUC', fontsize=23, weight='semibold')
    ax.set_ylim(min(base_aucs.min(), iso_aucs.min()) - 0.02, 1)

    fig.savefig(
        os.path.join(plot_dir, 'iso__{}-{}_{}_levels-{}__samps-{}.png'.format(
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
    use_lvls = ['Gene'] + args.mut_levels.split('__')

    out_auc, out_aupr = load_acc(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.classif,
        'samps_{}'.format(args.samp_cutoff), args.mut_levels
        ))

    basic_mtypes = sorted(
        mtype for mtype in out_auc.index if len(mtype.subkeys()) == 1
        or (mtype & MuType({('Scale', 'Point'): None})).is_empty()
        )
    basic_clrs = {mtype: clr for mtype, clr in zip(basic_mtypes, type_pal)}

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=use_lvls,
        expr_source='Firehose', var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=expr_dir, copy_dir=copy_dir,
        cv_prop=1.0, syn=syn
        )

    plot_basic(out_auc.copy().loc[basic_mtypes], args, cdata)
    plot_iso_aucs(out_auc.copy(), basic_clrs, args, cdata)


if __name__ == '__main__':
    main()

