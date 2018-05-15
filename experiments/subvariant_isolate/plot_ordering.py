
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'ordering')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import iso_output, simil_cmap

import numpy as np
import pandas as pd

import argparse
import synapseclient
from itertools import product

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def get_similarities(iso_df, args, cdata):
    base_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', args.gene): None})))

    simil_df = pd.DataFrame(index=iso_df.index, columns=iso_df.index,
                            dtype=np.float)
    annot_df = pd.DataFrame(index=iso_df.index, columns=iso_df.index,
                            dtype=np.float)

    for cur_mtype, other_mtype in product(iso_df.index, repeat=2):
        cur_pheno = np.array(cdata.train_pheno(cur_mtype))
        other_pheno = np.array(cdata.train_pheno(other_mtype))

        if cur_mtype == other_mtype:
            rel_prob = 1.0

            none_vals = np.concatenate(
                iso_df.loc[cur_mtype, ~base_pheno].values)
            cur_vals = np.concatenate(iso_df.loc[cur_mtype, cur_pheno].values)

            auc_val = np.less.outer(none_vals, cur_vals).mean()

        else:
            auc_val = -1.0
            none_vals = np.concatenate(iso_df.loc[
                cur_mtype, ~cur_pheno & ~other_pheno & ~base_pheno].values)
            
            if not np.any(~cur_pheno & other_pheno):
                other_vals = np.concatenate(iso_df.loc[
                    cur_mtype, other_pheno].values)
                cur_vals = np.concatenate(iso_df.loc[
                    cur_mtype, cur_pheno & ~other_pheno].values)

            elif not np.any(cur_pheno & ~other_pheno):
                other_vals = np.concatenate(iso_df.loc[
                    cur_mtype, ~cur_pheno & other_pheno].values)
                cur_vals = np.concatenate(iso_df.loc[
                    cur_mtype, cur_pheno].values)

            else:
                other_vals = np.concatenate(iso_df.loc[
                    cur_mtype, ~cur_pheno & other_pheno].values)
                cur_vals = np.concatenate(iso_df.loc[
                    cur_mtype, cur_pheno & ~other_pheno].values)

            other_none_prob = np.greater.outer(none_vals, other_vals).mean()
            other_cur_prob = np.greater.outer(other_vals, cur_vals).mean()
            cur_none_prob = np.greater.outer(none_vals, cur_vals).mean()
            
            rel_prob = (
                (other_cur_prob - other_none_prob) / (0.5 - cur_none_prob))

        simil_df.loc[cur_mtype, other_mtype] = rel_prob
        annot_df.loc[cur_mtype, other_mtype] = auc_val

    return simil_df, annot_df


def plot_singleton_ordering(iso_df, args, cdata):
    singl_types = [mtype for mtype in iso_df.index
                   if len(mtype.subkeys()) == 1]

    fig, ax = plt.subplots(figsize=(2.1 + len(singl_types) * 0.84,
                                    1.0 + len(singl_types) * 0.81))
    simil_df, annot_df = get_similarities(
        iso_df.loc[singl_types, :], args, cdata)

    annot_df = annot_df.applymap('{:.3f}'.format)
    annot_df[annot_df == '-1.000'] = ''

    simil_rank = simil_df.mean(axis=1) - simil_df.mean(axis=0)
    simil_order = simil_rank.sort_values().index
    simil_df = simil_df.loc[simil_order, reversed(simil_order)]
    annot_df = annot_df.loc[simil_order, reversed(simil_order)]

    xlabs = ['{} ({})'.format(mtype, len(mtype.get_samples(cdata.train_mut)))
             for mtype in simil_df.columns]
    ylabs = [repr(mtype).replace(' WITH ', '\n')
             for mtype in simil_df.index]

    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1.0, vmax=2.0,
                     xticklabels=xlabs, yticklabels=ylabs,
                     annot=annot_df, fmt='', annot_kws={'size': 16})

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1.0, 0.0, 1.0, 2.0])
    cbar.set_ticklabels(['M2 < WT', 'M2 = WT', 'M2 = M1', 'M2 > M1'])

    plt.xticks(rotation=40, ha='right', size=17)
    plt.yticks(size=14)
    plt.xlabel('Testing Mutation (# of samples)', size=20, weight='semibold')
    plt.ylabel('Training Mutation', size=22, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, "singleton_ordering__{}_{}__{}__samps_{}__{}.png".format(
            args.cohort, args.gene, args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def plot_comb_ordering(iso_df, args, cdata, max_comb=2):
    comb_types = [mtype for mtype in iso_df.index
                  if len(mtype.subkeys()) <= max_comb]

    fig, ax = plt.subplots(figsize=(1.68 + len(comb_types) * 0.089,
                                    1.0 + len(comb_types) * 0.081))
    simil_df, _ = get_similarities(iso_df.loc[comb_types, :], args, cdata)

    simil_rank = simil_df.mean(axis=1) - simil_df.mean(axis=0)
    simil_order = simil_rank.sort_values().index
    simil_df = simil_df.loc[simil_order, reversed(simil_order)]

    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1.0, vmax=2.0,
                     xticklabels=False, yticklabels=True)

    plt.xticks(rotation=40, ha='right', size=7)
    plt.yticks(size=5)
    plt.ylabel('Training Mutation', size=22, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, "comb{}_ordering__{}_{}__{}__samps_{}__{}.png".format(
            max_comb, args.cohort, args.gene, args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the ordering of a gene's subtypes in a given cohort based on "
        "how their isolated expression signatures classify one another."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Form_base__Exon')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    iso_df = iso_output(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.classif,
        'samps_{}'.format(args.samp_cutoff), args.mut_levels
        ))

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=[args.gene],
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    plot_singleton_ordering(iso_df.copy(), args, cdata)
    plot_comb_ordering(iso_df.copy(), args, cdata, max_comb=2)


if __name__ == '__main__':
    main()

