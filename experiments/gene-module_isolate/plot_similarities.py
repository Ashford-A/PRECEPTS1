
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'similarities')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output, simil_cmap

import numpy as np
import pandas as pd
from functools import reduce
from operator import or_

import argparse
import synapseclient
from itertools import product

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def get_aucs(iso_df, base_pheno, cdata):
    auc_list = pd.Series(index=iso_df.index, dtype=np.float)

    for mtype in iso_df.index:
        none_vals = np.concatenate(iso_df.loc[mtype, ~base_pheno].values)
        cur_pheno = np.array(cdata.train_pheno(mtype))
        cur_vals = np.concatenate(iso_df.loc[mtype, cur_pheno].values)
        auc_list[mtype] = np.less.outer(none_vals, cur_vals).mean()

    return auc_list


def get_similarities(iso_df, base_pheno, cdata):
    simil_df = pd.DataFrame(index=iso_df.index, columns=iso_df.index,
                            dtype=np.float)

    for cur_mtype, other_mtype in product(iso_df.index, repeat=2):
        if cur_mtype != other_mtype:
            none_vals = np.concatenate(iso_df.loc[
                cur_mtype, ~base_pheno].values)
            
            cur_pheno = np.array(cdata.train_pheno(cur_mtype))
            other_pheno = np.array(cdata.train_pheno(other_mtype))

            if not np.any(~cur_pheno & other_pheno):
                cur_vals = np.concatenate(iso_df.loc[
                    cur_mtype, cur_pheno & ~other_pheno].values)
                other_vals = np.concatenate(iso_df.loc[
                    cur_mtype, other_pheno].values)

            elif not np.any(cur_pheno & ~other_pheno):
                cur_vals = np.concatenate(iso_df.loc[
                    cur_mtype, cur_pheno].values)
                other_vals = np.concatenate(iso_df.loc[
                    cur_mtype, ~cur_pheno & other_pheno].values)

            else:
                cur_vals = np.concatenate(iso_df.loc[
                    cur_mtype, cur_pheno & ~other_pheno].values)
                other_vals = np.concatenate(iso_df.loc[
                    cur_mtype, ~cur_pheno & other_pheno].values)

            other_none_prob = np.greater.outer(none_vals, other_vals).mean()
            other_cur_prob = np.greater.outer(other_vals, cur_vals).mean()
            cur_none_prob = np.greater.outer(none_vals, cur_vals).mean()
            
            simil_df.loc[cur_mtype, other_mtype] = (
                (other_cur_prob - other_none_prob) / (0.5 - cur_none_prob))

    return simil_df


def plot_gene_ordering(simil_df, auc_list, args, cdata):
    fig, ax = plt.subplots(figsize=(16, 14))

    simil_rank = simil_df.mean(axis=1) - simil_df.mean(axis=0)
    simil_order = simil_rank.sort_values().index
    simil_df = simil_df.loc[simil_order, reversed(simil_order)].fillna(1.0)
    
    gene_sorted = sorted(simil_df.index,
                         key=lambda mtype: mtype.subtype_list()[0][0])
    simil_df = simil_df.loc[gene_sorted, gene_sorted]

    annot_df = pd.DataFrame(-1.0,
                            index=simil_df.index, columns=simil_df.columns)
    for mtype in annot_df.index:
        annot_df.loc[mtype, mtype] = auc_list[mtype]

    annot_df = annot_df.applymap('{:.3f}'.format)
    annot_df[annot_df == '-1.000'] = ''

    xlabs = ['{} ({})'.format(mtype, len(mtype.get_samples(cdata.train_mut)))
             for mtype in simil_df.columns]
    ylabs = [repr(mtype).replace(' WITH ', '\n')
             for mtype in simil_df.index]

    # draw the heatmap
    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1.0, vmax=2.0,
                     xticklabels=xlabs, yticklabels=ylabs, square=True,
                     annot=annot_df, fmt='', annot_kws={'size': 11})

    gene_seps = [i for i, mtype in enumerate(simil_df.index[:-1])
                 if (mtype.subtype_list()[0][0]
                     != simil_df.index[i + 1].subtype_list()[0][0])]

    for i in gene_seps:
        ax.axhline(y=i + 1, xmin=-0.5, xmax=1.5, c='0.59',
                   linewidth=3.7, linestyle=':')
        ax.axvline(x=i + 1, ymin=-0.5, ymax=1.5, c='0.59',
                   linewidth=3.7, linestyle=':')

    # configure the tick labels on the colourbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1.0, 0.0, 1.0, 2.0])
    cbar.set_ticklabels(['M2 < WT', 'M2 = WT', 'M2 = M1', 'M2 > M1'])
    cbar.ax.tick_params(labelsize=21) 

    # configure the tick labels on the heatmap proper
    plt.xticks(rotation=38, ha='right', size=17)
    plt.yticks(size=10)

    plt.xlabel('M2: Testing Mutation (# of samples)',
               size=28, weight='semibold')
    plt.ylabel('M1: Training Mutation', size=28, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir,
                     "gene-ordering__{}__{}__{}__samps_{}__{}.png".format(
                         args.cohort, '_'.join(sorted(args.genes)),
                         args.classif, args.samp_cutoff, args.mut_levels
                        )
                    ),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the ordering of the simplest subtypes within a module of genes "
        "in a given cohort based on how their isolated expression signatures "
        "classify one another."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', type=str,
                        help='a set of mutation annotation levels')
    parser.add_argument('genes', type=str, nargs='+',
                        help='a list of mutated genes')
    parser.add_argument('--samp_cutoff', type=int, default=25)

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=args.genes,
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    infer_df = load_infer_output(
        os.path.join(base_dir, 'output', args.cohort,
                     '_'.join(sorted(args.genes)), args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        )

    base_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', tuple(args.genes)): None})))
    auc_list = get_aucs(
        infer_df, base_pheno, cdata).sort_values(ascending=False)
    auc_list = auc_list[auc_list > 0.6]

    mtype_lens = {mtype: len(mtype.subkeys()) for mtype in auc_list.index}
    mtype_list = sorted(auc_list.index, key=lambda mtype: mtype_lens[mtype])

    mtype_genes = {mtype: mtype.subtype_list()[0][0]
                   for mtype in auc_list.index}
    mtype_samps = {mtype: mtype.get_samples(cdata.train_mut)
                   for mtype in auc_list.index}

    plot_mtypes = reduce(or_,
                         [set([mtype for mtype in mtype_list
                               if mtype_genes[mtype] == gene][:3])
                          for gene in args.genes])

    ovlp_threshold = 0.5
    i = j = 1
    while len(plot_mtypes) <= 15:
        ovlp_score = min(
            len(mtype_samps[mtype_list[i]] ^ mtype_samps[plot_mtype])
            / max(len(mtype_samps[mtype_list[i]]),
                  len(mtype_samps[plot_mtype]))
            for plot_mtype in plot_mtypes
            )
        
        if ovlp_score >= ovlp_threshold:
            plot_mtypes |= {mtype_list[i]}

        i += 1
        if i >= len(mtype_list):
            j += 1
            i = j
            ovlp_threshold **= 4/3

    simil_df = get_similarities(infer_df.loc[plot_mtypes, :],
                                base_pheno, cdata)
    plot_gene_ordering(simil_df, auc_list, args, cdata)


if __name__ == '__main__':
    main()

