
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'ordering')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output, simil_cmap

import numpy as np
import pandas as pd

from scipy.spatial import distance
from scipy.cluster import hierarchy

import argparse
import synapseclient
from itertools import product

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def get_similarities(iso_df, base_gene, cdata):
    base_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', base_gene): None})))

    simil_df = pd.DataFrame(index=iso_df.index, columns=iso_df.index,
                            dtype=np.float)
    auc_list = pd.Series(index=iso_df.index, dtype=np.float)

    for cur_mtype, other_mtype in product(iso_df.index, repeat=2):
        none_vals = np.concatenate(iso_df.loc[
            cur_mtype, ~base_pheno].values)

        cur_pheno = np.array(cdata.train_pheno(cur_mtype))
        other_pheno = np.array(cdata.train_pheno(other_mtype))

        if cur_mtype == other_mtype:
            simil_df.loc[cur_mtype, other_mtype] = 1.0
            cur_vals = np.concatenate(iso_df.loc[cur_mtype, cur_pheno].values)
            auc_list[cur_mtype] = np.less.outer(none_vals, cur_vals).mean()

        else:
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

    return simil_df, auc_list


def plot_singleton_ordering(simil_df, auc_list, args, cdata):
    singl_mtypes = [mtype for mtype in simil_df.index
                    if len(mtype.subkeys()) == 1]
    simil_df = simil_df.loc[singl_mtypes, singl_mtypes]

    fig, ax = plt.subplots(figsize=(2.1 + len(singl_mtypes) * 0.84,
                                    1.0 + len(singl_mtypes) * 0.81))

    annot_df = pd.DataFrame(-1.0, index=singl_mtypes, columns=singl_mtypes)
    for singl_mtype in singl_mtypes:
        annot_df.loc[singl_mtype, singl_mtype] = auc_list[singl_mtype]

    annot_df = annot_df.applymap('{:.3f}'.format)
    annot_df[annot_df == '-1.000'] = ''
    annot_df = annot_df.loc[simil_df.index, simil_df.columns]

    xlabs = ['{} ({})'.format(mtype, len(mtype.get_samples(cdata.train_mut)))
             for mtype in simil_df.columns]
    ylabs = [repr(mtype).replace(' WITH ', '\n')
             for mtype in simil_df.index]

    # draw the heatmap
    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1.0, vmax=2.0,
                     xticklabels=xlabs, yticklabels=ylabs, square=True,
                     annot=annot_df, fmt='', annot_kws={'size': 14})

    # configure the tick labels on the colourbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1.0, 0.0, 1.0, 2.0])
    cbar.set_ticklabels(['M2 < WT', 'M2 = WT', 'M2 = M1', 'M2 > M1'])
    cbar.ax.tick_params(labelsize=13) 

    # configure the tick labels on the heatmap proper
    plt.xticks(rotation=40, ha='right', size=12)
    plt.yticks(size=13)

    plt.xlabel('M2: Testing Mutation (# of samples)',
               size=19, weight='semibold')
    plt.ylabel('M1: Training Mutation', size=19, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, "singleton_ordering__{}_{}__{}__samps_{}__{}.png".format(
            args.cohort, args.gene, args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def plot_all_ordering(simil_df, auc_list, args, cdata):
    row_linkage = hierarchy.linkage(
        distance.pdist(simil_df, metric='cityblock'), method='centroid')

    gr = sns.clustermap(
        simil_df, cmap=simil_cmap, figsize=(16, 13), vmin=-1.0, vmax=2.0,
        row_linkage=row_linkage, col_linkage=row_linkage,
        )

    gr.ax_heatmap.set_xticks([])
    gr.cax.set_visible(False)

    plt.savefig(os.path.join(
        plot_dir, "all_ordering__{}_{}__{}__samps_{}__{}.png".format(
            args.cohort, args.gene, args.classif,
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
    parser.add_argument('mut_levels', default='Form_base__Exon',
                        help='a set of mutation annotation levels')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=[args.gene],
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    simil_df, auc_list = get_similarities(load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.gene, args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        ),
        args.gene, cdata)

    simil_rank = simil_df.mean(axis=1) - simil_df.mean(axis=0)
    simil_order = simil_rank.sort_values().index
    simil_df = simil_df.loc[simil_order, reversed(simil_order)]

    plot_singleton_ordering(simil_df.copy(), auc_list.copy(), args, cdata)
    plot_all_ordering(simil_df.copy(), auc_list.copy(), args, cdata)


if __name__ == '__main__':
    main()

