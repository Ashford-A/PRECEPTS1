
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_infer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'similarity')

from HetMan.experiments.subvariant_infer.setup_infer import Mcomb, ExMcomb
from HetMan.experiments.subvariant_infer.merge_infer import merge_cohort_data
from HetMan.experiments.subvariant_infer.utils import compare_scores
from HetMan.experiments.utilities import simil_cmap

import argparse
from pathlib import Path
import dill as pickle
import bz2

import numpy as np
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ordering(simil_df, auc_list, pheno_dict, args):
    fig_size = 5. + simil_df.shape[0] * 0.43
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    simil_rank = simil_df.mean(axis=1) - simil_df.mean(axis=0)
    simil_order = simil_rank.sort_values().index
    simil_df = simil_df.loc[simil_order, simil_order]

    annot_df = simil_df.copy()
    annot_df[annot_df < 3.] = 0.0
    for mcomb in simil_df.index:
        annot_df.loc[mcomb, mcomb] = auc_list.loc[mcomb, 'Iso']

    annot_df = annot_df.applymap('{:.2f}'.format).applymap(
        lambda x: ('' if x == '0.00' else '1.0' if x == '1.00'
                   else x.lstrip('0'))
        )

    xlabs = ['{}  ({})'.format(mcomb, np.sum(pheno_dict[mcomb]))
             for mcomb in simil_df.index]
    ylabs = [repr(mcomb).replace('ONLY ', '').replace(' AND ', '\nAND\n')
             for mcomb in simil_df.index]

    xlabs = [xlab.replace('Point:', '') for xlab in xlabs]
    xlabs = [xlab.replace('Copy:', '') for xlab in xlabs]
    ylabs = [ylab.replace('Scale IS Point WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('Scale IS Copy WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('\nScale IS Point\nWITH', '\n') for ylab in ylabs]
    ylabs = [ylab.replace('\nScale IS Copy\nWITH', '\n') for ylab in ylabs]

    # draw the heatmap
    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1., vmax=2.,
                     xticklabels=xlabs, yticklabels=ylabs, square=True,
                     annot=annot_df, fmt='', annot_kws={'size': 14},
                     rasterized=True)

    # configure the tick labels on the colourbar
    cbar = ax.collections[-1].colorbar
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels([
        'M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'])
    cbar.ax.tick_params(labelsize=17)

    # configure the tick labels on the heatmap proper
    plt.xticks(rotation=27, ha='right', size=13)
    plt.yticks(size=10)

    plt.xlabel("M2: Testing Mutation (# of samples)",
               size=24, weight='semibold')
    plt.ylabel("M1: Training Mutation", size=26, weight='semibold')

    plt.savefig(os.path.join(plot_dir, '__'.join([args.cohort, args.gene]),
                             "ordering__{}__{}.svg".format(args.mut_levels,
                                                           args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_clustering(simil_df, auc_list, pheno_dict, args):
    fig_size = 5. + simil_df.shape[0] * 0.43
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    row_order = dendrogram(linkage(distance.pdist(
        simil_df, metric='cityblock'), method='centroid'))['leaves']
    simil_df = simil_df.iloc[row_order, row_order]

    annot_df = simil_df.copy()
    annot_df[annot_df < 3.] = 0.0
    for mcomb in simil_df.index:
        annot_df.loc[mcomb, mcomb] = auc_list.loc[mcomb, 'Iso']

    annot_df = annot_df.applymap('{:.2f}'.format).applymap(
        lambda x: ('' if x == '0.00' else '1.0' if x == '1.00'
                   else x.lstrip('0'))
        )
 
    xlabs = ['{}  ({})'.format(mcomb, np.sum(pheno_dict[mcomb]))
             for mcomb in simil_df.index]
    ylabs = [repr(mcomb).replace('ONLY ', '').replace(' AND ', '\nAND\n')
             for mcomb in simil_df.index]

    xlabs = [xlab.replace('Point:', '') for xlab in xlabs]
    xlabs = [xlab.replace('Copy:', '') for xlab in xlabs]
    ylabs = [ylab.replace('Scale IS Point WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('Scale IS Copy WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('\nScale IS Point\nWITH', '\n') for ylab in ylabs]
    ylabs = [ylab.replace('\nScale IS Copy\nWITH', '\n') for ylab in ylabs]

    # draw the heatmap
    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1., vmax=2.,
                     xticklabels=xlabs, yticklabels=ylabs, square=True,
                     annot=annot_df, fmt='', annot_kws={'size': 14},
                     rasterized=True)

    # configure the tick labels on the colourbar
    ax.collections = [ax.collections[-1]]
    cbar = ax.collections[-1].colorbar
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels([
        'M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'])
    cbar.ax.tick_params(labelsize=17)

    # configure the tick labels on the heatmap proper
    plt.xticks(rotation=27, ha='right', size=13)
    plt.yticks(size=10)

    plt.xlabel("M2: Testing Mutation (# of samples)",
               size=22, weight='semibold')
    plt.ylabel("M1: Training Mutation", size=25, weight='semibold')

    plt.savefig(os.path.join(plot_dir, '__'.join([args.cohort, args.gene]),
                             "clustering__{}__{}.svg".format(args.mut_levels,
                                                             args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the ordering of a gene's subtypes in a given cohort based on "
        "how their isolated expression signatures classify one another."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Location__Protein',
                        help='a set of mutation annotation levels')

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.cohort, args.gene])),
                exist_ok=True)
    out_fl = "out-data__{}__{}.p.gz".format(args.mut_levels, args.classif)

    use_ctf = min(
        int(out_file.parts[-3].split('__samps-')[1])
        for out_file in Path(base_dir).glob(os.path.join(
            "{}__samps-*".format(args.cohort), args.gene, out_fl))
        )

    out_tag = "{}__samps-{}".format(args.cohort, use_ctf)
    cdata = merge_cohort_data(os.path.join(base_dir, out_tag, args.gene),
                              args.mut_levels, use_seed=709)

    with bz2.BZ2File(os.path.join(base_dir, out_tag, args.gene, out_fl),
                     'r') as fl:
        iso_df = pickle.load(fl)['Infer']['Iso']

    use_mtypes = [mcomb for mcomb in iso_df.index
                  if isinstance(mcomb, ExMcomb)]

    if len(use_mtypes) > 12:
        use_mtypes = [mcomb for mcomb in use_mtypes
                      if all(len(mtype.subkeys()) == 1
                             for mtype in mcomb.mtypes)]

    pheno_dict, auc_list, simil_df = compare_scores(iso_df.loc[use_mtypes],
                                                    cdata)

    plot_ordering(simil_df.copy(), auc_list.copy(), pheno_dict.copy(), args)
    plot_clustering(simil_df.copy(), auc_list.copy(), pheno_dict.copy(), args)


if __name__ == '__main__':
    main()

