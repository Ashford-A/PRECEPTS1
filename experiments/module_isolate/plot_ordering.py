
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'module_isolate')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'ordering')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.module_isolate import *
from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.experiments.subvariant_isolate.utils import compare_scores
from HetMan.experiments.utilities import load_infer_output, simil_cmap

import argparse
import synapseclient
import pandas as pd

from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt


def plot_singleton_ordering(simil_df, auc_list, size_list, args):
    singl_mtypes = [mtypes for mtypes in simil_df.index
                    if all(len(mtype.subkeys()) == 1 for mtype in mtypes)]

    fig, ax = plt.subplots(figsize=(11, 10))
    simil_df = simil_df.loc[singl_mtypes, singl_mtypes]
    annot_df = pd.DataFrame(0.0, index=singl_mtypes, columns=singl_mtypes)

    for mtypes in singl_mtypes:
        annot_df.loc[mtypes, mtypes] = auc_list[mtypes]

    annot_df = annot_df.applymap('{:.2f}'.format).applymap(
        lambda x: ('' if x == '0.00' else '1.0' if x == '1.00'
                   else x.lstrip('0'))
        )

    xlabs = [str(mtypes[0]) if len(mtypes) == 1
             else ' & '.join(str(mtype) for mtype in sorted(mtypes))
             for mtypes in singl_mtypes]
    xlabs = ['{}  ({})'.format(xlab, size_list[mtypes])
             for xlab, mtypes in zip(xlabs, singl_mtypes)]
 
    ylabs = ['ONLY\n{}'.format(repr(mtypes[0])).replace(' WITH ', '\nWITH ')
             if len(mtypes) == 1
             else '\nAND '.join(repr(mtype) for mtype in sorted(mtypes))
             for mtypes in singl_mtypes]
 
    xlabs = [xlab.replace('Point:', '') for xlab in xlabs]
    xlabs = [xlab.replace('Copy:', '') for xlab in xlabs]
    ylabs = [ylab.replace('Scale IS Point WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('Scale IS Copy WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('\nWITH Scale IS Point', '') for ylab in ylabs]
    ylabs = [ylab.replace('\nWITH Scale IS Copy', '') for ylab in ylabs]

    # draw the heatmap
    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1., vmax=2.,
                     xticklabels=xlabs, yticklabels=ylabs, square=True,
                     annot=annot_df, fmt='', annot_kws={'size': 15})
 
    # configure the tick labels on the colourbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels([
        'M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'])
    cbar.ax.tick_params(labelsize=17)

    # configure the tick labels on the heatmap proper
    plt.xticks(rotation=26, ha='right', size=12)
    plt.yticks(size=9)

    # configure the titles of the axes
    plt.xlabel('M2: Testing Mutation (# of samples)',
               size=23, weight='semibold')
    plt.ylabel('M1: Training Mutation', size=23, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, "singleton_ordering__{}__{}__{}__samps_{}__{}.png".format(
            args.cohort, '_'.join(sorted(args.genes)), args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def plot_singleton_clustering(simil_df, auc_list, size_list, args):
    singl_mtypes = [mtypes for mtypes in simil_df.index
                    if all(len(mtype.subkeys()) == 1 for mtype in mtypes)]
    fig_size = 5. + len(singl_mtypes) * 0.43

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    simil_df = simil_df.loc[singl_mtypes, singl_mtypes]

    row_order = dendrogram(linkage(distance.pdist(
        simil_df, metric='cityblock'), method='centroid'))['leaves']
    simil_df = simil_df.iloc[row_order, row_order]

    annot_df = simil_df.copy()
    annot_df[annot_df < 3.] = 0.0
    for mtypes in singl_mtypes:
        annot_df.loc[mtypes, mtypes] = auc_list[mtypes]

    annot_df = annot_df.applymap('{:.2f}'.format).applymap(
        lambda x: ('' if x == '0.00' else '1.0' if x == '1.00'
                   else x.lstrip('0'))
        )
 
    xlabs = [str(mtypes[0]) if len(mtypes) == 1
             else ' & '.join(str(mtype) for mtype in sorted(mtypes))
             for mtypes in singl_mtypes]
    xlabs = ['{}  ({})'.format(xlab, size_list[mtypes])
             for xlab, mtypes in zip(xlabs, singl_mtypes)]

    ylabs = ['ONLY\n{}'.format(repr(mtypes[0])).replace(' WITH ', '\nWITH ')
             if len(mtypes) == 1
             else '\nAND '.join(repr(mtype) for mtype in sorted(mtypes))
             for mtypes in singl_mtypes]

    xlabs = [xlab.replace('Point:', '') for xlab in xlabs]
    xlabs = [xlab.replace('Copy:', '') for xlab in xlabs]
    ylabs = [ylab.replace('Scale IS Point WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('Scale IS Copy WITH ', '') for ylab in ylabs]
    ylabs = [ylab.replace('\nScale IS Point\nWITH', '\n') for ylab in ylabs]
    ylabs = [ylab.replace('\nScale IS Copy\nWITH', '\n') for ylab in ylabs]

    # draw the heatmap
    ax = sns.heatmap(simil_df, cmap=simil_cmap, vmin=-1., vmax=2.,
                     xticklabels=xlabs, yticklabels=ylabs, square=True,
                     annot=annot_df, fmt='', annot_kws={'size': 14})

    # configure the tick labels on the colourbar
    ax.collections = [ax.collections[-1]]
    cbar = ax.collections[-1].colorbar
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels([
        'M2 < WT', 'M2 = WT', 'WT < M2 < M1', 'M2 = M1', 'M2 > M1'])
    cbar.ax.tick_params(labelsize=17)

    # configure the tick labels on the heatmap proper
    plt.xticks(rotation=27, ha='right', size=13)
    plt.yticks(size=8)

    plt.xlabel("M2: Testing Mutation (# of samples)",
               size=22, weight='semibold')
    plt.ylabel("M1: Training Mutation", size=25, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, "singleton_clustering__{}__{}__{}__samps_{}__{}.png".format(
            args.cohort, '_'.join(sorted(args.genes)), args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def plot_all_clustering(simil_df, auc_list, args):
    use_mtypes = auc_list[auc_list > 0.7].index

    simil_df = simil_df.loc[use_mtypes, use_mtypes]
    simil_df.index = [str(mtypes[0]) if len(mtypes) == 1
                      else ' & '.join(str(mtype) for mtype in sorted(mtypes))
                      for mtypes in simil_df.index]

    simil_df.index = [xlab.replace('Point:', '') for xlab in simil_df.index]
    simil_df.index = [xlab.replace('Copy:', '') for xlab in simil_df.index]

    # cluster the subtypes based on similarity, plot the clustering
    row_linkage = linkage(
        distance.pdist(simil_df, metric='cityblock'), method='centroid')
    gr = sns.clustermap(
        simil_df, cmap=simil_cmap, figsize=(12, 11), vmin=-1., vmax=2.,
        row_linkage=row_linkage, col_linkage=row_linkage, square=True
        )

    gr.ax_heatmap.set_xticks([])
    gr.cax.set_visible(False)

    plt.savefig(os.path.join(
        plot_dir, "all_clustering__{}__{}__{}__samps_{}__{}.png".format(
            args.cohort, '_'.join(sorted(args.genes)), args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the ordering of the subtypes of a module of genes in a given "
        "cohort based on how their isolated expression signatures classify "
        "one another."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', type=str,
                        help='a set of mutation annotation levels')
    parser.add_argument('genes', type=str, nargs='+',
                        help='a list of mutated genes')
    parser.add_argument('--samp_cutoff', type=int, default=20)

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=args.genes,
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=expr_dir,
                           var_source='mc3', copy_source='Firehose',
                           domain_dir=domain_dir, annot_file=annot_file,
                           syn=syn, cv_prop=1.0)

    simil_df, auc_list, size_list = compare_scores(
        load_infer_output(
            os.path.join(base_dir, 'output',
                         args.cohort, '_'.join(sorted(args.genes)),
                         args.classif, 'samps_{}'.format(args.samp_cutoff),
                         args.mut_levels)
            ), cdata
        )

    simil_rank = simil_df.mean(axis=1) - simil_df.mean(axis=0)
    simil_order = [
        mtypes for mtypes, _ in sorted(
            tuple(simil_rank.iteritems()),
            key=lambda k: (k[0][0].subtype_list()[0][0], k[1])
            )
        ]

    simil_df = simil_df.loc[simil_order, simil_order[::-1]]
    plot_singleton_ordering(
        simil_df.copy(), auc_list.copy(), size_list.copy(), args)
    plot_singleton_clustering(
        simil_df.copy(), auc_list.copy(), size_list.copy(), args)
    plot_all_clustering(simil_df.copy(), auc_list.copy(), args)


if __name__ == '__main__':
    main()

