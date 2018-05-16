
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'subtypes')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.describe.transformers import *

import synapseclient
import argparse
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def plot_subtype_clustering(trans_expr, args, cdata, use_gene,
                            pca_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=3, ncols=4, figsize=(18, 13),
                              sharex=True, sharey=True)
    fig.tight_layout(pad=2.4, w_pad=2.1, h_pad=5.4)

    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pca_comps = np.array(pca_comps)
    trans_expr = trans_expr[:, pca_comps]
    mut_clr = sns.light_palette((1/3, 0, 0), input="rgb",
                                n_colors=5, reverse=True)[1]

    base_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', use_gene): None})))
    axarr[0, 0].set_title(use_gene, size=23)

    axarr[0, 0].scatter(
        trans_expr[~base_pheno, 0], trans_expr[~base_pheno, 1],
        marker='o', s=14, c='0.4', alpha=0.25, edgecolor='none'
        )

    axarr[0, 0].scatter(
        trans_expr[base_pheno, 0], trans_expr[base_pheno, 1],
        marker='o', s=45, c=mut_clr, alpha=0.4, edgecolor='none'
        )

    plot_mtypes = {
        MuType({('Gene', use_gene): mtype})
        for mtype in cdata.train_mut[use_gene].branchtypes(min_size=20)
        }

    plot_phenos = sorted([(mtype, np.array(cdata.train_pheno(mtype)))
                          for mtype in plot_mtypes],
                         key=lambda x: np.sum(x[1]), reverse=True)

    if len(plot_mtypes) < 11:
        comb_mtypes = {
            MuType({('Gene', use_gene): mtype})
            for mtype in cdata.train_mut[use_gene].combtypes(
                min_type_size=25, comb_sizes=(2, ))
            }

        plot_phenos += sorted([(mtype, np.array(cdata.train_pheno(mtype)))
                               for mtype in comb_mtypes],
                              key=lambda x: np.sum(x[1]), reverse=True
                             )[:(11 - len(plot_mtypes))]

    for ax, (mtype, pheno) in zip(axarr.reshape(-1)[1:], plot_phenos[:11]):
        ax.set_title(
            repr(mtype).replace(' WITH ', '\n').replace(' OR ', '\n'),
            size=16
            )

        ax.scatter(trans_expr[~pheno, 0], trans_expr[~pheno, 1],
                   marker='o', s=14, c='0.4', alpha=0.25, edgecolor='none')
        ax.scatter(trans_expr[pheno, 0], trans_expr[pheno, 1],
                   marker='o', s=45, c=mut_clr, alpha=0.4, edgecolor='none')

    fig.text(0.5, 0.02, 'Component {}'.format(pca_comps[0] + 1),
             size=24, weight='semibold', ha='center')
    fig.text(0.02, 0.5, 'Component {}'.format(pca_comps[1] + 1),
             size=24, weight='semibold', va='center', rotation='vertical')

    fig.savefig(os.path.join(
        plot_dir, "{}_clustering_comps_{}-{}__{}_{}__levels__{}.png".format(
            args.cohort, pca_comps[0], pca_comps[1],
            use_gene, args.transform, args.mut_levels,
            )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the clustering done by an unsupervised learning method on a "
        "TCGA cohort with subtypes of particular genes highlighted."
        )

    parser.add_argument('cohort', type=str, help='a cohort in TCGA')
    parser.add_argument('transform', type=str,
                        help='an unsupervised learning method')
    parser.add_argument('mut_levels', type=str,
                        help='a set of mutation annotation levels')
    parser.add_argument('--genes', type=str, nargs='+', default=['TP53'],
                        help='a list of mutated genes')

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=args.genes,
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           cv_prop=1.0, syn=syn)

    mut_trans = eval(args.transform)()
    trans_expr = mut_trans.fit_transform_coh(cdata)

    for gene in args.genes:
        plot_subtype_clustering(trans_expr.copy(), args, cdata, gene)


if __name__ == "__main__":
    main()

