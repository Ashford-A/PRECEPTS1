
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'all')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import PanCancerMutCohort
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


def plot_all_clustering(trans_dict, args, cdata, pca_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=1, ncols=len(trans_dict),
                              figsize=(21, 7))
    fig.tight_layout(pad=1.6)

    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, trans_expr in enumerate(trans_dict):
        axarr[i].scatter(
            trans_expr[:, pca_comps[0]], trans_expr[:, pca_comps[1]],
            marker='o', s=8, c='0.5', alpha=0.2, edgecolor='none'
            )

    fig.savefig(os.path.join(plot_dir,
                             "clustering_comps_{}-{}.png".format(
                                 pca_comps[0], pca_comps[1])),
                dpi=200, bbox_inches='tight')

    plt.close()


def plot_gene_clustering(trans_dict, args, cdata, pca_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=1, ncols=len(trans_dict),
                              figsize=(21, 7))
    fig.tight_layout(pad=1.6)

    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    base_mtype = MuType({('Gene', args.gene): None})
    base_pheno = np.array(cdata.train_pheno(base_mtype))
    mut_clr = sns.light_palette((1/3, 0, 0), input="rgb",
                                n_colors=5, reverse=True)[1]

    for i, trans_expr in enumerate(trans_dict):
        axarr[i].scatter(trans_expr[~base_pheno, pca_comps[0]],
                         trans_expr[~base_pheno, pca_comps[1]],
                         marker='o', s=6, c='0.5', alpha=0.15,
                         edgecolor='none')

        axarr[i].scatter(trans_expr[base_pheno, pca_comps[0]],
                         trans_expr[base_pheno, pca_comps[1]],
                         marker='o', s=10, c=mut_clr, alpha=0.3,
                         edgecolor='none')

    fig.savefig(os.path.join(plot_dir,
                             "clustering-gene_{}__comps_{}-{}.png".format(
                                 args.gene, pca_comps[0], pca_comps[1])),
                dpi=200, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gene', type=str, help='a gene mutated in TCGA')
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = PanCancerMutCohort(mut_genes=[args.gene], mut_levels=['Gene'],
                               expr_source='Firehose', expr_dir=firehose_dir,
                               var_source='mc3', cv_prop=1.0, syn=syn)

    mut_trans = [('PCA', OmicPCA()), ('t-SNE', OmicTSNE()),
                 ('UMAP', OmicUMAP())]
    trans_dict = [(trs_lbl, trs.fit_transform_coh(cdata))
                  for trs_lbl, trs in mut_trans]

    plot_all_clustering(trans_dict, args, cdata)
    plot_gene_clustering(trans_dict, args, cdata)


if __name__ == "__main__":
    main()

