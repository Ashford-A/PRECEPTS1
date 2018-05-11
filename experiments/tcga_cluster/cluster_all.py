
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'all')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import PanCancerMutCohort
from HetMan.features.mutations import MuType
from HetMan.describe.transformers import *
from HetMan.experiments.utilities.pcawg_colours import cohort_clrs

import synapseclient
import argparse
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def plot_all_clustering(trans_dict, args, cdata, use_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=1, ncols=len(trans_dict),
                              figsize=(21, 7))
    fig.tight_layout(pad=1.6)

    # extracts the given pair of components from each transformed dataset
    use_comps = np.array(use_comps)
    trans_dict = [(trs_lbl, trans_expr[:, use_comps])
                  for trs_lbl, trans_expr in trans_dict]

    # turn off the axis tick labels
    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, (trs_lbl, trans_expr) in enumerate(trans_dict):
        axarr[i].set_title(trs_lbl, size=24, weight='semibold')

        # plot all the TCGA points
        axarr[i].scatter(trans_expr[:, 0], trans_expr[:, 1],
                         marker='o', s=8, c='0.5', alpha=0.2,
                         edgecolor='none')

    fig.savefig(os.path.join(plot_dir,
                             "clustering_comps_{}-{}.png".format(
                                 use_comps[0], use_comps[1])),
                dpi=300, bbox_inches='tight')

    plt.close()


def plot_cohort_clustering(trans_dict, args, cdata, use_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=1, ncols=len(trans_dict),
                              figsize=(21, 7))
    fig.tight_layout(pad=1.6)

    samp_list = sorted(cdata.samples)
    use_comps = np.array(use_comps)
    trans_dict = [(trs_lbl, trans_expr[:, use_comps])
                  for trs_lbl, trans_expr in trans_dict]

    # turn off the axis tick labels
    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i, (trs_lbl, trans_expr) in enumerate(trans_dict):
        axarr[i].set_title(trs_lbl, size=24, weight='semibold')

        for cohort, samps in cdata.cohort_samps.items():
            samp_indx = np.array([samp_list.index(samp) for samp in samps])

            if cohort in cohort_clrs:
                use_clr = cohort_clrs[cohort]
            else:
                use_clr = '0.5'

            axarr[i].scatter(
                trans_expr[samp_indx, 0], trans_expr[samp_indx, 1],
                marker='o', s=10, c=use_clr, alpha=0.16, edgecolor='none'
                )

    fig.savefig(os.path.join(plot_dir,
                             "clustering-cohort_comps_{}-{}.png".format(
                                 use_comps[0], use_comps[1])),
                dpi=300, bbox_inches='tight')

    plt.close()


def plot_gene_clustering(trans_dict, args, cdata, use_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=1, ncols=len(trans_dict),
                              figsize=(21, 7))
    fig.tight_layout(pad=1.6)

    # extracts the given pair of components from each transformed dataset
    use_comps = np.array(use_comps)
    trans_dict = [(trs_lbl, trans_expr[:, use_comps])
                  for trs_lbl, trans_expr in trans_dict]

    # turn off the axis tick labels
    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    base_mtype = MuType({('Gene', args.gene): None})
    base_pheno = np.array(cdata.train_pheno(base_mtype))
    mut_clr = sns.light_palette((1/3, 0, 0), input="rgb",
                                n_colors=5, reverse=True)[1]

    for i, (trs_lbl, trans_expr) in enumerate(trans_dict):
        axarr[i].set_title(trs_lbl, size=24, weight='semibold')

        # plot the wild-type points
        axarr[i].scatter(
            trans_expr[~base_pheno, 0], trans_expr[~base_pheno, 1],
            marker='o', s=6, c='0.5', alpha=0.15, edgecolor='none'
            )

        # plot the mutated points
        axarr[i].scatter(
            trans_expr[base_pheno, 0], trans_expr[base_pheno, 1],
            marker='o', s=10, c=mut_clr, alpha=0.3, edgecolor='none'
            )

    fig.savefig(os.path.join(plot_dir,
                             "clustering-gene_{}__comps_{}-{}.png".format(
                                 args.gene, use_comps[0], use_comps[1])),
                dpi=300, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=("Plots the clusters returned by running a set of "
                     "unsupervised learning methods on all TCGA cohorts "
                     "concatenated together.")
        )

    # parses command line arguments, creates the directory where the
    # plots will be saved
    parser.add_argument('--gene', type=str, default='TP53',
                        help='a gene mutated in TCGA')
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # load RNAseq and mutation call data for all TCGA cohorts
    cdata = PanCancerMutCohort(mut_genes=[args.gene], mut_levels=['Gene'],
                               expr_source='Firehose', expr_dir=firehose_dir,
                               var_source='mc3', cv_prop=1.0, syn=syn)

    # create the pipelines for unsupervised learning
    mut_trans = [('PCA', OmicPCA()),
                 ('t-SNE', OmicTSNE()),
                 ('UMAP', OmicUMAP())]

    # apply the pipelines to the TCGA pan-cancer dataset
    trans_dict = [(trs_lbl, trs.fit_transform_coh(cdata))
                  for trs_lbl, trs in mut_trans]

    plot_all_clustering(trans_dict, args, cdata)
    plot_cohort_clustering(trans_dict, args, cdata)
    plot_gene_clustering(trans_dict, args, cdata)


if __name__ == "__main__":
    main()

