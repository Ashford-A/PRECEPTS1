
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'mut_freq')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutFreqCohort
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


def plot_freq_clustering(trans_dict, args, cdata, pca_comps=(0, 1)):
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
    fig.tight_layout(pad=1.8, h_pad=1.3)

    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plot_pheno = np.log10(np.array(cdata.train_pheno()))
    pheno_range = np.percentile(plot_pheno, q=(10, 90))
    freq_cmap = sns.diverging_palette(145, 280, s=77, l=71,
                                      center='dark', as_cmap=True)

    pca_comps = np.array(pca_comps)
    trans_dict = [(trs_lbl, trans_expr[:, pca_comps])
                  for trs_lbl, trans_expr in trans_dict]

    for i, (trs_lbl, trans_expr) in enumerate(trans_dict):
        axarr[i].set_title(trs_lbl, size=24, weight='semibold')

        axarr[i].scatter(trans_expr[:, 0], trans_expr[:, 1], c=plot_pheno,
                         cmap=freq_cmap, vmin=pheno_range[0],
                         vmax=pheno_range[1],
                         marker='o', s=42, alpha=0.19, edgecolor='none')

    fig.savefig(os.path.join(
        plot_dir, "{}_clustering_comps_{}-{}.png".format(
            args.cohort, pca_comps[0], pca_comps[1]
            )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cohort', type=str, help='a cohort in TCGA')
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutFreqCohort(cohort=args.cohort, expr_source='Firehose',
                          expr_dir=firehose_dir, cv_prop=1.0, syn=syn)

    mut_trans = [('PCA', OmicPCA()), ('t-SNE', OmicTSNE()),
                 ('UMAP', OmicUMAP())]
    trans_dict = [(trs_lbl, trs.fit_transform_coh(cdata))
                  for trs_lbl, trs in mut_trans]

    plot_freq_clustering(trans_dict, args, cdata)


if __name__ == "__main__":
    main()

