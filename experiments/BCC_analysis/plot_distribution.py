
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'distribution')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.BCC_analysis import *
from HetMan.experiments.BCC_analysis.cohorts import CancerCohort
from dryadic.features.mutations import MuType
from HetMan.experiments.BCC_analysis.fit_gene_models import load_output

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from operator import itemgetter

import argparse
import synapseclient
import subprocess

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

wt_clr = '0.29'
mut_clrs = sns.light_palette('#C50000', reverse=True)
test_genes = ['TP53', 'KRAS', 'SMAD4']


def plot_label_distribution(infer_vals, args, cdata):
    samp_list = cdata.subset_samps()
    fig, axarr = plt.subplots(figsize=(15, 14),
                              nrows=1, ncols=len(test_genes), sharey=True)

    for ax, gene in zip(axarr, test_genes):
        infer_means = np.apply_along_axis(
            lambda vals: np.mean(np.concatenate(vals)), 1, infer_vals[gene])
 
        tcga_means = pd.Series(
            {samp: val for samp, val in zip(samp_list, infer_means)
             if 'TCGA' in samp}
            )
        
        bcc_means = sorted(
            [(samp, val) for samp, val in zip(samp_list, infer_means)
             if 'TCGA' not in samp],
            key=itemgetter(1)
            )

        if np.all(infer_means >= 0):
            plt_ymin, plt_ymax = -0.08, max(np.max(infer_means) * 1.09, 1)

        else:
            plt_ymax = np.max([np.max(np.absolute(infer_means)) * 1.09, 1.1])
            plt_ymin = -plt_ymax

        plt.ylim(plt_ymin, plt_ymax)
        lbl_pad = (plt_ymax - plt_ymin) / 79

        use_mtype = MuType({('Gene', gene): None})
        mtype_stat = np.array(cdata.train_mut.status(tcga_means.index))
        kern_bw = (plt_ymax - plt_ymin) / 47

        sns.kdeplot(tcga_means[~mtype_stat], color=wt_clr, vertical=True,
                    shade=True, alpha=0.36, linewidth=0, bw=kern_bw, cut=0,
                    gridsize=1000, ax=ax)

        sns.kdeplot(tcga_means[mtype_stat], color=mut_clrs[0], vertical=True,
                    shade=True, alpha=0.36, linewidth=0, bw=kern_bw, cut=0,
                    gridsize=1000, ax=ax)

        for i, (patient, val) in enumerate(bcc_means):
            ax.axhline(y=val, xmin=0, xmax=0.09,
                       c='blue', ls='--', lw=2.3)

            if i > 0 and bcc_means[i - 1][1] > (val - lbl_pad):
                txt_va = 'bottom'

            elif (i < (len(bcc_means) - 1)
                  and bcc_means[i + 1][1] < (val + lbl_pad)):
                txt_va = 'top'

            else:
                txt_va = 'center'

            ax.text(1.1, val, patient,
                    size=11, ha='left', va=txt_va)

        ax.set_xticklabels([])
        ax.legend([Patch(color=mut_clrs[0], alpha=0.36),
                   Patch(color=wt_clr, alpha=0.36)],
                  ["TCGA-PAAD {} Mutants".format(gene),
                   "TCGA-PAAD Wild-Types"],
                  fontsize=17, loc=8, ncol=1)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'labels.png'),
                dpi=300, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the predicted mutation scores for a given cohort of SMMART "
        "samples against the distribution of scores for the matching cohort "
        "of TCGA samples."
        )

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    out_list = load_output()
    infer_vals = pd.concat([pd.DataFrame(ols['Infer']) for ols in out_list],
                           axis=1)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = CancerCohort(mut_genes=test_genes, mut_levels=['Gene'],
                         toil_dir=toil_dir, patient_dir=bcc_dir, syn=syn,
                         copy_dir=copy_dir, annot_file=annot_file,
                         cv_prop=1.0)

    plot_label_distribution(infer_vals, args, cdata)


if __name__ == '__main__':
    main()

