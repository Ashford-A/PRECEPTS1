
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'distribution')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.SMMART_analysis.cohorts import CancerCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.SMMART_analysis.fit_gene_models import load_output

import numpy as np
import pandas as pd
from operator import itemgetter

import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

wt_clr = '0.29'
mut_clr = sns.hls_palette(1, l=.51, s=.88)[0]


def plot_label_distribution(infer_vals, args, cdata):
    fig, ax = plt.subplots(figsize=(7, 14))

    samp_list = cdata.subset_samps()
    infer_means = np.apply_along_axis(
        lambda x: np.mean(np.concatenate(x)), 1, infer_vals)

    tcga_means = pd.Series(
        {samp: val for samp, val in zip(samp_list, infer_means)
         if 'TCGA' in samp}
        )

    smrt_means = sorted(
        [(samp, val) for samp, val in zip(samp_list, infer_means)
         if 'TCGA' not in samp],
        key=itemgetter(1)
        )

    if np.all(infer_means >= 0):
        plt_ymin, plt_ymax = 0, max(np.max(infer_means), 1)

    else:
        plt_ymax = np.max([np.max(np.absolute(infer_means)) * 1.05, 1.1])
        plt_ymin = -plt_ymax

    use_mtype = MuType({('Gene', args.gene): None})
    mtype_stat = np.array(cdata.train_mut.status(tcga_means.index))
    kern_bw = (plt_ymax - plt_ymin) / 47

    ax = sns.kdeplot(tcga_means[~mtype_stat], color=wt_clr, alpha=0.4,
                     vertical=True, shade=True, linewidth=2.1, bw=kern_bw,
                     gridsize=1000, label='Wild-Type')

    ax = sns.kdeplot(tcga_means[mtype_stat], color=mut_clr, alpha=0.4,
                     vertical=True, shade=True, linewidth=2.1, bw=kern_bw,
                     gridsize=1000, label='{} Mutant'.format(args.gene))

    plt_xmin, plt_xmax = plt.xlim()
    lbl_pad = (plt_ymax - plt_ymin) / 79

    for i, (patient, val) in enumerate(smrt_means):
        ax.axhline(y=val, xmin=0, xmax=plt_xmax * 0.22,
                   ls=':', lw=3.9)

        if i > 0 and smrt_means[i - 1][1] > (val - lbl_pad):
            ax.text(plt_xmax * 0.32, val, patient,
                    size=10, ha='left', va='bottom')

        elif (i < (len(smrt_means) - 1)
              and smrt_means[i + 1][1] < (val + lbl_pad)):
            ax.text(plt_xmax * 0.32, val, patient,
                    size=10, ha='left', va='top')

        else:
            ax.text(plt_xmax * 0.32, val, patient,
                    size=10, ha='left', va='center')

    tcga_auc = np.greater.outer(tcga_means[mtype_stat],
                                tcga_means[~mtype_stat]).mean()
    ax.text(ax.get_xlim()[1] * 0.56, plt_ymax * 0.83,
            "TCGA AUC: {:.3f}".format(tcga_auc), size=21)

    plt.legend(frameon=False, fontsize=18, loc=8, ncol=2)
    plt.ylim(plt_ymin, plt_ymax)

    plt.xlabel('TCGA-{} Density'.format(args.cohort),
               fontsize=23, weight='semibold')
    plt.ylabel('Inferred {} Mutation Score'.format(args.gene),
               fontsize=23, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     'labels__{}-{}-{}.png'.format(
                         args.cohort, args.gene, args.classif)),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the predicted mutation scores for a given cohort of SMMART "
        "samples against the distribution of scores for the matching cohort "
        "of TCGA samples."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')

    parser.add_argument(
        'toil_dir', type=str,
        help='the directory where toil expression data is saved'
        )
    parser.add_argument('syn_root', type=str,
                        help='Synapse cache root directory')
    parser.add_argument(
        'patient_dir', type=str,
        help='directy where SMMART patient RNAseq abundances are stored'
        )

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    out_list = load_output(args.cohort, args.gene, args.classif)
    infer_vals = np.stack([np.array(ols['Infer']) for ols in out_list],
                          axis=1)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = args.syn_root
    syn.login()

    cdata = CancerCohort(
        cancer=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        tcga_dir=args.toil_dir, patient_dir=args.patient_dir, syn=syn,
        collapse_txs=True, cv_prop=1.0
        )

    plot_label_distribution(infer_vals, args, cdata)


if __name__ == '__main__':
    main()

