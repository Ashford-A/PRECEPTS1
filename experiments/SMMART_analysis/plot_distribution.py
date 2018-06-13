
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
        plt_ymin, plt_ymax = 0, max(np.max(infer_means) * 1.09, 1)

    else:
        plt_ymax = np.max([np.max(np.absolute(infer_means)) * 1.09, 1.1])
        plt_ymin = -plt_ymax

    plt.ylim(plt_ymin, plt_ymax)
    plt_xmin, plt_xmax = plt.xlim()
    lbl_pad = (plt_ymax - plt_ymin) / 79

    use_mtype = MuType({('Gene', args.gene): None})
    mtype_stat = np.array(cdata.train_mut.status(tcga_means.index))
    kern_bw = (plt_ymax - plt_ymin) / 47

    ax = sns.kdeplot(tcga_means[~mtype_stat], color=wt_clr, vertical=True,
                     shade=True, alpha=0.36, linewidth=0, bw=kern_bw, cut=0,
                     gridsize=1000, label='Wild-Type')

    ax = sns.kdeplot(tcga_means[mtype_stat], color=mut_clrs[0], vertical=True,
                     shade=True, alpha=0.36, linewidth=0, bw=kern_bw, cut=0,
                     gridsize=1000, label='{} Mutant'.format(args.gene))

    # for each SMMART patient, check if they have a mutation of the given gene
    for i, (patient, val) in enumerate(smrt_means):
        if patient in cdata.train_mut.get_samples():

            mut_list = []
            for lbl, muts in cdata.train_mut[args.gene]:
                if patient in muts:
                    mut_list += [lbl]

            plt_str = '{} ({})'.format(patient, '+'.join(mut_list))
            plt_clr = mut_clrs[1]
            plt_lw = 3.1

        # if the patient's RNAseq sample did not have any mutations, check all
        # the samples associated with the patient
        else:
            mut_dir = os.path.join(
                args.patient_dir, "16113-{}".format(patient.split(' ---')[0]),
                'output', 'cancer_exome'
                )

            # check if any mutation calling was done for this patient
            mut_files = subprocess.run(
                'find {}'.format(mut_dir), shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ).stdout.decode('utf-8')

            # if calling was done for any sample associated with this patient,
            # check for mutations of the given gene
            if mut_files:
                mut_grep = subprocess.run(
                    'grep "^{}" {}'.format(
                        args.gene, os.path.join(
                            mut_dir, "*SMMART_Cancer_Exome*", "*.maf")
                        ),
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    ).stdout.decode('utf-8')

                if mut_grep:
                    mut_list = []
                    for mut_match in mut_grep.split('\n'):
                        if mut_match:
                            mut_list += [mut_match.split('\t')[8]]

                    plt_str = '{} ({})'.format(
                        patient, '+'.join(np.unique(mut_list)))
                    plt_clr = mut_clrs[3]
                    plt_lw = 2.6

                else:
                    plt_str = '{}'.format(patient)
                    plt_clr = wt_clr
                    plt_lw = 1.7

            else:
                plt_str = '{}'.format(patient)
                plt_clr = '#2E6AF3'
                plt_lw = 1.7

        ax.axhline(y=val, xmin=0, xmax=plt_xmax * 0.22,
                   c=plt_clr, ls='--', lw=plt_lw)

        if i > 0 and smrt_means[i - 1][1] > (val - lbl_pad):
            txt_va = 'bottom'

        elif (i < (len(smrt_means) - 1)
              and smrt_means[i + 1][1] < (val + lbl_pad)):
            txt_va = 'top'

        else:
            txt_va = 'center'

        ax.text(plt_xmax * 0.32, val, plt_str, size=9, ha='left', va=txt_va)

    # calculate the accuracy of the mutation scores inferred across
    # validation runs in predicting mutation status
    tcga_f1 = average_precision_score(mtype_stat, tcga_means)
    tcga_auc = np.greater.outer(tcga_means[mtype_stat],
                                tcga_means[~mtype_stat]).mean()

    # add annotation about the mutation scores' accuracy to the plot
    ax.text(ax.get_xlim()[1] * 0.91, plt_ymax * 0.82, size=18, ha='right',
            s="TCGA AUPR:{:8.3f}".format(tcga_f1))
    ax.text(ax.get_xlim()[1] * 0.91, plt_ymax * 0.88, size=18, ha='right',
            s="TCGA AUC:{:8.3f}".format(tcga_auc))

    plt.xlabel('TCGA-{} Density'.format(args.cohort),
               fontsize=21, weight='semibold')
    plt.ylabel('Inferred {} Mutation Score'.format(args.gene),
               fontsize=21, weight='semibold')

    plt.legend([Line2D([0], [0], color=mut_clrs[1], lw=3.7, ls='--'),
                Line2D([0], [0], color=mut_clrs[3], lw=3.7, ls='--'),
                Patch(color=mut_clrs[0], alpha=0.36),
                Line2D([0], [0], color=wt_clr, lw=3.7, ls='--'),
                Line2D([0], [0], color='#2E6AF3', lw=3.7, ls='--'),
                Patch(color=wt_clr, alpha=0.36)],
               ["Sample {} Mutant".format(args.gene),
                "Patient {} Mutant".format(args.gene),
                "TCGA {} Mutants".format(args.gene), "SMMART Wild-Type",
                "No Mutation Calls", "TCGA Wild-Types"],
               fontsize=13, loc=8, ncol=2)

    fig.savefig(
        os.path.join(plot_dir,
                     'labels__{}-{}-{}.png'.format(
                         args.cohort, args.gene, args.classif)),
        dpi=300, bbox_inches='tight'
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

    cdata = CancerCohort(cancer=args.cohort, mut_genes=[args.gene],
                         mut_levels=['Gene', 'Form'], tcga_dir=args.toil_dir,
                         patient_dir=args.patient_dir, syn=syn,
                         collapse_txs=True, cv_prop=1.0)

    plot_label_distribution(infer_vals, args, cdata)


if __name__ == '__main__':
    main()

