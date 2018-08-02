
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'report')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.SMMART_analysis.cohorts import CancerCohort
from dryadic.features.mutations import MuType
from HetMan.experiments.SMMART_analysis.fit_gene_models import load_output

import numpy as np
import pandas as pd

import argparse
import synapseclient
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

wt_clr = '0.29'
mut_clr = '#C50000'


def plot_gene_scores(out_df, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 10),
                              nrows=2, ncols=6, sharex=True, sharey=True)

    samp_list = cdata.subset_samps()
    tcga_indx = np.array([samp.split('-')[0] == "TCGA" for samp in samp_list])
    pt_indx = [i for i, samp in enumerate(samp_list)
                if samp.split(' --- ')[0] == args.patient][0]

    for ax, (gene, vals) in zip(axarr.flatten(), out_df.iteritems()):
        cur_mtype = MuType({('Gene', gene): None})
        cur_pheno = np.array(cdata.train_pheno(cur_mtype))

        tcga_wt = vals[~cur_pheno & tcga_indx].quantile(q=(0.25, 0.75))
        tcga_mut = vals[cur_pheno & tcga_indx].quantile(q=(0.25, 0.75))
        norm_val = tcga_mut[0.75] - tcga_wt[0.25]

        ax.add_patch(Rectangle(
            (0.2, 0), 0.6, (tcga_wt[0.75] - tcga_wt[0.25]) / norm_val,
            color=wt_clr, ec=wt_clr, alpha=0.31, lw=2.1
            ))

        ax.add_patch(Rectangle(
            (0.2, (tcga_mut[0.25] - tcga_wt[0.25]) / norm_val),
            0.6, (tcga_mut[0.75] - tcga_mut[0.25]) / norm_val,
            color=mut_clr, ec=mut_clr, alpha=0.31, lw=2.1
            ))

        pt_val = (vals[pt_indx] - tcga_wt[0.25]) / norm_val
        ax.axhline(np.clip((vals[pt_indx] - tcga_wt[0.25]) / norm_val, 0, 1),
                   xmin=0.13, xmax=0.87, color='black', lw=2.9)

        ax.axes.get_xaxis().set_visible(False)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax.set_yticklabels(['WT', '', '', '', 'Mut'])
        ax.set_title(gene)

    plt.tight_layout(w_pad=1.7, h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir,
                     'patient_{}__{}-{}.png'.format(
                         args.patient, args.cohort, args.classif)),
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
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('patient', help='a SMMART patient')

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
    out_dir = Path(os.path.join(base_dir, 'output',
                                'gene_models', args.cohort))

    use_genes = set()
    for out_fl in out_dir.glob("*/{}__cv-0.p".format(args.classif)):
        out_gene = out_fl.parent.name

        if len(tuple(out_dir.glob("{}/{}__cv-*.p".format(
                out_gene, args.classif)))) == 5:
            use_genes |= {out_gene}

    out_df = pd.DataFrame.from_dict({
        use_gene: np.apply_along_axis(
            lambda vals: np.mean(np.concatenate(vals)), 1,
            np.stack([np.array(ols['Infer'])
                      for ols in load_output(
                          args.cohort, use_gene, args.classif)],
                     axis=1)
            )
        for use_gene in use_genes
        })

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = args.syn_root
    syn.login()

    cdata = CancerCohort(
        cancer=args.cohort, mut_genes=list(use_genes), mut_levels=['Gene'],
        tcga_dir=args.toil_dir, patient_dir=args.patient_dir, syn=syn,
        collapse_txs=True, cv_prop=1.0
        )

    plot_gene_scores(out_df.iloc, args, cdata)


if __name__ == '__main__':
    main()

