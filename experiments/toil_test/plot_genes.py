
"""Plotting the results of testing Toil Kallisto expression data.

"""

import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots', 'genes')
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.toil_test.fit_genes import load_output
from HetMan.features.variants import MuType
from HetMan.features.cohorts.mut import VariantCohort

import numpy as np
import pandas as pd

import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'
toil_dir = '/home/exacloud/lustre1/CompBio/data/toil_tcga'


def plot_auc_comp(out_data, args, cdata_dict):

    # turns on drawing borders around the edges of the plotting area
    mpl.rcParams['axes.linewidth'] = 2.7
    mpl.rcParams['axes.edgecolor'] = '0.59'

    # instantiates the plot, decides the file where it will be saved
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_fl = 'acc-perf_{}-{}.png'.format(args.cohort, args.classif)

    # gets the 1st quartile of classifier performance on each input dataset
    fire_acc = out_data.loc[:, 'Firehose'].quantile(axis=1, q=0.25)
    toil_acc = out_data.loc[:, 'toil'].quantile(axis=1, q=0.25)

    plot_min = min(np.min(fire_acc), np.min(toil_acc)) - 0.01
    lbl_offset = {mtype: None for mtype in out_data.index}
    lbl_align = {mtype: 'left' for mtype in out_data.index}
    use_samps = cdata_dict['Firehose'].samples & cdata_dict['toil'].samples

    for mtype in out_data.index:
        if not ((fire_acc > fire_acc[mtype])
                & (fire_acc < fire_acc[mtype] + 0.04)
                & (toil_acc > toil_acc[mtype] - 0.01)
                & (toil_acc < toil_acc[mtype] + 0.03)).any():
            lbl_offset[mtype] = (1, 1)

    for mtype in out_data.index:
        if not lbl_offset[mtype]:
            if not ((fire_acc > fire_acc[mtype])
                    & (fire_acc < fire_acc[mtype] + 0.04)
                    & (toil_acc > toil_acc[mtype] - 0.03)
                    & (toil_acc < toil_acc[mtype] + 0.01)).any():
                lbl_offset[mtype] = (1, -1)

    for mtype in out_data.index:
        if not lbl_offset[mtype]:
            if not ((fire_acc > fire_acc[mtype] - 0.04)
                    & (fire_acc < fire_acc[mtype])
                    & (toil_acc > toil_acc[mtype] - 0.01)
                    & (toil_acc < toil_acc[mtype] + 0.03)).any():
                lbl_offset[mtype] = (-1, 1)
                lbl_align[mtype] = 'right'

    for mtype in out_data.index:
        if not lbl_offset[mtype]:
            if not ((fire_acc > fire_acc[mtype] - 0.04)
                    & (fire_acc < fire_acc[mtype])
                    & (toil_acc > toil_acc[mtype] - 0.03)
                    & (toil_acc < toil_acc[mtype] + 0.01)).any():
                lbl_offset[mtype] = (-1, -1)
                lbl_align[mtype] = 'right'

    for mtype in out_data.index:
        mtype_size = (
            np.sum(np.array(cdata_dict['toil'].train_pheno(mtype, use_samps)))
            * (975 / len(use_samps))
            )

        ax.scatter(fire_acc[mtype], toil_acc[mtype],
                   s=mtype_size, c='black', marker='o', alpha=0.27)

        if lbl_offset[mtype]:
            ax.text(x=(fire_acc[mtype]
                       + lbl_offset[mtype][0] * (mtype_size / 4e6) ** 0.5),
                    y=(toil_acc[mtype]
                       + lbl_offset[mtype][1] * (mtype_size / 4e6) ** 0.5),
                    s=str(mtype), size=8, va='center', ha=lbl_align[mtype],
                    stretch=10)

    plt.axhline(color='r', y=0.5, xmin=-2, xmax=2,
                linewidth=1.1, linestyle=':')
    plt.axvline(color='r', x=0.5, ymin=-2, ymax=2,
                linewidth=1.1, linestyle=':')
    plt.plot([-1, 2], [-1, 2], linewidth=1.7, linestyle='--')

    # creates axis titles and labels
    plt.xlabel('Firehose Gene AUC', fontsize=21)
    plt.ylabel('Toil Transcript AUC', fontsize=21)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # sets axis limits, ensuring the plot is square
    ax.set_xlim(plot_min, 1)
    ax.set_ylim(plot_min, 1)

    # removes plot border on the labelled sides
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # sets the size of the plotting area and saves the plot to file
    fig.set_size_inches(10, 10)
    plt.savefig(os.path.join(plot_dir, plot_fl),
                dpi=500, bbox_inches='tight')
    plt.close()


def main():
    """Creates plots for a given combination of cohort and classifier."""
    parser = argparse.ArgumentParser()

    # parses command line arguments
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')
    args = parser.parse_args()

    # loads output of the experiment, extracts the genes that were tested
    acc_df, coef_df = load_output(
        os.path.join(base_dir, 'output', args.cohort, args.classif, 'genes'))
    use_genes = [mtype.subtype_list()[0][0] for mtype in acc_df.index]

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/mgrzad"
                                "/input-data/synapse")
    syn.login()

    # loads expression and mutation data using Firehose as a source
    cdata_fire = VariantCohort(
        cohort=args.cohort, mut_genes=use_genes, mut_levels=['Gene'],
        expr_source='Firehose', data_dir=firehose_dir, cv_prop=1.0, syn=syn
        )

    # loads expression and mutation data using Toil as a source
    cdata_toil = VariantCohort(
        cohort=args.cohort, mut_genes=use_genes, mut_levels=['Gene'],
        expr_source='toil', data_dir=toil_dir, cv_prop=1.0, syn=syn
        )

    # creates plots
    cdata_dict = {'Firehose': cdata_fire, 'toil': cdata_toil}
    plot_auc_comp(acc_df, args, cdata_dict)


if __name__ == '__main__':
    main()

