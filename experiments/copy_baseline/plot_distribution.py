
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'copy_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'distribution')

from HetMan.experiments.copy_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.variant_baseline.plot_experiment import cv_clrs
from HetMan.experiments.utilities.colour_maps import cor_cmap

import argparse
import bz2
import dill as pickle

import numpy as np
import pandas as pd
import numbers
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_label_distribution(gene, score_dict, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 6), nrows=1, ncols=3)

    for i, cv_mth in enumerate(['random', 'fivefold', 'infer']):
        for samp, lbls in score_dict[cv_mth][gene].iteritems():
            if isinstance(lbls, numbers.Number):
                axarr[i].scatter(cdata.copy_mat.loc[samp, gene], lbls, s=13,
                                 c='black', alpha=0.31, edgecolors='none')

            else:
                for lbl in lbls:
                    if lbl == lbl:
                        axarr[i].scatter(cdata.copy_mat.loc[samp, gene], lbl,
                                         s=9, c='black', alpha=0.31,
                                         edgecolors='none')

    fig.tight_layout(w_pad=1.7, h_pad=2.3)
    fig.savefig(
        os.path.join(plot_dir, args.expr_source,
                     "{}__samps-{}".format(args.cohort, args.samp_cutoff),
                     args.model_name,
                     "{}_label-distribution.svg".format(gene)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the distributions of the labels assigned by a copy number "
        "alteration score regressor for a set of genetic features."
        )

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source was used")
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, args.samp_cutoff)

    os.makedirs(os.path.join(plot_dir, args.expr_source,
                             "{}__samps-{}".format(args.cohort,
                                                   args.samp_cutoff),
                             args.model_name),
                exist_ok=True)

    cdata = merge_cohort_data(os.path.join(base_dir, out_tag))
    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-data__{}.p.gz".format(
                                      args.model_name)),
                     'r') as fl:
        out_dict = pickle.load(fl)

    auc_vals = out_dict['Fit']['test']['Cor'].quantile(q=0.25, axis=1)
    for gene in auc_vals.index[auc_vals > auc_vals.quantile(q=0.8)]:
        plot_label_distribution(gene, out_dict['Scores'], args, cdata)


if __name__ == "__main__":
    main()

