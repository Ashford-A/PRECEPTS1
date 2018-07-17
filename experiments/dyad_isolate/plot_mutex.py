
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'mutex')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.experiments.utilities import load_infer_output, simil_cmap

import numpy as np
import pandas as pd

import argparse
import synapseclient
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def get_pair_data(iso_df, args, cdata):
    out_df = pd.DataFrame(index=iso_df.index,
                          columns=['Mutex', 'Simil', 'AUC'])

    for mtype_pair, iso_vals in iso_df.iterrows():
        if mtype_pair[0] < mtype_pair[1]:
            out_df.loc[mtype_pair, 'Mutex'] = cdata.mutex_test(*mtype_pair)

        cur_pheno = np.array(cdata.train_pheno(mtype_pair[0]))
        other_pheno = np.array(cdata.train_pheno(mtype_pair[1]))
 
        none_vals = np.concatenate(iso_vals.values[~cur_pheno & ~other_pheno])
        cur_vals = np.concatenate(iso_vals.values[cur_pheno & ~other_pheno])
        other_vals = np.concatenate(iso_vals.values[~cur_pheno & other_pheno])
 
        other_none_prob = np.greater.outer(none_vals, other_vals).mean()
        other_cur_prob = np.greater.outer(other_vals, cur_vals).mean()
        cur_none_prob = np.greater.outer(none_vals, cur_vals).mean()
 
        out_df.loc[mtype_pair, 'Simil'] = (
            (other_cur_prob - other_none_prob) / (0.5 - cur_none_prob))
        out_df.loc[mtype_pair, 'AUC'] = 1 - cur_none_prob

    return out_df


def place_annot(x_vec, y_vec, size_vec, annot_vec, x_range, y_range):
    placed_annot = []
 
    for i, (xval, yval, size_val, annot) in enumerate(zip(
            x_vec, y_vec, size_vec, annot_vec)):

        if all((xs > (xval + x_range * 0.08)) | (xs < xval)
               | (ys > (yval + y_range * 0.02))
               | (ys < (yval - y_range * 0.02))
               for xs, ys in zip(x_vec[:i] + x_vec[(i + 1):],
                                 y_vec[:i] + y_vec[(i + 1):])):

            lbl_gap = (size_val ** 0.5) / 235
            placed_annot += [(xval + lbl_gap, yval + lbl_gap, annot)]

    return placed_annot


def plot_mutex_similarity(out_df, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))

    x_vec = []
    y_vec = []
    annot_vec = []
    size_vec = []

    for mtype_pair, (mtx_val, sim_val, auc_val) in out_df.iterrows():
        if auc_val > 0.65:
            pair_size = len((mtype_pair[0] | mtype_pair[1]).get_samples(
                cdata.train_mut))
 
            if mtype_pair[0] > mtype_pair[1]:
                x_vec += [-np.log10(out_df.loc[tuple(sorted(mtype_pair)),
                                               'Mutex'])]
            else:
                x_vec += [-np.log10(mtx_val)]

            size_vec += [127 * pair_size / len(cdata.samples)]
            annot_vec += ['{}->{}'.format(*mtype_pair)]
            y_vec += [sim_val]

            ax.scatter(x_vec[-1], y_vec[-1], c='#801515', s=size_vec[-1],
                       alpha=auc_val ** 6, edgecolors='none')

    plt_xmin, plt_xmax = plt.xlim()
    plt_ymin, plt_ymax = plt.ylim()

    if plt_xmin > 0:
        plt_xmin = 0

    if plt_ymin > -1.0:
        plt_ymin = -1.0
    if plt_ymax < 1.0:
        plt_ymax = 1.0

    y_bound = np.max(np.absolute([plt_ymin, plt_ymax]))
    x_range = plt_xmax - plt_xmin
    plt.xlim(plt_xmin, plt_xmax)
    plt.ylim(-y_bound, y_bound)

    for annot_x, annot_y, annot in place_annot(
            x_vec, y_vec, size_vec, annot_vec, x_range, y_bound * 2):
        ax.text(annot_x, annot_y, annot, size=7, stretch='extra-condensed')

    plt.xlabel('Mutual Mutation Exclusivity', size=18, weight='semibold')
    plt.ylabel('Mutual Signature Similarity', size=18, weight='semibold')
    plt.yticks([-1, 0, 1], ['M1 > WT > M2', 'M1 > M2 = WT', 'M1 = M2 > WT'],
               size=12)

    plt.savefig(
        os.path.join(plot_dir,
                     "simil_{}-{}__samps_{}".format(
                         args.cohort, args.classif, args.samp_cutoff)
                    ),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def plot_mutex_pair_similarity(out_df, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))

    x_vec = []
    y_vec = []
    annot_vec = []
    size_vec = []

    for mtype1, mtype2 in set(tuple(sorted(mtype_pair))
                              for mtype_pair in out_df.index):
        auc_vals = out_df.loc[[(mtype1, mtype2), (mtype2, mtype1)], 'AUC']

        if min(auc_vals) > 0.6:
            auc_adj = np.fmax(auc_vals - 0.5, 0)
            pair_size = len((mtype1 | mtype2).get_samples(cdata.train_mut))

            x_vec += [-np.log10(out_df.loc[(mtype1, mtype2), 'Mutex'])]
            size_vec += [159 * pair_size / len(cdata.samples)]
            annot_vec += ['{}+{}'.format(mtype1, mtype2)]
 
            y_vec += [np.average(
                out_df.loc[[(mtype1, mtype2), (mtype2, mtype1)], 'Simil'],
                weights=auc_adj ** (18 / 13)
                )]
 
            ax.scatter(x_vec[-1], y_vec[-1], c='#801515', s=size_vec[-1],
                       alpha=max(auc_adj) ** 0.5, edgecolors='none')

    plt_xmin, plt_xmax = plt.xlim()
    plt_ymin, plt_ymax = plt.ylim()

    if plt_xmin > 0:
        plt_xmin = 0

    if plt_ymin > -1.0:
        plt_ymin = -1.0
    if plt_ymax < 1.0:
        plt_ymax = 1.0

    y_bound = np.max(np.absolute([plt_ymin, plt_ymax]))
    x_range = plt_xmax - plt_xmin
    plt.xlim(plt_xmin, plt_xmax)
    plt.ylim(-y_bound, y_bound)

    for annot_x, annot_y, annot in place_annot(
            x_vec, y_vec, size_vec, annot_vec, x_range, y_bound * 2):
        ax.text(annot_x, annot_y, annot, size=7, stretch='extra-condensed')

    plt.xlabel('Mutual Mutation Exclusivity', size=18, weight='semibold')
    plt.ylabel('Mutual Signature Similarity', size=18, weight='semibold')
    plt.yticks([-1, 0, 1], ['M1 > WT > M2', 'M1 > M2 = WT', 'M1 = M2 > WT'],
               size=12)

    plt.savefig(
        os.path.join(plot_dir,
                     "paired-simil_{}-{}__samps_{}".format(
                         args.cohort, args.classif, args.samp_cutoff)
                    ),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the similarities of the expression effects of each gene pair's "
        "mutations based on how their signatures classify one other against "
        "the mutual exclusivity of their occurence."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument(
        '--samp_cutoff', default=40, type=int,
        help='minimum number of samples a gene must be mutated in'
        )

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=None, mut_levels=['Gene'],
        expr_source='Firehose', expr_dir=firehose_dir, var_source='mc3',
        samp_cutoff=args.samp_cutoff, syn=syn, cv_prop=1.0
        )

    out_df = get_pair_data(load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.classif,
                     'samps_{}'.format(args.samp_cutoff)),
        ),
        args, cdata)

    plot_mutex_similarity(out_df.copy(), args, cdata)
    plot_mutex_pair_similarity(out_df.copy(), args, cdata)


if __name__ == '__main__':
    main()

