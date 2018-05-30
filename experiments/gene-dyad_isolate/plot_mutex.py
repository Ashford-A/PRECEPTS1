
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


def get_similarities(iso_df, args, cdata):
    use_mtypes = set(iso_df.index.levels[0]) | set(iso_df.index.levels[1])

    simil_df = pd.DataFrame(index=use_mtypes, columns=use_mtypes,
                            dtype=np.float)
    mutex_df = pd.DataFrame(index=use_mtypes, columns=use_mtypes,
                            dtype=np.float)
    auc_df = pd.DataFrame(index=use_mtypes, columns=use_mtypes,
                          dtype=np.float)

    for mtype1, mtype2 in combn(use_mtypes, 2):
        if (mtype1, mtype2) in iso_df.index:
            mutex_df.loc[mtype1, mtype2] = cdata.mutex_test(mtype1, mtype2)
            mutex_df.loc[mtype2, mtype1] = cdata.mutex_test(mtype1, mtype2)

            pheno1 = np.array(cdata.train_pheno(mtype1))
            pheno2 = np.array(cdata.train_pheno(mtype2))

            use_vals1 = iso_df.loc[[(mtype1, mtype2)], :].values.flatten()
            use_vals2 = iso_df.loc[[(mtype2, mtype1)], :].values.flatten()
            
            none_vals1 = np.concatenate(use_vals1[~pheno1 & ~pheno2])
            cur_vals1 = np.concatenate(use_vals1[pheno1 & ~pheno2])
            other_vals1 = np.concatenate(use_vals1[pheno2 & ~pheno1])
            
            none_vals2 = np.concatenate(use_vals2[~pheno1 & ~pheno2])
            cur_vals2 = np.concatenate(use_vals2[pheno2 & ~pheno1])
            other_vals2 = np.concatenate(use_vals2[pheno1 & ~pheno2])

            other_none_prob1 = np.greater.outer(
                none_vals1, other_vals1).mean()
            other_cur_prob1 = np.greater.outer(other_vals1, cur_vals1).mean()
            cur_none_prob1 = np.greater.outer(none_vals1, cur_vals1).mean()
            
            other_none_prob2 = np.greater.outer(
                none_vals2, other_vals2).mean()
            other_cur_prob2 = np.greater.outer(other_vals2, cur_vals2).mean()
            cur_none_prob2 = np.greater.outer(none_vals2, cur_vals2).mean()
            
            simil_df.loc[mtype1, mtype2] = (
                (other_cur_prob1 - other_none_prob1) / (0.5 - cur_none_prob1))
            simil_df.loc[mtype2, mtype1] = (
                (other_cur_prob2 - other_none_prob2) / (0.5 - cur_none_prob2))

            auc_df.loc[mtype1, mtype2] = np.greater.outer(
                cur_vals1, none_vals1).mean()
            auc_df.loc[mtype2, mtype1] = np.greater.outer(
                cur_vals2, none_vals2).mean()

    return simil_df, mutex_df, auc_df


def plot_mutex_similarity(simil_df, annot_df, auc_df, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))

    x_vec = []
    y_vec = []
    annot_vec = []
    size_vec = []

    for mtype1, mtype2 in combn(simil_df.index, 2):
        if (not np.isnan(simil_df.loc[mtype1, mtype2])
                and (auc_df.loc[mtype1, mtype2] > 0.6
                     or auc_df.loc[mtype2, mtype1] > 0.6)):

            size_val = ((len(mtype1.get_samples(cdata.train_mut))
                         + len(mtype2.get_samples(cdata.train_mut)))
                        / len(cdata.samples)) * 119

            mutex_val = -np.log10(annot_df.loc[mtype1, mtype2])
            alpha_val = np.max([auc_df.loc[mtype1, mtype2],
                                auc_df.loc[mtype2, mtype1]]) ** 4 - 0.07

            if auc_df.loc[mtype1, mtype2] > auc_df.loc[mtype2, mtype1]:
                simil_val = simil_df.loc[mtype1, mtype2]
            else:
                simil_val = simil_df.loc[mtype2, mtype1]

            x_vec += [mutex_val]
            y_vec += [simil_val]
            annot_vec += ['{}+{}'.format(mtype1, mtype2)]
            size_vec += [size_val]

            ax.scatter(mutex_val, simil_val,
                       c='#801515', s=size_val, alpha=alpha_val,
                       edgecolors='none')

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

    for i, (xval, yval, annot, size_val) in enumerate(zip(
            x_vec, y_vec, annot_vec, size_vec)):

        if all((xs > (xval + x_range * 0.08)) | (xs < xval)
               | (ys > (yval + y_bound * 0.05))
               | (ys < (yval - y_bound * 0.03))
               for xs, ys in zip(x_vec[:i] + x_vec[(i + 1):],
                                 y_vec[:i] + y_vec[(i + 1):])):

            lbl_gap = (size_val ** 0.5) / 235
            ax.text(xval + lbl_gap, yval + lbl_gap, annot,
                    size=7, stretch='extra-condensed')

    plt.xlabel('Mutual Mutation Exclusivity', size=18, weight='semibold')
    plt.ylabel('Mutual Signature Similarity', size=18, weight='semibold')
    plt.yticks([-1, 0, 1], ['M1 > WT > M2', 'M1 > M2 = WT', 'M1 = M2 > WT'],
               size=12)

    plt.savefig(
        os.path.join(plot_dir,
                     "mutex-simil_{}-{}".format(args.cohort, args.classif)),
        dpi=300, bbox_inches='tight'
        )


def main():
    parser = argparse.ArgumentParser(
        "Plot the ordering of a gene's subtypes in a given cohort based on "
        "how their isolated expression signatures classify one another."
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

    simil_df, annot_df, auc_df = get_similarities(load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.classif,
                     'samps_{}'.format(args.samp_cutoff)),
        ),
        args, cdata)

    plot_mutex_similarity(simil_df.copy(), annot_df.copy(), auc_df.copy(),
                          args, cdata)


if __name__ == '__main__':
    main()

