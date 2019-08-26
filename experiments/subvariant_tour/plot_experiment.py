
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'experiment')

from HetMan.experiments.subvariant_tour import cis_lbls
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from HetMan.experiments.subvariant_infer import variant_clrs

import argparse
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

cis_clrs = {'None': 'black', 'Self': 'green', 'Chrm': 'blue'}


def lbl_norm(lbls, wt_val, mut_val):
    norm_diff = mut_val - wt_val

    if norm_diff == 0:
        norm_lbls = [0 for lbl in lbls]
    else:
        norm_lbls = [(lbl - wt_val) / norm_diff for lbl in lbls]

    return np.mean(norm_lbls), np.var(norm_lbls)


def plot_label_stability(infer_dfs, auc_df, pheno_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 8), nrows=3, ncols=4)

    use_aucs = auc_df.round(4)
    auc_bins = pd.qcut(use_aucs.values.flatten(), 4, precision=5).categories

    for i, cis_lbl in enumerate(cis_lbls):
        axarr[i, 0].set_ylabel(cis_lbl, fontsize=21)

        use_bins = np.array([
            auc_bins.get_loc(auc_val)
            for auc_val in use_aucs.loc[infer_dfs[cis_lbl].index, cis_lbl]
            ])

        for auc_indx, infer_mat in infer_dfs[cis_lbl].groupby(by=use_bins):
            mtype_norms = {mtype: (
                np.mean(np.concatenate(infr_vals.values[~pheno_dict[mtype]])),
                np.mean(np.concatenate(infr_vals.values[pheno_dict[mtype]]))
                ) for mtype, infr_vals in infer_mat.iterrows()}

            wt_vals = np.vstack([
                np.vstack(infer_mat.ix[mtype, ~pheno_dict[mtype]].apply(
                    lambda vals: lbl_norm(vals, wt_val, mut_val)).values)
                for mtype, (wt_val, mut_val) in mtype_norms.items()
                ])

            mut_vals = np.vstack([
                np.vstack(infer_mat.ix[mtype, pheno_dict[mtype]].apply(
                    lambda vals: lbl_norm(vals, wt_val, mut_val)).values)
                for mtype, (wt_val, mut_val) in mtype_norms.items()
                ])

            if i == 0:
                axarr[i, auc_indx].set_title(
                    "{:.3f} - {:.3f}".format(auc_bins[auc_indx].left,
                                             auc_bins[auc_indx].right),
                    fontsize=21
                    )

            qnt_pnts = np.linspace(2.5, 97.5, 190)
            wt_qnts = np.unique(np.percentile(wt_vals[:, 0], q=qnt_pnts))
            mut_qnts = np.unique(np.percentile(mut_vals[:, 0], q=qnt_pnts))

            wt_vars = np.vstack([
                np.percentile(wt_vals[
                    (wt_vals[:, 0] >= wt_qnts[i - 3])
                    & (wt_vals[:, 0] < wt_qnts[i + 3]), 1
                    ], q=[25, 50, 75])
                for i in range(3, len(wt_qnts) - 3)
                ])

            mut_vars = np.vstack([
                np.percentile(mut_vals[
                    (mut_vals[:, 0] >= mut_qnts[i - 3])
                    & (mut_vals[:, 0] < mut_qnts[i + 3]), 1
                    ], q=[25, 50, 75])
                for i in range(3, len(mut_qnts) - 3)
                ])

            for j, (use_lw, use_ls) in enumerate(zip([1.9, 3.1, 1.9],
                                                     [':', '-', ':'])):
                axarr[i, auc_indx].plot(
                    wt_qnts[3:-3], wt_vars[:, j], color=variant_clrs['WT'],
                    linewidth=use_lw, alpha=0.59, linestyle=use_ls
                    )

                axarr[i, auc_indx].plot(mut_qnts[3:-3], mut_vars[:, j],
                                        color=variant_clrs['Point'],
                                        linewidth=use_lw, alpha=0.59,
                                        linestyle=use_ls)

            axarr[i, auc_indx].set_ylim(0, axarr[i, auc_indx].get_ylim()[1])

    fig.tight_layout(w_pad=2.3, h_pad=1.9)
    plt.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(
                         args.expr_source, args.cohort, args.samp_cutoff),
                     "label-stability_{}__{}.svg".format(
                         args.mut_levels, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_auc_stability(aucs_df, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    stat_dict = {'Mean': aucs_df.applymap(np.mean),
                 'Var': aucs_df.applymap(np.std)}

    for i, cis_lbl in enumerate(cis_lbls):
        for mtype, mean_val in stat_dict['Mean'][cis_lbl].iteritems():
            ax.scatter(mean_val, stat_dict['Var'].loc[mtype, cis_lbl],
                       s=217 * np.mean(pheno_dict[mtype]),
                       c=cis_clrs[cis_lbl], alpha=0.17, edgecolors='none')

    ax.set_ylim([-ax.get_ylim()[1] / 91, ax.get_ylim()[1]])
    plt.xlabel("Average AUC Across CVs", fontsize=23, weight='semibold')
    plt.ylabel("AUC Standard Deviation", fontsize=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(
                         args.expr_source, args.cohort, args.samp_cutoff),
                     "auc-stability_{}__{}.svg".format(
                         args.mut_levels, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_auc_comparisons(auc_df, aucs_df, pheno_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 12), nrows=3, ncols=3)

    var_df = aucs_df.applymap(np.var).applymap(np.log10)
    auc_rng = np.percentile(auc_df.values, q=[0, 100])
    var_rng = np.percentile(var_df.values, q=[0, 100])

    for i, cis_lbl in enumerate(cis_lbls):
        axarr[i, i].axis('off')
        axarr[i, i].text(0.5, 0.5, cis_lbl, size=23,
                         weight='semibold', ha='center', va='center')

    for (i, cis_lbl1), (j, cis_lbl2) in combn(enumerate(cis_lbls), 2):
        for mtype in auc_df.index:
            mtype_size = 211 * np.mean(pheno_dict[mtype])

            axarr[i, j].scatter(
                auc_df.loc[mtype, cis_lbl1], auc_df.loc[mtype, cis_lbl2],
                marker='o', s=mtype_size, alpha=0.41, edgecolor='none'
                )

            axarr[j, i].scatter(
                var_df.loc[mtype, cis_lbl2], var_df.loc[mtype, cis_lbl1],
                marker='o', s=mtype_size, alpha=0.41, edgecolor='none'
                )

        axarr[i, j].plot([auc_rng[0], auc_rng[1]], [auc_rng[0], auc_rng[1]],
                         linewidth=1.3, linestyle='--', color='#550000',
                         alpha=0.53)

        axarr[j, i].plot([var_rng[0], var_rng[1]], [var_rng[0], var_rng[1]],
                         linewidth=1.3, linestyle='--', color='#550000',
                         alpha=0.53)

    fig.tight_layout(w_pad=1.9, h_pad=1.9)
    plt.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(
                         args.expr_source, args.cohort, args.samp_cutoff),
                     "auc-comparisons_{}__{}.svg".format(
                         args.mut_levels, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots general information about a particular run of the experiment.")

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('samp_cutoff',
                        help="a mutation frequency cutoff", type=int)

    parser.add_argument('mut_levels',
                        help="a set of mutation annotation levels", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, args.samp_cutoff)
    os.makedirs(os.path.join(plot_dir, out_tag), exist_ok=True)

    cdata = merge_cohort_data(os.path.join(base_dir, out_tag),
                              args.mut_levels, use_seed=8713)

    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-data__{}__{}.p.gz".format(
                                      args.mut_levels, args.classif)),
                     'r') as f:
        infer_dfs = pickle.load(f)['Infer']

    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in infer_dfs['None'].index}

    auc_df = pd.DataFrame({cis_lbl: {
        mtype: (
            np.greater.outer(
                np.concatenate(infer_vals.values[pheno_dict[mtype]]),
                np.concatenate(infer_vals.values[~pheno_dict[mtype]])
                ).mean()
            + 0.5 * np.equal.outer(
                np.concatenate(infer_vals.values[pheno_dict[mtype]]),
                np.concatenate(infer_vals.values[~pheno_dict[mtype]])
                ).mean()
            ) for mtype, infer_vals in infer_df.iterrows()
        }
        for cis_lbl, infer_df in infer_dfs.items()})

    aucs_df = pd.DataFrame({cis_lbl: {mtype: [
        np.greater.outer(
            np.array([vals[i]
                      for vals in infer_vals.values[pheno_dict[mtype]]]),
            np.array([vals[i]
                      for vals in infer_vals.values[~pheno_dict[mtype]]])
            ).mean() + 0.5 * np.equal.outer(
                np.array([vals[i]
                          for vals in infer_vals.values[pheno_dict[mtype]]]),
                np.array([vals[i]
                          for vals in infer_vals.values[~pheno_dict[mtype]]])
                ).mean()
        for i in range(10)
        ] for mtype, infer_vals in infer_df.iterrows()}
        for cis_lbl, infer_df in infer_dfs.items()})

    plot_label_stability(infer_dfs, auc_df, pheno_dict, args)
    plot_auc_stability(aucs_df, pheno_dict, args)
    plot_auc_comparisons(auc_df, aucs_df, pheno_dict, args)


if __name__ == '__main__':
    main()

