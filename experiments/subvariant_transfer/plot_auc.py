
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'subvariant_transfer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'auc')

from HetMan.experiments.subvariant_transfer import *
from dryadic.features.mutations import MuType

import argparse
import dill as pickle
from glob import glob
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'


def plot_auc_comparisons(auc_dict, size_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 12), nrows=2, ncols=2)

    cohort_cmap = sns.hls_palette(len(args.cohorts), l=.4, s=.71)
    coh_clrs = {coh: cohort_cmap[sorted(args.cohorts).index(coh)]
                for coh in args.cohorts}

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_size = (0.49 * size_dict[tst_coh, mtype]) ** 0.43

            if trn_coh == tst_coh:
                axarr[0, 0].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

                for trn_coh2, tst_coh2 in auc_dict['All'][mtype]:
                    if trn_coh2 == trn_coh and tst_coh2 != tst_coh:
                        axarr[0, 1].plot(
                            auc_dict['All'][mtype][(trn_coh, tst_coh)],
                            auc_dict['All'][mtype][(trn_coh, tst_coh2)],
                            marker='o', markersize=mtype_size, alpha=0.19,
                            color=coh_clrs[tst_coh]
                            )

                        axarr[1, 0].plot(
                            auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                            auc_dict['Iso'][mtype][(trn_coh, tst_coh2)],
                            marker='o', markersize=mtype_size, alpha=0.19,
                            color=coh_clrs[tst_coh]
                            )

            else:
                axarr[1, 1].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

    plot_min = min(auc_val for exp_dict in auc_dict.values()
                   for mtype_dict in exp_dict.values()
                   for auc_val in mtype_dict.values()) - 0.02

    for ax in axarr.flatten():
        ax.plot([-1, 2], [-1, 2],
                linewidth=1.5, linestyle='--', color='#550000', alpha=0.49)

        ax.axhline(y=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)
        ax.axvline(x=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_min, 1.005)
        ax.set_ylim(plot_min, 1.005)
        ax.tick_params(labelsize=13, pad=2.9)

    axarr[0, 0].set_xlabel("Default AUC", fontsize=19, weight='semibold')
    axarr[0, 0].set_ylabel("Isolated AUC", fontsize=19, weight='semibold')

    axarr[0, 1].set_xlabel("Default AUC", fontsize=19, weight='semibold')
    axarr[0, 1].set_ylabel("Transfer Default AUC",
                           fontsize=19, weight='semibold')

    axarr[1, 0].set_xlabel("Isolated AUC", fontsize=19, weight='semibold')
    axarr[1, 0].set_ylabel("Transfer Isolated AUC",
                           fontsize=19, weight='semibold')

    axarr[1, 1].set_xlabel("Transfer Default AUC",
                           fontsize=19, weight='semibold')
    axarr[1, 1].set_ylabel("Transfer Isolated AUC",
                           fontsize=19, weight='semibold')

    fig.tight_layout(w_pad=3.1, h_pad=2.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     "{}_{}__acc-comparisons.svg".format(args.classif,
                                                         args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_cohort_transfer(auc_dict, size_dict, args):
    fig_size = 1 + len(args.cohorts) * 2.9
    fig, axarr = plt.subplots(figsize=(fig_size, fig_size),
                              nrows=len(args.cohorts),
                              ncols=len(args.cohorts))

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_size = 0.051 * fig_size * size_dict[tst_coh, mtype]
            mtype_size **= 0.43

            ax_i = sorted(args.cohorts).index(trn_coh)
            ax_j = sorted(args.cohorts).index(tst_coh)

            axarr[ax_i, ax_j].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                   auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                   marker='o', markersize=mtype_size,
                                   color='black', markeredgecolor='none',
                                   alpha=0.29)

    plot_min = min(auc_val for exp_dict in auc_dict.values()
                   for mtype_dict in exp_dict.values()
                   for auc_val in mtype_dict.values()) - 0.01

    for ax in axarr.flatten():
        ax.plot([-1, 2], [-1, 2],
                linewidth=1.5, linestyle='--', color='#550000', alpha=0.49)

        ax.axhline(y=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)
        ax.axvline(x=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_min, 1.005)
        ax.set_ylim(plot_min, 1.005)
        ax.tick_params(labelsize=4 + fig_size, pad=fig_size / 4.7)

    fig.text(0.5, 0, "Default AUC", size=13 + fig_size * 0.63,
             ha='center', va='top', fontweight='semibold')
    fig.text(0, 0.5, "Isolated AUC", size=13 + fig_size * 0.63,
             rotation=90, ha='right', va='center', fontweight='semibold')

    fig.text(0.5, 1.02, "Training Cohort", size=13 + fig_size * 0.63,
             ha='center', va='bottom', fontweight='semibold')
    fig.text(1.02, 0.5, "Testing Cohort", size=13 + fig_size * 0.63,
             rotation=270, ha='left', va='center', fontweight='semibold')

    for i, cohort in enumerate(sorted(args.cohorts)):
        axarr[0, i].text(0.5, 1.02, cohort, size=10 + fig_size * 0.63,
                         ha='center', va='bottom', fontweight='semibold',
                         transform=axarr[0, i].transAxes)

        axarr[i, -1].text(1.02, 0.5, cohort, size=10 + fig_size * 0.63,
                          ha='left', va='center', rotation=270,
                          fontweight='semibold',
                          transform=axarr[i, -1].transAxes)

    for i in range(len(args.cohorts)):
        for j in range(len(args.cohorts)):
            axarr[i, j].set_xticks([0.5, 0.7, 0.9], minor=False)
            axarr[i, j].set_yticks([0.5, 0.7, 0.9], minor=False)

            if i != (len(args.cohorts) - 1):
                axarr[i, j].set_xticklabels([])
            if j != 0:
                axarr[i, j].set_yticklabels([])

    fig.tight_layout(w_pad=1.3, h_pad=1.3)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     "{}_{}__cohort-transfer.svg".format(args.classif,
                                                         args.ex_mtype)),
        dpi=600, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_stability_comparisons(stab_dict, auc_dict, size_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 12), nrows=2, ncols=2)

    cohort_cmap = sns.hls_palette(len(args.cohorts), l=.4, s=.71)
    coh_clrs = {coh: cohort_cmap[sorted(args.cohorts).index(coh)]
                for coh in args.cohorts}

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_size = (0.49 * size_dict[tst_coh, mtype]) ** 0.43

            if trn_coh == tst_coh:
                axarr[0, 0].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['All'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

                axarr[0, 1].plot(auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size, alpha=0.19,
                                 color=coh_clrs[tst_coh])

            else:
                axarr[1, 0].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['All'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size, alpha=0.19,
                                 color=coh_clrs[tst_coh])

                axarr[1, 1].plot(auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

    plot_xmin = min(auc_val for exp_dict in auc_dict.values()
                    for mtype_dict in exp_dict.values()
                    for auc_val in mtype_dict.values()) - 0.01

    plot_ymax_wthn = max(
        stab_val for exp_dict in stab_dict.values()
        for mtype_dict in exp_dict.values()
        for (trn_coh, tst_coh), stab_val in mtype_dict.items()
        if trn_coh == tst_coh
        ) + 0.02

    plot_ymax_btwn = max(
        stab_val for exp_dict in stab_dict.values()
        for mtype_dict in exp_dict.values()
        for (trn_coh, tst_coh), stab_val in mtype_dict.items()
        if trn_coh != tst_coh
        ) + 0.02

    for ax in axarr.flatten():
        ax.axvline(x=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.tick_params(labelsize=13, pad=2.9)
        ax.set_xlim(plot_xmin, 1.005)

    axarr[0, 0].set_ylim(0, plot_ymax_wthn)
    axarr[0, 1].set_ylim(0, plot_ymax_wthn)
    axarr[1, 0].set_ylim(0, plot_ymax_btwn)
    axarr[1, 1].set_ylim(0, plot_ymax_btwn)

    axarr[0, 0].set_xlabel("Default AUC", fontsize=19, weight='semibold')
    axarr[0, 0].set_ylabel("Default Instability",
                           fontsize=19, weight='semibold')

    axarr[0, 1].set_xlabel("Isolated AUC", fontsize=19, weight='semibold')
    axarr[0, 1].set_ylabel("Isolated Instability",
                           fontsize=19, weight='semibold')

    axarr[1, 0].set_xlabel("Transfer Default AUC",
                           fontsize=19, weight='semibold')
    axarr[1, 0].set_ylabel("Transfer Default Instability",
                           fontsize=19, weight='semibold')

    axarr[1, 1].set_xlabel("Transfer Isolated AUC",
                           fontsize=19, weight='semibold')
    axarr[1, 1].set_ylabel("Transfer Isolated Instability",
                           fontsize=19, weight='semibold')

    fig.tight_layout(w_pad=1.3, h_pad=1.3)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     "{}_{}__stability-comparison.svg".format(args.classif,
                                                              args.ex_mtype)),
        dpi=600, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_holdout_error(auc_dict, oth_dict, size_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 12), nrows=2, ncols=2)

    cohort_cmap = sns.hls_palette(len(args.cohorts), l=.4, s=.71)
    coh_clrs = {coh: cohort_cmap[sorted(args.cohorts).index(coh)]
                for coh in args.cohorts}

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_size = (0.49 * size_dict[tst_coh, mtype]) ** 0.43

            if trn_coh == tst_coh:
                axarr[0, 0].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 oth_dict['All'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

                axarr[1, 0].plot(auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 oth_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

            elif (trn_coh, tst_coh) in oth_dict['All'][mtype]:
                axarr[0, 1].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 oth_dict['All'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

                axarr[1, 1].plot(auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 oth_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

    plot_min = min(min(auc_val for exp_dict in auc_dict.values()
                       for mtype_dict in exp_dict.values()
                       for auc_val in mtype_dict.values()),
                   min(auc_val for exp_dict in oth_dict.values()
                       for mtype_dict in exp_dict.values()
                       for auc_val in mtype_dict.values())) - 0.02

    for ax in axarr.flatten():
        ax.plot([-1, 2], [-1, 2],
                linewidth=1.5, linestyle='--', color='#550000', alpha=0.49)

        ax.axhline(y=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)
        ax.axvline(x=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_min, 1.005)
        ax.set_ylim(plot_min, 1.005)
        ax.tick_params(labelsize=13, pad=2.9)

    axarr[0, 0].set_xlabel("Default AUC", fontsize=19, weight='semibold')
    axarr[0, 0].set_ylabel("Default Hold-out AUC",
                           fontsize=19, weight='semibold')

    axarr[1, 0].set_xlabel("Isolated AUC", fontsize=19, weight='semibold')
    axarr[1, 0].set_ylabel("Isolated Hold-out AUC",
                           fontsize=19, weight='semibold')

    axarr[0, 1].set_xlabel("Transfer Default AUC",
                           fontsize=19, weight='semibold')
    axarr[0, 1].set_ylabel("Transfer Default Hold-out AUC",
                           fontsize=19, weight='semibold')

    axarr[1, 1].set_xlabel("Transfer Isolated AUC",
                           fontsize=19, weight='semibold')
    axarr[1, 1].set_ylabel("Transfer Isolated Hold-out AUC",
                           fontsize=19, weight='semibold')

    fig.tight_layout(w_pad=3.1, h_pad=2.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     "{}_{}__holdout-error.svg".format(args.classif,
                                                       args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot how isolating subvariants affects classification performance "
        "within and between cohorts for a given transfer experiment."
        )

    parser.add_argument('classif', type=str,
                        help="the mutation classification algorithm used")
    parser.add_argument('ex_mtype', type=str)

    parser.add_argument('cohorts', type=str, nargs='+',
                        help="which TCGA cohort to use")
    parser.add_argument('--samp_cutoff', default=20,
                        help='subtype sample frequency threshold')

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_tag = "{}__samps-{}".format('__'.join(args.cohorts), args.samp_cutoff)
    os.makedirs(os.path.join(plot_dir, out_tag), exist_ok=True)

    out_files = glob(os.path.join(
        base_dir, out_tag, "out-data__*_{}_{}.p".format(
            args.classif, args.ex_mtype)
        ))

    # load mutation scores inferred by the classifier in the experiment
    out_list = [pickle.load(open(out_file, 'rb'))['Infer']
                for out_file in out_files]
    all_df = pd.concat([ols['All'] for ols in out_list])
    iso_df = pd.concat([ols['Iso'] for ols in out_list])

    assert tuple(all_df.index) == tuple(iso_df.index), (
        "There is a discrepancy between the set of mutations tested using "
        "the default approach and those tested using the isolation approach!"
        )
    assert len(set(all_df.index)) == len(tuple(all_df.index)), (
        "There are duplicates in the set of mutations tested!")

    assert all_df.shape[1] == iso_df.shape[1], (
        "Number of samples for which scores were inferred using the "
        "default approach does not equal the corresponding number for "
        "the isolation approach!"
        )

    out_mdls = [out_file.split("out-data__")[1].split(".p")[0]
                for out_file in out_files]

    # load expression and mutation data for each of the cohorts considered
    cdata_dict = {lvl: merge_cohort_data(os.path.join(base_dir, out_tag),
                                         use_lvl=lvl)
                  for lvl in [mdl.split('_{}_'.format(args.classif))[0]
                              for mdl in out_mdls]}
    cdata = tuple(cdata_dict.values())[0]

    # find which cohort each sample belongs to
    use_samps = sorted(cdata.train_samps)
    coh_stat = {
        cohort: np.array([samp in cdata.cohort_samps[cohort.split('_')[0]]
                          for samp in use_samps])
        for cohort in args.cohorts
        }

    auc_dict = {'All': dict(), 'Iso': dict()}
    stab_dict = {'All': dict(), 'Iso': dict()}
    oth_dict = {'All': dict(), 'Iso': dict()}
    size_dict = dict()

    # for each mutation task, calculate classifier performance when using
    # naive approach and when using isolation approach
    for (coh, mtype) in all_df.index:
        if mtype not in auc_dict['All']:
            auc_dict['All'][mtype] = dict()
            auc_dict['Iso'][mtype] = dict()
            stab_dict['All'][mtype] = dict()
            stab_dict['Iso'][mtype] = dict()
            oth_dict['All'][mtype] = dict()
            oth_dict['Iso'][mtype] = dict()

        use_gene, use_type = mtype.subtype_list()[0]
        mtype_lvls = use_type.get_sorted_levels()[1:]

        if '__'.join(mtype_lvls) in cdata_dict:
            use_lvls = '__'.join(mtype_lvls)
        elif not mtype_lvls or mtype_lvls == ('Copy', ):
            use_lvls = 'Location__Protein'

        # find the samples harbouring this mutation, and the inferred scores
        # predicted by the mutation classifier when trained on this cohort
        mtype_stat = np.array(
            cdata_dict[use_lvls].train_mut.status(use_samps, mtype))
        all_vals = all_df.loc[[(coh, mtype)]].values[0]
        iso_vals = iso_df.loc[[(coh, mtype)]].values[0]

        # get the gene associated with this mutation and the samples
        # harbouring any mutation of this gene
        muts = cdata_dict[use_lvls].train_mut[use_gene]
        gene_mtype = MuType(muts.allkey()) - ex_mtypes[args.ex_mtype]
        all_stat = np.array(muts.status(use_samps, gene_mtype))

        # find the cohorts that had a sufficient number of samples with this
        # mutation to test the performance of the trained classifier
        for test_coh in args.cohorts:
            mtype_size = np.sum(coh_stat[test_coh] & mtype_stat)

            if mtype_size >= 20:
                size_dict[test_coh, mtype] = mtype_size

                # calculate the ratio of the variances of scores across CVs
                # within each sample to the variance of scores between samples
                stab_dict['All'][mtype][coh, test_coh] = np.mean([
                    np.std(vals) for vals in all_vals[coh_stat[test_coh]]])
                stab_dict['All'][mtype][coh, test_coh] /= np.std([
                    np.mean(vals) for vals in all_vals[coh_stat[test_coh]]])

                stab_dict['Iso'][mtype][coh, test_coh] = np.mean([
                    np.std(vals) for vals in iso_vals[coh_stat[test_coh]]])
                stab_dict['Iso'][mtype][coh, test_coh] /= np.std([
                    np.mean(vals) for vals in iso_vals[coh_stat[test_coh]]])

                # find the scores corresponding to samples that were
                # negatively-labelled in both versions of the classifier
                wt_stat = coh_stat[test_coh] & ~mtype_stat
                wt_vals = np.concatenate(all_vals[wt_stat])
                none_stat = coh_stat[test_coh] & ~all_stat
                none_vals = np.concatenate(iso_vals[none_stat])

                if test_coh == coh:
                    cv_count = 30
                else:
                    cv_count = 120

                assert len(wt_vals) / np.sum(wt_stat) == cv_count, (
                    "Number of inferred wild-type values does not match the "
                    "number of cross-validations!"
                    )

                assert len(none_vals) / np.sum(none_stat) == cv_count, (
                    "Number of inferred w/o gene values does not match the "
                    "number of cross-validations!"
                    )

                cur_stat = coh_stat[test_coh] & mtype_stat
                cur_all_vals = np.concatenate(all_vals[cur_stat])
                cur_iso_vals = np.concatenate(iso_vals[cur_stat])

                assert len(cur_all_vals) / np.sum(cur_stat) == cv_count, (
                    "Number of naively inferred mutant values does not match "
                    "the number of cross-validations!"
                    )

                assert len(cur_iso_vals) / np.sum(cur_stat) == cv_count, (
                    "Number of isolated inferred mutant values does not "
                    "match the number of cross-validations!"
                    )

                auc_dict['All'][mtype][(coh, test_coh)] = np.greater.outer(
                    cur_all_vals, wt_vals).mean()
                auc_dict['All'][mtype][(coh, test_coh)] += np.equal.outer(
                    cur_all_vals, wt_vals).mean() / 2

                auc_dict['Iso'][mtype][(coh, test_coh)] = np.greater.outer(
                    cur_iso_vals, none_vals).mean()
                auc_dict['Iso'][mtype][(coh, test_coh)] += np.equal.outer(
                    cur_iso_vals, none_vals).mean() / 2

                oth_stat = coh_stat[test_coh] & all_stat & ~mtype_stat
                if np.sum(oth_stat) >= 5:
                    oth_all_vals = np.concatenate(all_vals[oth_stat])
                    oth_iso_vals = np.concatenate(iso_vals[oth_stat])

                    assert len(oth_all_vals) / np.sum(oth_stat) == cv_count, (
                        "Number of naively inferred values for held-out "
                        "samples doesn't match the # of cross-validations!"
                        )

                    assert len(oth_iso_vals) / np.sum(oth_stat) == 120, (
                        "Number of isolated inferred values for held-out "
                        "samples does not match the # of cross-validations!"
                        )

                    oth_dict['All'][mtype][(coh, test_coh)] = np.greater.outer(
                        cur_all_vals, oth_all_vals).mean()
                    oth_dict['All'][mtype][(coh, test_coh)] += np.equal.outer(
                        cur_all_vals, oth_all_vals).mean() / 2

                    oth_dict['Iso'][mtype][(coh, test_coh)] = np.greater.outer(
                        cur_iso_vals, oth_iso_vals).mean()
                    oth_dict['Iso'][mtype][(coh, test_coh)] += np.equal.outer(
                        cur_iso_vals, oth_iso_vals).mean() / 2

    plot_auc_comparisons(auc_dict, size_dict, args)
    plot_cohort_transfer(auc_dict, size_dict, args)
    plot_stability_comparisons(stab_dict, auc_dict, size_dict, args)
    plot_holdout_error(auc_dict, oth_dict, size_dict, args)


if __name__ == '__main__':
    main()

