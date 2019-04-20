
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
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'


def plot_auc_comparisons(auc_dict, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 12), nrows=2, ncols=2)

    cohort_cmap = sns.hls_palette(len(args.cohorts), l=.4, s=.71)
    coh_clrs = {coh: cohort_cmap[sorted(args.cohorts).index(coh)]
                for coh in args.cohorts}

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_samps = mtype.get_samples(cdata.train_mut)

            mtype_size = (
                len(mtype_samps & cdata.cohort_samps[trn_coh.split('_')[0]])
                + len(mtype_samps & cdata.cohort_samps[tst_coh.split('_')[0]])
                )
            mtype_size = (0.27 * mtype_size) ** 0.41

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
                     args.mut_levels,
                     "{}__acc-comparisons.svg".format(args.classif)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_cohort_transfer(auc_dict, args, cdata):
    fig_size = 1 + len(args.cohorts) * 2.9
    fig, axarr = plt.subplots(figsize=(fig_size, fig_size),
                              nrows=len(args.cohorts),
                              ncols=len(args.cohorts))

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_samps = mtype.get_samples(cdata.train_mut)
            ax_i = sorted(args.cohorts).index(trn_coh)
            ax_j = sorted(args.cohorts).index(tst_coh)

            mtype_size = (
                len(mtype_samps & cdata.cohort_samps[trn_coh.split('_')[0]])
                + len(mtype_samps & cdata.cohort_samps[tst_coh.split('_')[0]])
                )
            mtype_size = (0.051 * fig_size * mtype_size) ** 0.43

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
                     args.mut_levels,
                     "{}__cohort-transfer.svg".format(args.classif)),
        dpi=600, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_stability_comparisons(stab_dict, auc_dict, args, cdata):
    fig, axarr = plt.subplots(figsize=(13, 12), nrows=2, ncols=2)

    cohort_cmap = sns.hls_palette(len(args.cohorts), l=.4, s=.71)
    coh_clrs = {coh: cohort_cmap[sorted(args.cohorts).index(coh)]
                for coh in args.cohorts}

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_samps = mtype.get_samples(cdata.train_mut)

            mtype_size = (
                len(mtype_samps & cdata.cohort_samps[trn_coh.split('_')[0]])
                + len(mtype_samps & cdata.cohort_samps[tst_coh.split('_')[0]])
                )
            mtype_size = (0.27 * mtype_size) ** 0.41

            if trn_coh == tst_coh:
                axarr[0, 0].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['All'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

                axarr[1, 0].plot(auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size, alpha=0.19,
                                 color=coh_clrs[tst_coh])

            else:
                axarr[0, 1].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['All'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size, alpha=0.19,
                                 color=coh_clrs[tst_coh])

                axarr[1, 1].plot(auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 stab_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.19)

    plot_xmin = min(auc_val for exp_dict in auc_dict.values()
                    for mtype_dict in exp_dict.values()
                    for auc_val in mtype_dict.values()) - 0.02

    for ax in axarr.flatten():
        ax.axvline(x=0.5,
                   linewidth=0.9, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_xmin, 1.005)
        ax.tick_params(labelsize=13, pad=2.9)

    axarr[0, 0].set_xlabel("Default AUC", fontsize=19, weight='semibold')
    axarr[0, 0].set_ylabel("Default Instability",
                           fontsize=19, weight='semibold')

    axarr[0, 1].set_xlabel("Transfer Default AUC",
                           fontsize=19, weight='semibold')
    axarr[0, 1].set_ylabel("Transfer Default Instability",
                           fontsize=19, weight='semibold')

    axarr[1, 0].set_xlabel("Isolated AUC", fontsize=19, weight='semibold')
    axarr[1, 0].set_ylabel("Isolated Instability",
                           fontsize=19, weight='semibold')

    axarr[1, 1].set_xlabel("Transfer Isolated AUC",
                           fontsize=19, weight='semibold')
    axarr[1, 1].set_ylabel("Transfer Isolated Instability",
                           fontsize=19, weight='semibold')

    fig.tight_layout(w_pad=1.3, h_pad=1.3)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     args.mut_levels,
                     "{}__stability-comparison.svg".format(args.classif)),
        dpi=600, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot how isolating subvariants affects classification performance "
        "within and between cohorts for a given transfer experiment."
        )

    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
    parser.add_argument('classif', type=str,
                        help="the mutation classification algorithm used")

    parser.add_argument('cohorts', type=str, nargs='+',
                        help="which TCGA cohort to use")
    parser.add_argument('--samp_cutoff', default=20,
                        help='subtype sample frequency threshold')

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_tag = "{}__samps-{}".format('__'.join(args.cohorts), args.samp_cutoff)
    os.makedirs(os.path.join(plot_dir, out_tag, args.mut_levels),
                exist_ok=True)

    # load expression and mutation data for each of the cohorts considered
    cdata = merge_cohort_data(os.path.join(base_dir, out_tag))
    use_samps = sorted(cdata.train_samps)

    # find which cohort each sample belongs to
    coh_stat = {
        cohort: np.array([samp in cdata.cohort_samps[cohort.split('_')[0]]
                          for samp in use_samps])
        for cohort in args.cohorts
        }

    # load mutation scores inferred by the classifier in the experiment
    out_dict = pickle.load(open(os.path.join(
        base_dir, out_tag,
        "out-data__{}_{}.p".format(args.mut_levels, args.classif)
        ), 'rb'))

    auc_dict = {'All': dict(), 'Iso': dict()}
    stab_dict = {'All': dict(), 'Iso': dict()}

    # for each mutation task, calculate classifier performance when using
    # naive approach and when using isolation approach
    for (coh, mtype) in out_dict['Infer']['All'].index:
        if mtype not in auc_dict['All']:
            auc_dict['All'][mtype] = dict()
            auc_dict['Iso'][mtype] = dict()
            stab_dict['All'][mtype] = dict()
            stab_dict['Iso'][mtype] = dict()

        # find the samples harbouring this mutation, and the inferred scores
        # predicted by the mutation classifier when trained on this cohort
        mtype_stat = np.array(cdata.train_mut.status(use_samps, mtype))
        all_vals = out_dict['Infer']['All'].loc[[(coh, mtype)]].values[0]
        iso_vals = out_dict['Infer']['Iso'].loc[[(coh, mtype)]].values[0]

        # get the gene associated with this mutation and the samples
        # harbouring any mutation of this gene
        use_gene = mtype.subtype_list()[0][0]
        muts = cdata.train_mut[use_gene]
        gene_mtype = MuType(muts.allkey()) - MuType({('Scale', 'Copy'): {(
            'Copy', ('ShalGain', 'ShalDel')): None}})
        all_stat = np.array(muts.status(use_samps, gene_mtype))

        for test_coh in args.cohorts:
            if np.sum(coh_stat[test_coh] & mtype_stat) >= 20:
                stab_dict['All'][mtype][coh, test_coh] = np.mean([
                    np.std(vals) for vals in all_vals[coh_stat[test_coh]]])
                stab_dict['All'][mtype][coh, test_coh] /= np.std([
                    np.mean(vals) for vals in all_vals[coh_stat[test_coh]]])

                stab_dict['Iso'][mtype][coh, test_coh] = np.mean([
                    np.std(vals) for vals in iso_vals[coh_stat[test_coh]]])
                stab_dict['Iso'][mtype][coh, test_coh] /= np.std([
                    np.mean(vals) for vals in iso_vals[coh_stat[test_coh]]])

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

    plot_auc_comparisons(auc_dict, args, cdata)
    plot_cohort_transfer(auc_dict, args, cdata)
    plot_stability_comparisons(stab_dict, auc_dict, args, cdata)


if __name__ == '__main__':
    main()

