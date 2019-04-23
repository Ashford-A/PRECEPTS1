
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'subvariant_transfer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'stability')

from HetMan.experiments.subvariant_transfer import *
from dryadic.features.mutations import MuType

import argparse
import dill as pickle
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'


def plot_auc_stability(auc_dict, args, cdata):
    fig, (wthn_ax, btwn_ax) = plt.subplots(figsize=(14, 11), nrows=2, ncols=1)

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_samps = mtype.get_samples(cdata.train_mut)

            mtype_size = (
                len(mtype_samps & cdata.cohort_samps[trn_coh.split('_')[0]])
                + len(mtype_samps & cdata.cohort_samps[tst_coh.split('_')[0]])
                )
            mtype_size = (0.27 * mtype_size) ** 0.43

            if trn_coh == tst_coh:
                cur_ax = wthn_ax
            else:
                cur_ax = btwn_ax

            cur_ax.plot(np.mean(auc_dict['All'][mtype][trn_coh, tst_coh]),
                        np.std(auc_dict['All'][mtype][trn_coh, tst_coh]),
                        marker='o', markersize=mtype_size,
                        color='#A6D000', alpha=0.23)
 
            cur_ax.plot(np.mean(auc_dict['Iso'][mtype][trn_coh, tst_coh]),
                        np.std(auc_dict['Iso'][mtype][trn_coh, tst_coh]),
                        marker='o', markersize=mtype_size,
                        color='#960084', alpha=0.23)

    plot_xmin = min(np.percentile(auc_vals, q=49)
                    for exp_dict in auc_dict.values()
                    for mtype_dict in exp_dict.values()
                    for auc_vals in mtype_dict.values())

    for ax in [wthn_ax, btwn_ax]:
        ax.axvline(x=0.5,
                   linewidth=1.3, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_xmin, 1.005)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.tick_params(labelsize=13, pad=2.9)

    fig.tight_layout(h_pad=2.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     args.mut_levels,
                     "{}_{}__auc-stability.svg".format(args.classif,
                                                       args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_rank_concordance(auc_dict, stab_dict, args, cdata):
    fig, (wthn_ax, btwn_ax) = plt.subplots(figsize=(14, 11), nrows=2, ncols=1)

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_samps = mtype.get_samples(cdata.train_mut)

            mtype_size = (
                len(mtype_samps & cdata.cohort_samps[trn_coh.split('_')[0]])
                + len(mtype_samps & cdata.cohort_samps[tst_coh.split('_')[0]])
                )
            mtype_size = (0.27 * mtype_size) ** 0.43

            if trn_coh == tst_coh:
                cur_ax = wthn_ax
            else:
                cur_ax = btwn_ax

            cur_ax.plot(np.mean(auc_dict['All'][mtype][trn_coh, tst_coh]),
                        stab_dict['All'][mtype][trn_coh, tst_coh],
                        marker='o', markersize=mtype_size,
                        color='#A6D000', alpha=0.23)
 
            cur_ax.plot(np.mean(auc_dict['Iso'][mtype][trn_coh, tst_coh]),
                        stab_dict['Iso'][mtype][trn_coh, tst_coh],
                        marker='o', markersize=mtype_size,
                        color='#960084', alpha=0.23)

    plot_xmin = min(np.percentile(auc_vals, q=49)
                    for exp_dict in auc_dict.values()
                    for mtype_dict in exp_dict.values()
                    for auc_vals in mtype_dict.values())

    for ax in [wthn_ax, btwn_ax]:
        ax.axvline(x=0.5,
                   linewidth=1.3, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_xmin, 1.005)
        ax.set_ylim(ax.get_ylim()[0], 1)
        ax.tick_params(labelsize=13, pad=2.9)

    fig.tight_layout(h_pad=2.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     args.mut_levels,
                     "{}_{}__rank-concordance.svg".format(args.classif,
                                                          args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_between_concordance(auc_dict, stab_dict, args, cdata):
    fig, (wthn_ax, btwn_ax) = plt.subplots(figsize=(14, 11), nrows=2, ncols=1)

    for mtype in auc_dict['All']:
        for trn_coh, tst_coh in auc_dict['All'][mtype]:
            mtype_samps = mtype.get_samples(cdata.train_mut)

            mtype_size = (
                len(mtype_samps & cdata.cohort_samps[trn_coh.split('_')[0]])
                + len(mtype_samps & cdata.cohort_samps[tst_coh.split('_')[0]])
                )
            mtype_size = (0.27 * mtype_size) ** 0.43

            if trn_coh == tst_coh:
                cur_ax = wthn_ax
            else:
                cur_ax = btwn_ax

            cur_ax.plot(np.mean(auc_dict['All'][mtype][trn_coh, tst_coh]),
                        stab_dict['Btwn'][mtype][trn_coh, tst_coh],
                        marker='o', markersize=mtype_size,
                        color='#A6D000', alpha=0.23)
 
            cur_ax.plot(np.mean(auc_dict['Iso'][mtype][trn_coh, tst_coh]),
                        stab_dict['Btwn'][mtype][trn_coh, tst_coh],
                        marker='o', markersize=mtype_size,
                        color='#960084', alpha=0.23)

    plot_xmin = min(np.percentile(auc_vals, q=49)
                    for exp_dict in auc_dict.values()
                    for mtype_dict in exp_dict.values()
                    for auc_vals in mtype_dict.values())

    for ax in [wthn_ax, btwn_ax]:
        ax.axvline(x=0.5,
                   linewidth=1.3, linestyle='--', color='black', alpha=0.29)

        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)
        ax.set_xlim(plot_xmin, 1.005)
        ax.set_ylim(ax.get_ylim()[0], 1)
        ax.tick_params(labelsize=13, pad=2.9)

    fig.tight_layout(h_pad=2.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     args.mut_levels,
                     "{}_{}__between-concordance.svg".format(args.classif,
                                                             args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot how isolating subvariants affects inferred mutation score "
        "stability for a given transfer experiment."
        )

    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
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
        "out-data__{}_{}_{}.p".format(
            args.mut_levels, args.classif, args.ex_mtype)
        ), 'rb'))

    auc_dict = {'All': dict(), 'Iso': dict()}
    stab_dict = {'All': dict(), 'Iso': dict(), 'Btwn': dict()}

    for (coh, mtype) in out_dict['Infer']['All'].index:
        if mtype not in auc_dict['All']:
            auc_dict['All'][mtype] = dict()
            auc_dict['Iso'][mtype] = dict()
            stab_dict['All'][mtype] = dict()
            stab_dict['Iso'][mtype] = dict()
            stab_dict['Btwn'][mtype] = dict()

        mtype_stat = np.array(cdata.train_mut.status(use_samps, mtype))
        all_vals = out_dict['Infer']['All'].loc[[(coh, mtype)]].values[0]
        iso_vals = out_dict['Infer']['Iso'].loc[[(coh, mtype)]].values[0]

        use_gene = mtype.subtype_list()[0][0]
        muts = cdata.train_mut[use_gene]
        gene_mtype = MuType(muts.allkey()) - ex_mtypes[args.ex_mtype]
        all_stat = np.array(muts.status(use_samps, gene_mtype))

        for test_coh in args.cohorts:
            if np.sum(coh_stat[test_coh] & mtype_stat) >= 20:
                if test_coh == coh:
                    cv_count = 30
                else:
                    cv_count = 120

                wt_stat = coh_stat[test_coh] & ~mtype_stat
                wt_vals = np.vstack(all_vals[wt_stat])
                assert wt_vals.shape == (np.sum(wt_stat), cv_count)

                none_stat = coh_stat[test_coh] & ~all_stat
                none_vals = np.vstack(iso_vals[none_stat])
                assert none_vals.shape == (np.sum(none_stat), cv_count)

                cur_stat = coh_stat[test_coh] & mtype_stat
                cur_all_vals = np.vstack(all_vals[cur_stat])
                cur_iso_vals = np.vstack(iso_vals[cur_stat])
                assert cur_all_vals.shape == (np.sum(cur_stat), cv_count)
                assert cur_iso_vals.shape == (np.sum(cur_stat), cv_count)

                auc_dict['All'][mtype][(coh, test_coh)] = [
                    np.greater.outer(cur_all_vals[:, i], wt_vals[:, i]).mean()
                    for i in range(cv_count)
                    ]

                auc_dict['Iso'][mtype][(coh, test_coh)] = [
                    np.greater.outer(cur_iso_vals[:, i],
                                     none_vals[:, i]).mean()
                    for i in range(cv_count)
                    ]

                rank_df = pd.concat([pd.DataFrame(cur_all_vals).rank(),
                                     pd.DataFrame(cur_iso_vals).rank()],
                                    axis=1)
                corr_mat = rank_df.corr().values

                stab_dict['All'][mtype][(coh, test_coh)] = corr_mat[
                    :cv_count, :cv_count].mean()
                stab_dict['Iso'][mtype][(coh, test_coh)] = corr_mat[
                    cv_count:, cv_count:].mean()
                stab_dict['Btwn'][mtype][(coh, test_coh)] = corr_mat[
                    :cv_count, cv_count:].mean()

    plot_auc_stability(auc_dict, args, cdata)
    plot_rank_concordance(auc_dict, stab_dict, args, cdata)
    plot_between_concordance(auc_dict, stab_dict, args, cdata)


if __name__ == '__main__':
    main()

