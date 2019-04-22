
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'subvariant_transfer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_transfer import *
from HetMan.experiments.utilities.scatter_plotting import place_annot
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

use_marks = [(0, 3, 0)]
use_marks += [(i, 0, k) for k in (0, 140) for i in (3, 4, 5)]


def plot_auc_comparisons(auc_dict, args, cdata):
    fig, axarr = plt.subplots(figsize=(10, 9), nrows=2, ncols=2)

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
            mtype_size = (0.71 * mtype_size) ** 0.43

            if trn_coh == tst_coh:
                axarr[0, 0].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.31)

                for trn_coh2, tst_coh2 in auc_dict['All'][mtype]:
                    if trn_coh2 == trn_coh and tst_coh2 != tst_coh:
                        axarr[0, 1].plot(
                            auc_dict['All'][mtype][(trn_coh, tst_coh)],
                            auc_dict['All'][mtype][(trn_coh, tst_coh2)],
                            marker='o', markersize=mtype_size, alpha=0.31,
                            color=coh_clrs[tst_coh]
                            )

                        axarr[1, 0].plot(
                            auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                            auc_dict['Iso'][mtype][(trn_coh, tst_coh2)],
                            marker='o', markersize=mtype_size, alpha=0.31,
                            color=coh_clrs[tst_coh]
                            )

            else:
                axarr[1, 1].plot(auc_dict['All'][mtype][(trn_coh, tst_coh)],
                                 auc_dict['Iso'][mtype][(trn_coh, tst_coh)],
                                 marker='o', markersize=mtype_size,
                                 color=coh_clrs[tst_coh], alpha=0.31)

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
        ax.tick_params(labelsize=10, pad=2.7)

    axarr[0, 0].set_xlabel("Default AUC", fontsize=17, weight='semibold')
    axarr[0, 0].set_ylabel("Isolated AUC", fontsize=17, weight='semibold')

    axarr[0, 1].set_xlabel("Default AUC", fontsize=17, weight='semibold')
    axarr[0, 1].set_ylabel("Transfer Default AUC",
                           fontsize=17, weight='semibold')

    axarr[1, 0].set_xlabel("Isolated AUC", fontsize=17, weight='semibold')
    axarr[1, 0].set_ylabel("Transfer Isolated AUC",
                           fontsize=17, weight='semibold')

    axarr[1, 1].set_xlabel("Transfer Default AUC",
                           fontsize=17, weight='semibold')
    axarr[1, 1].set_ylabel("Transfer Isolated AUC",
                           fontsize=17, weight='semibold')

    fig.tight_layout(w_pad=2.9, h_pad=2.3)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     args.gene,
                     "{}__{}_{}__acc-comparisons.svg".format(
                         args.mut_levels, args.classif, args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot how isolating subvariants affects classification performance "
        "within and between cohorts for a gene in a transfer experiment."
        )

    parser.add_argument('gene', type=str, help="a mutated gene")
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
    os.makedirs(os.path.join(plot_dir, out_tag, args.gene),
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
    stab_dict = {'All': dict(), 'Iso': dict()}

    gene_muts = cdata.train_mut[args.gene]
    gene_mtype = MuType(gene_muts.allkey()) - ex_mtypes[args.ex_mtype]
    gene_stat = np.array(gene_muts.status(use_samps, gene_mtype))

    for (coh, mtype) in out_dict['Infer']['All'].index:
        if mtype.subtype_list()[0][0] == args.gene:
            if mtype not in auc_dict['All']:
                auc_dict['All'][mtype] = dict()
                auc_dict['Iso'][mtype] = dict()
                stab_dict['All'][mtype] = dict()
                stab_dict['Iso'][mtype] = dict()

            mtype_stat = np.array(cdata.train_mut.status(use_samps, mtype))
            all_vals = out_dict['Infer']['All'].loc[[(coh, mtype)]].values[0]
            iso_vals = out_dict['Infer']['Iso'].loc[[(coh, mtype)]].values[0]

            for tst_coh in args.cohorts:
                if np.sum(coh_stat[tst_coh] & mtype_stat) >= 20:
                    stab_dict['All'][mtype][coh, tst_coh] = np.mean([
                        np.std(vals) for vals in all_vals[coh_stat[tst_coh]]])
                    stab_dict['All'][mtype][coh, tst_coh] /= np.std([
                        np.mean(vals)
                        for vals in all_vals[coh_stat[tst_coh]]
                        ])

                    stab_dict['Iso'][mtype][coh, tst_coh] = np.mean([
                        np.std(vals) for vals in iso_vals[coh_stat[tst_coh]]])
                    stab_dict['Iso'][mtype][coh, tst_coh] /= np.std([
                        np.mean(vals)
                        for vals in iso_vals[coh_stat[tst_coh]]
                        ])

                    wt_stat = coh_stat[tst_coh] & ~mtype_stat
                    wt_vals = np.concatenate(all_vals[wt_stat])
                    none_stat = coh_stat[tst_coh] & ~gene_stat
                    none_vals = np.concatenate(iso_vals[none_stat])

                    if tst_coh == coh:
                        cv_count = 30
                    else:
                        cv_count = 120

                    cur_stat = coh_stat[tst_coh] & mtype_stat
                    cur_all_vals = np.concatenate(all_vals[cur_stat])
                    cur_iso_vals = np.concatenate(iso_vals[cur_stat])

                    auc_dict['All'][mtype][(coh, tst_coh)] = np.greater.outer(
                        cur_all_vals, wt_vals).mean()
                    auc_dict['All'][mtype][(coh, tst_coh)] += np.equal.outer(
                        cur_all_vals, wt_vals).mean() / 2

                    auc_dict['Iso'][mtype][(coh, tst_coh)] = np.greater.outer(
                        cur_iso_vals, none_vals).mean()
                    auc_dict['Iso'][mtype][(coh, tst_coh)] += np.equal.outer(
                        cur_iso_vals, none_vals).mean() / 2

    plot_auc_comparisons(auc_dict, args, cdata)


if __name__ == '__main__':
    main()

