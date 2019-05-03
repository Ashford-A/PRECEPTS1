
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'subvariant_transfer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'aupr')

from HetMan.experiments.subvariant_transfer import *
from HetMan.experiments.subvariant_transfer.utils import get_form
from HetMan.experiments.subvariant_transfer.plot_auc import (
    lgnd_ptchs, lgnd_lbls)
from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.mutations import MuType

import argparse
import dill as pickle
from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aupr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'


def plot_aupr_comparisons(aupr_dict, size_dict, form_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 13), nrows=2, ncols=2)
    pnt_size = len(aupr_dict) ** -0.19

    for mtype in aupr_dict['All']['All']:
        if form_dict[mtype] in variant_clrs:
            mtype_clr = variant_clrs[form_dict[mtype]]
        else:
            mtype_clr = '0.5'

        for trn_coh, tst_coh in aupr_dict['All']['All'][mtype]:
            mtype_size = 0.87 * (pnt_size * size_dict[tst_coh, mtype]) ** 0.43

            for j, hld in enumerate(['All', 'Hld']):
                axarr[int(trn_coh != tst_coh), j].plot(
                    aupr_dict['All'][hld][mtype][(trn_coh, tst_coh)],
                    aupr_dict['Iso'][hld][mtype][(trn_coh, tst_coh)],
                    marker='o', markersize=mtype_size, color=mtype_clr,
                    alpha=0.21
                    )

    for ax in axarr.flatten():
        ax.tick_params(labelsize=13, pad=2.9)
        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)

        plt_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(0, plt_max)
        ax.set_ylim(0, plt_max)

        ax.plot([-1, 2], [-1, 2],
                linewidth=2.1, linestyle='--', color='#550000', alpha=0.43)

    axarr[0, 0].set_xlabel("Default AUPR\nw/ Held-out Samples",
                           fontsize=17, weight='semibold')
    axarr[0, 0].set_ylabel("Isolation AUPR\nw/ Held-out Samples",
                           fontsize=17, weight='semibold')

    axarr[0, 1].set_xlabel("Default AUPR\nw/o Held-out Samples",
                           fontsize=17, weight='semibold')
    axarr[0, 1].set_ylabel("Isolation AUPR\nw/o Held-out Samples",
                           fontsize=17, weight='semibold')

    axarr[1, 0].set_xlabel("Transfer Default AUPR\nw/ Held-out Samples",
                           fontsize=17, weight='semibold')
    axarr[1, 0].set_ylabel("Transfer Isolation AUPR\nw/ Held-out Samples",
                           fontsize=17, weight='semibold')

    axarr[1, 1].set_xlabel("Transfer Default AUPR\nw/o Held-out Samples",
                           fontsize=17, weight='semibold')
    axarr[1, 1].set_ylabel("Transfer Isolation AUPR\nw/o Held-out Samples",
                           fontsize=17, weight='semibold')

    fig.legend(lgnd_ptchs, lgnd_lbls, frameon=False, fontsize=19, ncol=3,
               loc=8, handletextpad=0.06, markerscale=3.1,
               bbox_to_anchor=(9/19, -1/61))

    fig.tight_layout(w_pad=2.9, h_pad=2.1)
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     "{}_{}__acc-comparisons.svg".format(args.classif,
                                                         args.ex_mtype)),
        dpi=500, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_cohort_transfer(aupr_dict, size_dict, form_dict, args):
    fig_size = 1 + len(args.cohorts) * 2.9
    pnt_size = len(aupr_dict) ** -0.19

    fig, axarr = plt.subplots(
        figsize=(fig_size, fig_size * 17/18),
        nrows=len(args.cohorts) + 1, ncols=len(args.cohorts),
        gridspec_kw=dict(height_ratios=[43 * fig_size ** -0.97]
                         * len(args.cohorts) + [1])
        )

    for mtype in aupr_dict['All']['All']:
        if form_dict[mtype] in variant_clrs:
            mtype_clr = variant_clrs[form_dict[mtype]]
        else:
            mtype_clr = '0.5'

        for trn_coh, tst_coh in aupr_dict['All']['All'][mtype]:
            mtype_size = 0.1 * pnt_size * fig_size * size_dict[tst_coh, mtype]
            mtype_size **= 0.41

            ax_i = sorted(args.cohorts).index(trn_coh)
            ax_j = sorted(args.cohorts).index(tst_coh)

            axarr[ax_i, ax_j].plot(
                aupr_dict['All']['Hld'][mtype][(trn_coh, tst_coh)],
                aupr_dict['Iso']['Hld'][mtype][(trn_coh, tst_coh)],
                marker='o', markersize=mtype_size, color=mtype_clr,
                markeredgecolor='none', alpha=0.29
                )

    for ax in axarr.flatten()[:-len(args.cohorts)]:
        ax.tick_params(labelsize=7 + fig_size / 2.3, pad=fig_size / 4.7)
        ax.grid(color='0.23', linewidth=0.7, alpha=0.21)

        plt_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(0, plt_max)
        ax.set_ylim(0, plt_max)

        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], minor=False)
        ax.plot([-1, 2], [-1, 2],
                linewidth=1.7, linestyle='--', color='#550000', alpha=0.43)

    fig.text(0.5, 0.23 * fig_size ** -0.41,
             "Default AUPR w/o Held-out Samples",
             size=13 + fig_size * 0.63, ha='center', va='top',
             fontweight='semibold')

    fig.text(0, 0.5, "Isolation AUPR w/o Held-out Samples",
             size=13 + fig_size * 0.63, rotation=90,
             ha='right', va='center', fontweight='semibold')

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
        axarr[-1, i].axis('off')

    fig.legend(lgnd_ptchs, lgnd_lbls, frameon=False, ncol=3, loc=8,
               fontsize=4 + fig_size * 1.7, handletextpad=0.09,
               markerscale=3.2, bbox_to_anchor=(5/9, -0.01))

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, "{}__samps-{}".format('__'.join(args.cohorts),
                                                     args.samp_cutoff),
                     "{}_{}__cohort-transfer.svg".format(args.classif,
                                                         args.ex_mtype)),
        dpi=600, bbox_inches='tight', format='svg'
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

    aupr_dict = {'All': {'All': dict(), 'Hld': dict()},
                 'Iso': {'All': dict(), 'Hld': dict()}}
    size_dict = dict()
    form_dict = dict()

    # for each mutation task, calculate classifier performance when using
    # naive approach and when using isolation approach
    for (coh, mtype) in all_df.index:
        if mtype not in form_dict:
            form_dict[mtype] = get_form(mtype)

        if mtype not in aupr_dict['All']['All']:
            aupr_dict['All']['All'][mtype] = dict()
            aupr_dict['All']['Hld'][mtype] = dict()
            aupr_dict['Iso']['All'][mtype] = dict()
            aupr_dict['Iso']['Hld'][mtype] = dict()

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

        all_vals = np.array([np.mean(vls)
                             for vls in all_df.loc[[(coh, mtype)]].values[0]])
        iso_vals = np.array([np.mean(vls)
                             for vls in iso_df.loc[[(coh, mtype)]].values[0]])

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

                coh_all_vals = all_vals[coh_stat[test_coh]]
                coh_iso_vals = iso_vals[coh_stat[test_coh]]
                cur_stat = mtype_stat[coh_stat[test_coh]]
                hld_stat = ~(all_stat & ~mtype_stat)[coh_stat[test_coh]]

                aupr_dict['All']['All'][mtype][coh, test_coh] = aupr(
                    cur_stat, coh_all_vals)
                aupr_dict['All']['Hld'][mtype][coh, test_coh] = aupr(
                    cur_stat[hld_stat], coh_all_vals[hld_stat])

                aupr_dict['Iso']['All'][mtype][coh, test_coh] = aupr(
                    cur_stat, coh_iso_vals)
                aupr_dict['Iso']['Hld'][mtype][coh, test_coh] = aupr(
                    cur_stat[hld_stat], coh_iso_vals[hld_stat])

    plot_aupr_comparisons(aupr_dict, size_dict, form_dict, args)
    plot_cohort_transfer(aupr_dict, size_dict, form_dict, args)


if __name__ == '__main__':
    main()

