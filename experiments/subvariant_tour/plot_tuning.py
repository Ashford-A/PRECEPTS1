
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

from HetMan.experiments.subvariant_tour import cis_lbls
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from HetMan.experiments.variant_baseline.plot_tuning import detect_log_distr

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

use_marks = ['o']
use_marks += [(i, 0, k) for k in (0, 70, 140, 210) for i in range(3, 8)]


def plot_tuned_auc(out_dict, phn_dict, auc_dict, args):
    tune_priors = tuple(out_dict.values())[0]['Clf'].tune_priors
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(tune_priors)),
                              nrows=len(tune_priors), ncols=1, squeeze=False)

    use_srcs = sorted(set(src for src, _, _ in out_dict.keys()))
    src_mrks = dict(zip(use_srcs, use_marks[:len(use_srcs)]))

    use_cohs = sorted(set(coh for _, coh, _ in out_dict.keys()))
    coh_clrs = dict(zip(use_cohs,
                        sns.color_palette("muted", n_colors=len(use_cohs))))

    for ax, (par_name, tune_distr) in zip(axarr.flatten(), tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        for (src, coh, lvls), ols in out_dict.items():
            tune_df = ols['Pars'].loc[:, (slice(None), par_name)]
            phn_list = phn_dict[src, coh, lvls]
            auc_df = auc_dict[src, coh, lvls]

            for mtype, vals in tune_df.iterrows():
                for (cis_lbl, _), val in vals.iteritems():
                    ax.scatter(par_fnc(val), auc_df.loc[mtype, cis_lbl],
                               marker=src_mrks[src],
                               s=371 * np.mean(phn_list[mtype]),
                               c=[coh_clrs[coh]], alpha=0.17,
                               edgecolor='none')

        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0.48, 1.02)
        ax.tick_params(labelsize=19)
        ax.set_xlabel('Tuned {} Value'.format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, 'Aggregate AUC', ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__tuned-auc.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_profile(out_dict, auc_dict, args):
    tune_priors = tuple(out_dict.values())[0]['Clf'].tune_priors
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(tune_priors)),
                              nrows=len(tune_priors), ncols=1, squeeze=False)

    use_aucs = pd.concat(auc_dict.values()).round(4)
    auc_bins = pd.qcut(
        use_aucs.values.flatten(), q=[0., 0.5, 0.75, 0.8, 0.85, 0.9,
                                      0.92, 0.94, 0.96, 0.98, 0.99, 1.],
        precision=5
        ).categories

    use_srcs = sorted(set(src for src, _, _ in out_dict.keys()))
    src_mrks = dict(zip(use_srcs, use_marks[:len(use_srcs)]))

    use_cohs = sorted(set(coh for _, coh, _ in out_dict.keys()))
    coh_clrs = dict(zip(use_cohs,
                        sns.color_palette("muted", n_colors=len(use_cohs))))

    for ax, (par_name, tune_distr) in zip(axarr.flatten(), tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        plot_df = pd.DataFrame([])
        for (src, coh, lvls), ols in out_dict.items():
            for cis_lbl in cis_lbls:
                tune_vals = pd.DataFrame.from_records(
                    ols['Acc'].loc[:, (cis_lbl, 'avg')])

                tune_vals -= pd.DataFrame.from_records(
                    ols['Acc'].loc[:, (cis_lbl, 'std')])
                par_vals = pd.DataFrame.from_records(
                    ols['Acc'].loc[:, (cis_lbl, 'par')]).applymap(
                        itemgetter(par_name)).applymap(par_fnc)

                tune_vals.index = ols['Acc'].index
                par_vals.index = ols['Acc'].index
                par_df = pd.concat([par_vals.stack(), tune_vals.stack()],
                                   axis=1, keys=['par', 'auc'])

                par_df['auc_bin'] = [
                    auc_bins.get_loc(round(auc_dict[src, coh, lvls].loc[
                        mtype, cis_lbl], 4))
                    for mtype, _ in par_df.index
                    ]
                plot_df = pd.concat([plot_df, par_df])

        for auc_bin, bin_vals in plot_df.groupby('auc_bin'):
            plot_vals = bin_vals.groupby('par').mean()
            ax.plot(plot_vals.index, plot_vals.auc)

        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0.45, 1.01)
        ax.tick_params(labelsize=19)
        ax.set_xlabel('Tested {} Value'.format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, 'Aggregate AUC', ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__tuning-profile.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser("Plots the tuning characteristics of a "
                                     "classifier used to infer the presence "
                                     "of genes' subvariants across cohorts.")

    parser.add_argument('classif', help='a mutation classifier')
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir), exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "*__samps-*", "out-data__*__{}.p.gz".format(args.classif)))
        ]

    out_use = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split('__')[1:-1])}
        for out_data in out_datas
        ]).groupby(['Source', 'Cohort', 'Levels'])['Samps'].min()

    out_dict = dict()
    cdata_dict = dict()
    phn_dict = dict()
    auc_dict = dict()

    for (src, coh, lvls), ctf in out_use.iteritems():
        with bz2.BZ2File(
                os.path.join(base_dir,
                             "{}__{}__samps-{}".format(src, coh, ctf),
                             "out-data__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            out_dict[src, coh, lvls] = pickle.load(fl)

        with bz2.BZ2File(
                os.path.join(base_dir,
                             "{}__{}__samps-{}".format(src, coh, ctf),
                             "out-pheno__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            phn_dict[src, coh, lvls] = pickle.load(fl)

        with bz2.BZ2File(
                os.path.join(base_dir,
                             "{}__{}__samps-{}".format(src, coh, ctf),
                             "out-aucs__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            auc_dict[src, coh, lvls] = pickle.load(fl)

        cdata_dict[src, coh, lvls] = merge_cohort_data(
            os.path.join(base_dir, "{}__{}__samps-{}".format(src, coh, ctf)),
            use_seed=8713
            )

    plot_tuned_auc(out_dict, phn_dict, auc_dict, args)
    plot_tuning_profile(out_dict, auc_dict, args)


if __name__ == "__main__":
    main()

