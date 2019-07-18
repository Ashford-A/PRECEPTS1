
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_infer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

from HetMan.experiments.subvariant_infer.utils import compare_scores
from HetMan.experiments.subvariant_infer.merge_infer import merge_cohort_data
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.variant_baseline.plot_tuning import detect_log_distr

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from functools import reduce
from operator import add
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
use_marks = [(0, 3, 0)]
use_marks += [(i, 0, k) for k in (0, 70, 140, 210) for i in range(3, 8)]


def plot_tuning_auc(out_list, args):
    tune_priors = out_list[0][0]['Clf'].tune_priors
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(tune_priors)),
                              nrows=len(tune_priors), ncols=1, squeeze=False)

    use_cohs = sorted(set(coh for _, _, (coh, _) in out_list))
    coh_mrks = dict(zip(use_cohs, use_marks[:len(use_cohs)]))

    use_genes = sorted(set(gene for _, _, (_, gene) in out_list))
    gene_clrs = dict(zip(use_genes,
                         sns.color_palette("muted", n_colors=len(use_genes))))

    for ax, (par_name, tune_distr) in zip(axarr.flatten(), tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        for out_dict, (pheno_dict, auc_list), (coh, gene) in out_list:
            tune_df = out_dict['Tune'].loc[:, (slice(None), par_name)]

            for mtype, (all_val, iso_val) in tune_df.iterrows():
                ax.scatter(par_fnc(all_val), auc_list.loc[mtype, 'All'],
                           marker=coh_mrks[coh],
                           s=551 * np.mean(pheno_dict[mtype]),
                           c=gene_clrs[gene], alpha=0.23)

                ax.scatter(par_fnc(iso_val), auc_list.loc[mtype, 'Iso'],
                           marker=coh_mrks[coh],
                           s=551 * np.mean(pheno_dict[mtype]),
                           c=gene_clrs[gene], alpha=0.23)

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
        os.path.join(plot_dir, "{}__tuning-auc.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_grid(out_list, args):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(0.5 + 7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    coh_vec = reduce(add, [[out_path[0]] * infer_df['Iso'].shape[0] * 2
                           for out_path, infer_df, _, in out_list])
    use_cohs = sorted(set(coh_vec))
    mark_vec = [use_marks[use_cohs.index(coh)] for coh in coh_vec]

    gene_vec = reduce(add, [[out_path[1]] * infer_df['Iso'].shape[0] * 2
                            for out_path, infer_df, _, in out_list])
    use_genes = sorted(set(gene_vec))
    gene_clrs = sns.color_palette("muted", n_colors=len(use_genes))
    clr_vec = [gene_clrs[use_genes.index(gn)] for gn in gene_vec]

    size_vec = np.concatenate([
        np.repeat([np.sum(stat_dict[mcomb]) for mcomb in tune_df.index], 2)
        for (_, _, tune_df), (stat_dict, _) in zip(out_list, score_list)
        ])
    size_vec = 341 * size_vec / np.max(size_vec)

    par_vals = {
        par_name: np.concatenate([
            tune_df.loc[:, (slice(None), par_name)].values.flatten()
            for (_, _, tune_df) in out_list
            ])
        for par_name, _ in use_clf.tune_priors
        }

    auc_vals = np.concatenate([
        auc_df.loc[tune_df.index].values.flatten()
        for (_, _, tune_df), (_, auc_df) in zip(out_list, score_list)
        ])
    auc_clrs = [auc_cmap(auc_val) for auc_val in auc_vals]

    for i, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        axarr[i, i].grid(False)

        if detect_log_distr(tune_distr):
            use_distr = [np.log10(par_val) for par_val in tune_distr]
            par_lbl = par_name + '\n(log-scale)'

        else:
            use_distr = tune_distr
            par_lbl = par_name

        distr_diff = np.array(use_distr[-1]) - np.array(use_distr[0])
        plt_min = use_distr[0] - distr_diff / 9
        plt_max = use_distr[-1] + distr_diff / 9

        axarr[i, i].set_xlim(plt_min, plt_max)
        axarr[i, i].set_ylim(plt_min, plt_max)
        axarr[i, i].text(
            (plt_min + plt_max) / 2, (plt_min + plt_max) / 2, par_lbl,
            ha='center', fontsize=31, weight='semibold'
            )

        for par_val in use_distr:
            axarr[i, i].axhline(y=par_val, color='#116611',
                                ls='--', linewidth=1.9, alpha=0.23)
            axarr[i, i].axvline(x=par_val, color='#116611',
                                ls='--', linewidth=1.9, alpha=0.23)

    for (i, (par_name1, tn_distr1)), (j, (par_name2, tn_distr2)) in combn(
            enumerate(use_clf.tune_priors), 2):

        if detect_log_distr(tn_distr1):
            use_vals1 = np.log10(par_vals[par_name1])
            distr_diff = np.log10(np.array(tn_distr1[-1]))
            distr_diff -= np.log10(np.array(tn_distr1[0]))

            plt_ymin = np.log10(tn_distr1[0]) - distr_diff / 9
            plt_ymax = np.log10(tn_distr1[-1]) + distr_diff / 9

        else:
            use_vals1 = par_vals[par_name1]
            distr_diff = tn_distr1[-1] - tn_distr1[0]
            plt_ymin = tn_distr1[0] - distr_diff / 9
            plt_ymax = tn_distr1[-1] + distr_diff / 9

        if detect_log_distr(tn_distr2):
            use_vals2 = np.log10(par_vals[par_name2])
            distr_diff = np.log10(np.array(tn_distr2[-1]))
            distr_diff -= np.log10(np.array(tn_distr2[0]))

            plt_xmin = np.log10(tn_distr2[0]) - distr_diff / 9
            plt_xmax = np.log10(tn_distr2[-1]) + distr_diff / 9

        else:
            use_vals2 = par_vals[par_name2]
            distr_diff = tn_distr2[-1] - tn_distr2[0]
            plt_xmin = tn_distr2[0] - distr_diff / 9
            plt_xmax = tn_distr2[-1] + distr_diff / 9

        use_vals1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 11),
            auc_vals.shape[0]
            )
        use_vals2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 11),
            auc_vals.shape[0]
            )

        for use_val2, use_val1, mark_val, size_val, auc_val in zip(
                use_vals2, use_vals1, mark_vec, size_vec, auc_clrs):
            axarr[i, j].scatter(use_val2, use_val1, marker=mark_val,
                                s=size_val, c=auc_val, alpha=0.35,
                                edgecolor='black')

        for use_val1, use_val2, mark_val, size_val, gene_val in zip(
                use_vals1, use_vals2, mark_vec, size_vec, clr_vec):
            axarr[j, i].scatter(use_val1, use_val2, marker=mark_val,
                                s=size_val, c=gene_val, alpha=0.35,
                                edgecolor='black')

        axarr[i, j].set_xlim(plt_xmin, plt_xmax)
        axarr[i, j].set_ylim(plt_ymin, plt_ymax)
        axarr[j, i].set_ylim(plt_xmin, plt_xmax)
        axarr[j, i].set_xlim(plt_ymin, plt_ymax)

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, "{}__tuning-grid.svg".format(args.classif)),
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
        out_file.parts[-3:] for out_file in Path(base_dir).glob(os.path.join(
            "*__samps-*", "*", "out-data__*__{}.p.gz".format(args.classif)))
        ]

    out_use = pd.DataFrame([
        {'Cohort': out_data[0].split("__samps-")[0],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Gene': out_data[1],
         'Levels': '__'.join(out_data[2].split('__')[1:-1])}
        for out_data in out_datas
        ]).groupby(['Cohort', 'Gene', 'Levels'])['Samps'].min()

    out_list = [[None, None, None] for _ in range(len(out_use))]
    for i, ((coh, gene, lvls), ctf) in enumerate(out_use.iteritems()):
        with bz2.BZ2File(os.path.join(base_dir,
                                      "{}__samps-{}".format(coh, ctf),
                                      gene, "out-data__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as fl:
            out_list[i][0] = pickle.load(fl)

            cdata = merge_cohort_data(
                os.path.join(base_dir, "{}__samps-{}".format(coh, ctf), gene),
                lvls, use_seed=709
                )

            out_list[i][1] = compare_scores(out_list[i][0]['Infer']['Iso'],
                                            cdata, get_similarities=False)[:2]
            out_list[i][2] = coh, gene

    plot_tuning_auc(out_list, args)
    if len(out_list[0][0]['Clf'].tune_priors) > 1:
        plot_tuning_grid(out_list, args)


if __name__ == "__main__":
    main()

