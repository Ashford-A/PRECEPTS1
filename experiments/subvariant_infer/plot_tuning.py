
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_infer')

else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'tuning')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from HetMan.experiments.subvariant_infer.fit_infer import load_cohort_data
from HetMan.experiments.subvariant_infer.utils import (
    compare_scores, load_infer_tuning, load_infer_output)
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.utilities.classifiers import *
from HetMan.experiments.variant_baseline.plot_model import detect_log_distr

import argparse
from pathlib import Path
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


def plot_tuning_auc(out_list, score_list, use_clf, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

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
        np.repeat([np.mean(stat_dict[mcomb]) for mcomb in tune_df.index], 2)
        for (_, _, tune_df), (stat_dict, _) in zip(out_list, score_list)
        ])
    size_vec = 551 * size_vec / np.max(size_vec)

    auc_vals = np.concatenate([
        auc_df.loc[tune_df.index].values.flatten()
        for (_, _, tune_df), (_, auc_df) in zip(out_list, score_list)
        ])

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        par_vals = np.concatenate([
            tune_df.loc[:, (slice(None), par_name)].values.flatten()
            for (_, _, tune_df) in out_list
            ])

        if detect_log_distr(tune_distr):
            par_vals = np.log10(par_vals)
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        # jitters the paramater values and plots them against mutation AUC
        par_vals += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tune_distr) * 9), len(auc_vals))

        for par_val, auc_val, mark_val, size_val, clr_val in zip(
                par_vals, auc_vals, mark_vec, size_vec, clr_vec):
            ax.scatter(par_val, auc_val, marker=mark_val,
                       s=size_val, c=clr_val, alpha=0.37)

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
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_grid(out_list, score_list, use_clf, args):
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
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser("Plots the tuning characteristics of a "
                                     "classifier used to infer the presence "
                                     "of genes' subvariants across cohorts.")

    parser.add_argument('classif', help='a mutation classifier')
    args = parser.parse_args()

    os.makedirs(os.path.join(plot_dir), exist_ok=True)
    out_path = Path(os.path.join(base_dir, 'output'))
    mut_clf = eval(args.classif)

    out_dirs = [
        (i, out_dir.parent) for i, out_dir in enumerate(out_path.glob(
            "*/*/{}/**/out__task-0.p".format(args.classif)))
        if (len(tuple(out_dir.parent.glob("out__*.p"))) > 0
            and (len(tuple(out_dir.parent.glob("out__*.p")))
                 == len(tuple(out_dir.parent.glob("slurm/fit-*.txt")))))
        ]

    out_paths = [str(out_dir).split("/output/")[1].split('/')
                 for _, out_dir in out_dirs]

    for coh, gene, mut_levels in set((out_path[0], out_path[1], out_path[4])
                                     for out_path in out_paths):
        use_data = [(i, out_path) for i, out_path in enumerate(out_paths)
                    if (out_path[0] == coh and out_path[1] == gene
                        and out_path[4] == mut_levels)]

        if len(use_data) > 1:
            use_samps = np.argmin(int(x[1][2].split('samps_')[-1])
                                  for x in use_data)

            for x in use_data[:use_samps] + use_data[(use_samps + 1):]:
                del(out_dirs[x[0]])

    out_list = [[out_paths[i],
                 load_cohort_data(base_dir, out_paths[i][0],
                                  out_paths[i][1], out_paths[i][4]),
                 load_infer_output(str(out_dir)),
                 load_infer_tuning(str(out_dir))]
                for i, out_dir in out_dirs]

    score_list = [compare_scores(infer_df['Iso'], cdata,
                                 get_similarities=False)[:2]
                  for _, cdata, infer_df, _ in out_list]

    for i in range(len(out_list)):
        del(out_list[i][1])

    plot_tuning_auc(out_list, score_list, mut_clf, args)
    if len(mut_clf.tune_priors) > 1:
        plot_tuning_grid(out_list, score_list, mut_clf, args)


if __name__ == "__main__":
    main()

