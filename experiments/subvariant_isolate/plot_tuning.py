
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

from HetMan.experiments.subvariant_isolate import *
from HetMan.experiments.utilities.misc import (
    detect_log_distr, choose_label_colour)
from HetMan.experiments.utilities.colour_maps import auc_cmap

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'

use_marks = [(0, 3, 0)]
use_marks += [(i, 0, k) for k in (0, 70, 140, 210) for i in range(3, 8)]


def plot_chosen_parameters(out_tune, pheno_dict, out_aucs, use_clf, args):
    fig, axarr = plt.subplots(figsize=(1 + 6 * len(use_clf.tune_priors), 10),
                              nrows=3, ncols=len(use_clf.tune_priors),
                              squeeze=False)

    plt_ymin = 0.48
    for j, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        for gene, (pars_dfs, _, _) in out_tune.items():
            gene_clr = choose_label_colour(gene)

            for i, ex_lbl in enumerate(['All', 'Iso', 'IsoShal']):
                for mtype, par_vals in pars_dfs[ex_lbl][par_name].iterrows():
                    auc_val = np.mean(out_aucs[gene][ex_lbl]['CV'].loc[mtype])
                    plt_ymin = min(plt_ymin, auc_val - 0.02)
                    plt_sz = 91 * np.mean(pheno_dict[gene][mtype])

                    axarr[i, j].scatter(par_fnc(par_vals).mean(), auc_val,
                                        c=[gene_clr], s=plt_sz,
                                        alpha=0.23, edgecolor='none')

        for i in range(3):
            axarr[i, j].tick_params(labelsize=15)
            axarr[i, j].axhline(y=1.0, color='black',
                                linewidth=1.7, alpha=0.89)
            axarr[i, j].axhline(y=0.5, color='#550000',
                                linewidth=2.3, linestyle='--', alpha=0.29)

            for par_val in tune_distr:
                axarr[i, j].axvline(x=par_fnc(par_val), color='#116611',
                                    ls=':', linewidth=2.1, alpha=0.31)

            if i == 2:
                axarr[i, j].set_xlabel("Tested {} Value".format(par_name),
                                       fontsize=23, weight='semibold')
            else:
                axarr[i, j].set_xticklabels([])

    for i, ex_lbl in enumerate(['All', 'Iso', 'IsoShal']):
        axarr[i, 0].set_ylabel("Inferred {} AUC".format(ex_lbl),
                               fontsize=17, weight='semibold')

    for ax in axarr.flatten():
        ax.set_ylim(plt_xmin, plt_xmax)
        ax.set_ylim(plt_ymin, 1 + (1 - plt_ymin) / 53)
        ax.grid(axis='x', linewidth=0)
        ax.grid(axis='y', alpha=0.53, linewidth=1.3)

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, "{}__chosen-params_{}.svg".format(
            args.cohort, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_parameter_profile(out_tune, use_clf, args):
    fig, axarr = plt.subplots(figsize=(1 + 5 * len(use_clf.tune_priors), 11),
                              nrows=3, ncols=len(use_clf.tune_priors),
                              squeeze=False)

    plt_ymin = 0.48
    for j, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        for gene, (_, _, tune_dfs) in out_tune.items():
            gene_clr = choose_label_colour(gene)
 
            for i, ex_lbl in enumerate(['All', 'Iso', 'IsoShal']):
                for mtype, tune_vals in tune_dfs[ex_lbl].iterrows():
                    plt_pars = pd.concat([
                        pd.Series({par_fnc(pars[par_name]): avg_val
                                   for pars, avg_val in zip(par_ols,
                                                            avg_ols)})
                        for par_ols, avg_ols in zip(tune_vals['par'],
                                                    tune_vals['avg'])
                        ], axis=1).quantile(q=0.25, axis=1)

                    plt_ymin = min(plt_ymin, plt_pars.min() - 0.01)
                    axarr[i, j].plot(plt_pars.index, plt_pars.values,
                                     linewidth=5/13, alpha=0.17, c=gene_clr)

        for i in range(3):
            axarr[i, j].tick_params(labelsize=15)
            axarr[i, j].axhline(y=1.0, color='black',
                                linewidth=1.7, alpha=0.89)
            axarr[i, j].axhline(y=0.5, color='#550000',
                                linewidth=2.3, linestyle='--', alpha=0.29)

            for par_val in tune_distr:
                axarr[i, j].axvline(x=par_fnc(par_val), color='#116611',
                                    ls=':', linewidth=2.1, alpha=0.31)

            if i == 2:
                axarr[i, j].set_xlabel("Tested {} Value".format(par_name),
                                       fontsize=23, weight='semibold')
            else:
                axarr[i, j].set_xticklabels([])

    for i, ex_lbl in enumerate(['All', 'Iso', 'IsoShal']):
        axarr[i, 0].set_ylabel("Tuned {} AUC".format(ex_lbl),
                               fontsize=21, weight='semibold')

    for ax in axarr.flatten():
        ax.set_ylim(plt_xmin, plt_xmax)
        ax.set_ylim(plt_ymin, 1 + (1 - plt_ymin) / 53)
        ax.grid(axis='x', linewidth=0)
        ax.grid(axis='y', alpha=0.47, linewidth=1.3)

    plt.tight_layout(h_pad=1.3)
    fig.savefig(
        os.path.join(plot_dir, "{}__param-profile_{}.svg".format(
            args.cohort, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_grid(tune_df, auc_vals, size_vals, use_clf, coh_vec, args):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(0.5 + 7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    gene_vec = [[gn for gn, _ in mtypes[0].subtype_list()][0]
                for mtypes in auc_vals.index]
    size_vec = (341 * size_vals.values) / np.max(size_vals)

    # assigns a plotting colour to each gene whose mutations were tested
    use_genes = sorted(set(gene_vec))
    gene_clrs = sns.color_palette("muted", n_colors=len(use_genes))
    clr_vec = [gene_clrs[use_genes.index(gn)] for gn in gene_vec]

    # assigns a plotting marker to each cohort wherein mutations were tested
    use_cohs = sorted(set(coh_vec))
    mark_vec = [use_marks[use_cohs.index(coh)] for coh in coh_vec]
    auc_clrs = auc_vals.apply(auc_cmap)

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
        par_vals1 = np.array(tune_df.loc[:, par_name1].values, dtype=np.float)
        par_vals2 = np.array(tune_df.loc[:, par_name2].values, dtype=np.float)

        if detect_log_distr(tn_distr1):
            use_vals1 = np.log10(par_vals1)
            distr_diff = np.log10(np.array(tn_distr1[-1]))
            distr_diff -= np.log10(np.array(tn_distr1[0]))

            plt_ymin = np.log10(tn_distr1[0]) - distr_diff / 9
            plt_ymax = np.log10(tn_distr1[-1]) + distr_diff / 9

        else:
            use_vals1 = par_vals1
            distr_diff = tn_distr1[-1] - tn_distr1[0]
            plt_ymin = tn_distr1[0] - distr_diff / 9
            plt_ymax = tn_distr1[-1] + distr_diff / 9

        if detect_log_distr(tn_distr2):
            use_vals2 = np.log10(par_vals2)
            distr_diff = np.log10(np.array(tn_distr2[-1]))
            distr_diff -= np.log10(np.array(tn_distr2[0]))

            plt_xmin = np.log10(tn_distr2[0]) - distr_diff / 9
            plt_xmax = np.log10(tn_distr2[-1]) + distr_diff / 9

        else:
            use_vals2 = par_vals2
            distr_diff = tn_distr2[-1] - tn_distr2[0]
            plt_xmin = tn_distr2[0] - distr_diff / 9
            plt_xmax = tn_distr2[-1] + distr_diff / 9

        use_vals1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 11), auc_vals.shape[0])
        use_vals2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 11), auc_vals.shape[0])

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
        os.path.join(plot_dir, "{}__tuning-grid.png".format(args.classif)),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the tuning characteristics of a model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_list = tuple(Path(base_dir).glob(
        os.path.join("*", "out-siml__{}__*__*__{}.p.gz".format(
            args.cohort, args.classif))
        ))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    os.makedirs(os.path.join(plot_dir), exist_ok=True)
    out_use = pd.DataFrame(
        [{'Gene': out_file.parts[-2],
          'Levels': '__'.join(out_file.parts[-1].split('__')[2:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    out_iter = out_use.groupby(['Gene', 'Levels'])['File']
    out_tune = {(gene, lvls): list() for gene, lvls in out_iter.groups}
    out_aucs = {(gene, lvls): list() for gene, lvls in out_iter.groups}
    phn_dict = {gene: dict() for gene in out_use.Gene.unique()}

    tune_dfs = {gene: [{ex_lbl: pd.DataFrame([])
                        for ex_lbl in ['All', 'Iso', 'IsoShal']}
                       for _ in range(3)] + [[]]
                for gene in out_use.Gene.unique()}
    out_clf = None

    auc_dfs = {gene: {ex_lbl: pd.DataFrame([])
                      for ex_lbl in ['All', 'Iso', 'IsoShal']}
               for gene in out_use.Gene.unique()}

    for (gene, lvls), out_files in out_iter:
        for out_file in out_files:
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(base_dir, gene,
                                  '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict[gene].update(pickle.load(f))

            with bz2.BZ2File(Path(base_dir, gene,
                                  '__'.join(["out-tune", out_tag])),
                             'r') as f:
                out_tune[gene, lvls] += [pickle.load(f)]

            with bz2.BZ2File(Path(base_dir, gene,
                                  '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                out_aucs[gene, lvls] += [pickle.load(f)]

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[gene, lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            if out_clf is not None:
                if out_tune[gene, lvls][super_indx][3] != out_clf:
                    raise ValueError("Mismatching classifiers in subvariant "
                                     "isolation experment output!")
            else:
                out_clf = out_tune[gene, lvls][super_indx][3]

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[gene][ex_lbl] = pd.concat([
                    auc_dfs[gene][ex_lbl],
                    pd.DataFrame(out_aucs[gene, lvls][super_indx][ex_lbl])
                    ])

                for i in range(3):
                    tune_dfs[gene][i][ex_lbl] = pd.concat([
                        tune_dfs[gene][i][ex_lbl],
                        out_tune[gene, lvls][super_indx][i][ex_lbl]
                        ])

    auc_dfs = {gene: {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
                      for ex_lbl, auc_df in auc_dict.items()}
               for gene, auc_dict in auc_dfs.items()}

    tune_dfs = {gene: [{ex_lbl: out_df.loc[~out_df.index.duplicated()]
                        for ex_lbl, out_df in tune_dict.items()}
                       for tune_dict in tune_list[:3]]
                for gene, tune_list in tune_dfs.items()}

    plot_chosen_parameters(tune_dfs, phn_dict, auc_dfs, out_clf, args)
    plot_parameter_profile(tune_dfs, out_clf, args)

    if len(out_clf.tune_priors) > 1:
        plot_tuning_grid(*out_lists)


if __name__ == "__main__":
    main()

