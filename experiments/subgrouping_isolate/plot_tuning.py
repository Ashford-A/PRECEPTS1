
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

from HetMan.experiments.subvariant_test.utils import choose_label_colour
from HetMan.experiments.subgrouping_isolate.utils import get_mtype_gene
from HetMan.experiments.utilities.misc import detect_log_distr

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_chosen_parameters(pars_dfs, pheno_dict, auc_dfs, use_clf, args):
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

        for i, ex_lbl in enumerate(['All', 'Iso', 'IsoShal']):
            for mtype, par_vals in pars_dfs[ex_lbl][par_name].iterrows():
                cur_gene = get_mtype_gene(mtype)

                if cur_gene is None:
                    gene_clr = '0.89'
                else:
                    gene_clr = choose_label_colour(cur_gene)

                auc_val = np.mean(auc_dfs[ex_lbl].CV.loc[mtype])
                plt_ymin = min(plt_ymin, auc_val - 0.02)
                plt_sz = 91 * np.mean(pheno_dict[mtype])

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
    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__chosen-params_{}.svg".format(
                                 args.cohort, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_parameter_profile(acc_dfs, use_clf, args):
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

        for i, ex_lbl in enumerate(['All', 'Iso', 'IsoShal']):
            for mtype, acc_vals in acc_dfs[ex_lbl].iterrows():
                cur_gene = get_mtype_gene(mtype)

                if cur_gene is None:
                    gene_clr = '0.89'
                else:
                    gene_clr = choose_label_colour(cur_gene)

                plt_pars = pd.concat([
                    pd.Series({par_fnc(pars[par_name]): avg_val
                               for pars, avg_val in zip(par_ols,
                                                        avg_ols)})
                    for par_ols, avg_ols in zip(acc_vals['par'],
                                                acc_vals['avg'])
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
    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__param-profile_{}.svg".format(
                                 args.cohort, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the tuning characteristics of a model in predicting "
        "the mutation status of the subgroupings in a given cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(Path(out_dir).glob(
        "out-siml__*__*__{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)
    out_use = pd.DataFrame(
        [{'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    out_iter = out_use.groupby('Levels')['File']
    out_tune = {lvls: list() for lvls, _ in out_iter}
    out_aucs = {lvls: list() for lvls, _ in out_iter}
    phn_dict = dict()

    tune_list = [
        {ex_lbl: pd.DataFrame([]) for ex_lbl in ['All', 'Iso', 'IsoShal']}
        for _ in range(3)
        ]
    out_clf = None

    auc_dfs = {ex_lbl: pd.DataFrame([])
               for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for lvls, out_files in out_iter:
        for out_file in out_files:
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-tune", out_tag])),
                             'r') as f:
                out_tune[lvls] += [pickle.load(f)]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                out_aucs[lvls] += [pickle.load(f)]

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            if out_clf is not None:
                if out_tune[lvls][super_indx][3] != out_clf:
                    raise ValueError("Mismatching classifiers in subgrouping "
                                     "isolation experment output!")

            else:
                out_clf = out_tune[lvls][super_indx][3]

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[ex_lbl] = pd.concat([
                    auc_dfs[ex_lbl],
                    pd.DataFrame(out_aucs[lvls][super_indx][ex_lbl])
                    ])

                for i in range(3):
                    tune_list[i][ex_lbl] = pd.concat([
                        tune_list[i][ex_lbl],
                        out_tune[lvls][super_indx][i][ex_lbl]
                        ])

    auc_dfs = {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
               for ex_lbl, auc_df in auc_dfs.items()}

    pars_dfs, time_dfs, acc_dfs = tuple(
        {ex_lbl: out_df.loc[~out_df.index.duplicated()]
         for ex_lbl, out_df in tune_dict.items()}
        for tune_dict in tune_list
        )

    plot_chosen_parameters(pars_dfs, phn_dict, auc_dfs, out_clf, args)
    plot_parameter_profile(acc_dfs, out_clf, args)


if __name__ == "__main__":
    main()

