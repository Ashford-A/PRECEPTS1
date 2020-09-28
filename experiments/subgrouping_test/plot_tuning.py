
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from .utils import choose_mtype_colour
from ..utilities.misc import get_distr_transform, choose_label_colour

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'tuning')


def plot_chosen_parameters(pars_df, use_clf, phn_dict, auc_df, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

    par_fxs = {par_name: get_distr_transform(tune_distr)
               for par_name, tune_distr in use_clf.tune_priors}
    plt_ymin = 0.48

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        for mtype, par_vals in pars_df[par_name].iterrows():
            par_val = par_fxs[par_name](par_vals).mean()
            auc_val = np.mean(auc_df.loc[mtype, 'CV'])

            plt_ymin = min(plt_ymin, auc_val - 0.02)
            plt_sz = 307 * np.mean(phn_dict[mtype])

            ax.scatter(par_val, auc_val, c=[choose_mtype_colour(mtype)],
                       s=plt_sz, alpha=0.23, edgecolor='none')

        ax.tick_params(labelsize=19)
        ax.set_xlabel('Tuned {} Value'.format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.73)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.37)

        for par_val in tune_distr:
            ax.axvline(x=par_fxs[par_name](par_val), color='#116611',
                       ls=':', linewidth=3.1, alpha=0.37)

        ax.grid(axis='x', linewidth=0)
        ax.grid(axis='y', alpha=0.47, linewidth=1.3)

        ax.set_xlim((2 * par_fxs[par_name](tune_distr[0])
                     - par_fxs[par_name](tune_distr[1])),
                    (2 * par_fxs[par_name](tune_distr[-1])
                     - par_fxs[par_name](tune_distr[-2])))
        ax.set_ylim(plt_ymin, 1 + (1 - plt_ymin) / 53)

    fig.text(-0.01, 0.5, "Inferred AUC", ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "chosen-parameters_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_parameter_profile(acc_df, use_clf, auc_vals, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)
 
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    par_fxs = {par_name: get_distr_transform(tune_distr)
               for par_name, tune_distr in use_clf.tune_priors}
    plt_ymin = 0.48

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        for gene, auc_vec in use_aucs.groupby(
                lambda mtype: tuple(mtype.label_iter())[0]):
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            base_pars = pd.DataFrame.from_records([
                pd.Series({par_fxs[par_name](pars[par_name]): avg_val
                           for pars, avg_val in zip(par_ols, avg_ols)})
                for par_ols, avg_ols in zip(acc_df.loc[base_mtype, 'par'],
                                            acc_df.loc[base_mtype, 'avg'])
                ]).quantile(q=0.25)

            if len(auc_vec) > 1:
                best_subtype = auc_vec[:base_indx].append(
                    auc_vec[(base_indx + 1):]).idxmax()

                best_pars = pd.DataFrame.from_records([
                    pd.Series({par_fxs[par_name](pars[par_name]): avg_val
                               for pars, avg_val in zip(par_ols, avg_ols)})
                    for par_ols, avg_ols in zip(
                        acc_df.loc[best_subtype, 'par'],
                        acc_df.loc[best_subtype, 'avg']
                        )
                    ]).quantile(q=0.25)

            else:
                best_subtype = base_mtype

            if auc_vec[base_mtype] > 0.6 or auc_vec[best_subtype] > 0.6:
                plt_ymin = min(plt_ymin, base_pars.min() - 0.02)

                gene_clr = choose_label_colour(gene)
                ax.plot(base_pars.index, base_pars.values, c=gene_clr,
                        linewidth=2.3, alpha=0.53)

                if len(auc_vec) > 1:
                    plt_ymin = min(plt_ymin, best_pars.min() - 0.02)
                    ax.plot(best_pars.index, best_pars.values, c=gene_clr,
                            linewidth=2.9, alpha=0.37, linestyle=':')

        ax.set_xlim((2 * par_fxs[par_name](tune_distr[0])
                     - par_fxs[par_name](tune_distr[1])),
                    (2 * par_fxs[par_name](tune_distr[-1])
                     - par_fxs[par_name](tune_distr[-2])))
        ax.set_ylim(plt_ymin, 1 + (1 - plt_ymin) / 53)

        ax.tick_params(labelsize=19)
        ax.set_xlabel("Tested {} Value".format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, "Tuning AUC", ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "parameter-profile_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_accuracy(acc_df, auc_vals, phn_dict, args):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt_min = 0.47

    tune_accs = acc_df.avg.applymap(max).mean(axis=1)
    for mtype, tune_acc in tune_accs.iteritems():
        plt_min = min(plt_min, tune_acc - 0.01, auc_vals[mtype] - 0.01)

        ax.scatter(tune_acc, auc_vals[mtype],
                   s=371 * np.mean(phn_dict[mtype]),
                   c=[choose_mtype_colour(mtype)],
                   alpha=0.19, edgecolor='none')

    ax.set_xlim(plt_min, 1 + (1 - plt_min) / 53)
    ax.set_ylim(plt_min, 1 + (1 - plt_min) / 53)
    ax.tick_params(labelsize=19)

    ax.set_xlabel("Tuning Accuracy", fontsize=27, weight='semibold')
    ax.set_ylabel("Inferred Accuracy", fontsize=27, weight='semibold')

    ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
    ax.axvline(x=1.0, color='black', linewidth=2.1, alpha=0.37)

    ax.axhline(y=0.5, color='black',
               linewidth=1.9, linestyle='--', alpha=0.29)
    ax.axvline(x=0.5, color='black',
               linewidth=1.9, linestyle='--', alpha=0.29)

    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "tuning-accuracy_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_tuning',
        description="Plots a classifier's tuning characteristics in a cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_list = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                              'Levels': '__'.join(out_data[1].split(
                                  'out-trnsf__')[1].split('__')[:-1])}
                             for out_data in out_datas])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_use = out_list.groupby('Levels')['Samps'].min()
    out_pars = dict()
    out_acc = dict()
    out_clf = dict()

    phn_dict = dict()
    auc_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-tune__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as fl:
            (out_pars[lvls], _,
             out_acc[lvls], out_clf[lvls]) = pickle.load(fl)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_dict[lvls] = pd.DataFrame.from_dict(pickle.load(f))

    pars_df = pd.concat(out_pars.values())
    acc_df = pd.concat(out_acc.values())
    auc_df = pd.concat(auc_dict.values())

    out_clf = set(out_clf.values())
    assert len(out_clf) == 1
    out_clf = tuple(out_clf)[0]

    plot_chosen_parameters(pars_df, out_clf, phn_dict, auc_df, args)
    plot_tuning_accuracy(acc_df, auc_df['mean'], phn_dict, args)
    if 'Consequence__Exon' in out_use.index:
        plot_parameter_profile(acc_df, out_clf, auc_df['mean'], args)


if __name__ == "__main__":
    main()

