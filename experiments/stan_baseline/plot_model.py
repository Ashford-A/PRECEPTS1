
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'stan_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'model')

from HetMan.experiments.stan_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.variant_baseline.plot_model import detect_log_distr

import argparse
import dill as pickle
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'

fmth_clrs = dict(zip(['optim', 'varit', 'sampl'],
                     sns.hls_palette(6, l=.51, s=.83)[1::2]))


def plot_auc_distribution(acc_df, args):
    auc_df = pd.concat([pd.concat([f_dict['test'].AUC],
                                  keys=[fmth]).reorder_levels([1, 0])
                        for fmth, f_dict in acc_df.items()])
    fig, ax = plt.subplots(figsize=(auc_df.shape[0] / 5.3 + 2, 11))

    auc_order = auc_df.mean(axis=1).groupby(level=0).max().sort_values(
        ascending=False)
    auc_df.index = auc_df.index.set_names(['Mutation', 'Fit Method'])
    auc_df = auc_df.reset_index()

    sns.boxplot(data=pd.melt(auc_df, id_vars=['Mutation', 'Fit Method']),
                x='Mutation', y='value', hue='Fit Method', palette=fmth_clrs,
                order=auc_order.index, hue_order=['optim', 'varit', 'sampl'],
                linewidth=1.7, boxprops=dict(alpha=0.68), flierprops=dict(
                    marker='o', markerfacecolor='black', markersize=4,
                    markeredgecolor='none', alpha=0.4
                    ))
 
    plt.axhline(color='#550000', y=0.5, linewidth=3.3, alpha=0.32)
    plt.xlabel('Mutation', fontsize=21, weight='semibold')
    plt.ylabel('AUC', fontsize=23, weight='semibold')

    plt.xticks(size=13, rotation=37, ha='right')
    plt.yticks(size=16)
    ax.tick_params(axis='y', length=9, width=2)

    fig.savefig(
        os.path.join(plot_dir, "{}__{}".format(args.cohort, args.gene),
                     args.model_name.split('__')[0],
                     "{}__auc-distribution.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_generalization_error(acc_df, args):
    fig, axarr = plt.subplots(figsize=(19, 6), nrows=1, ncols=3, sharex=True)

    auc_dict = {smps: pd.concat([pd.concat([f_dict[smps].AUC], keys=[fmth])
                                 for fmth, f_dict in acc_df.items()])
                for smps in ['train', 'test']}

    plot_min = min(auc_dict['train'].values.min(),
                   auc_dict['test'].values.min()) - 0.01

    for fmth, ax in zip(['optim', 'varit', 'sampl'], axarr):
        train_aucs = auc_dict['train'].loc[fmth]
        test_aucs = auc_dict['test'].loc[fmth]

        if train_aucs.values.min() > 0.999:
            train_aucs += np.random.randn(train_aucs.shape) / 500

        sns.kdeplot(train_aucs.values.flatten(), test_aucs.values.flatten(),
                    shade=True, shade_lowest=False, cut=0, ax=ax)

        ax.set_xlim((plot_min, 1.01))
        ax.set_ylim((plot_min, 1.01))
        ax.tick_params(pad=3.9)
        ax.plot([-1, 2], [-1, 2], linewidth=1.6,
                linestyle='--', color='#550000', alpha=0.4)

        if fmth == 'optim':
            ax.set_ylabel('Testing AUC', fontsize=22, weight='semibold')
        if fmth == 'varit':
            ax.set_xlabel('Training AUC', fontsize=22, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir, "{}__{}".format(args.cohort, args.gene),
                     args.model_name.split('__')[0],
                     "{}__generalization.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_profile(tune_dict, args, cdata):
    tune_dfs = {mtype: tune_vals['mean'] - tune_vals['std']
                for mtype, tune_vals in tune_dict.items()}

    tune_pars = tuple(tune_dfs.values())[0].index.names
    fig, axarr = plt.subplots(figsize=(17, 0.3 + 7 * len(tune_pars)),
                              nrows=len(tune_pars), ncols=1, squeeze=False)

    for tune_par, ax in zip(tune_pars, axarr.flatten()):
        tune_df = pd.concat(
            [tune_vals.groupby(axis=0, level=tune_par).mean().groupby(
                axis=1, level='fmth').quantile(q=0.25, axis=1)
             for tune_vals in tune_dfs.values()],
            keys=tune_dfs.keys(), names=['mtype'], axis=1
            )

        if detect_log_distr(tune_df.index):
            use_distr = [np.log10(par_val) for par_val in tune_df.index]
            par_lbl = "{}\n(log-scale)".format(tune_par)

        else:
            use_distr = tune_df.index.tolist()
            par_lbl = str(tune_par)

        ax.axhline(color='#550000', y=0.5, linewidth=3.1, ls='--', alpha=0.32)
        ax.set_xlabel(par_lbl, fontsize=22, weight='semibold')
        ax.set_ylabel('Training AUC', fontsize=22, weight='semibold')

        for (mtype, fmth), tune_vals in tune_df.iteritems():
            ax.plot(use_distr, tune_vals.tolist(), '-',
                    linewidth=2.7, alpha=0.37, color=fmth_clrs[fmth])

        for par_val in use_distr:
            ax.axvline(x=par_val, color='#116611',
                       ls=':', linewidth=1.7, alpha=0.19)

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir,
                     "{}__{}".format(args.cohort, args.gene),
                     args.model_name.split('__')[0],
                     "{}__tuning-profile.svg".format(
                         args.model_name.split('__')[1])),
        dpi=300, bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance and tuning characteristics of a Stan model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    parser.add_argument('gene', type=str, help="a mutated gene")
    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    out_tag = "{}__{}".format(args.cohort, args.gene)
    os.makedirs(os.path.join(plot_dir, out_tag,
                             args.model_name.split('__')[0]),
                exist_ok=True)

    cdata = merge_cohort_data(os.path.join(base_dir, out_tag))
    with open(os.path.join(base_dir, out_tag,
                           "out-data__{}.p".format(args.model_name)),
              'rb') as fl:
        out_dict = pickle.load(fl)

    plot_auc_distribution(out_dict['Fit']['Acc'], args)
    plot_generalization_error(out_dict['Fit']['Acc'], args)
    plot_tuning_profile(out_dict['Tune']['Acc'], args, cdata)


if __name__ == "__main__":
    main()

