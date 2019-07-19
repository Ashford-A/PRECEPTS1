
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'variant_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'transfer')

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.variant_baseline.plot_model import cv_clrs

import argparse
from pathlib import Path
import dill as pickle
import bz2

import numpy as np
import pandas as pd
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_transfer_aucs(auc_df, auc_vals, stat_dict, args):
    fig_width = 2.9 + auc_df.shape[1] * 0.41
    fig_height = 1.7 + auc_df.shape[0] * 0.23
    fig_size = 18 - min((fig_width * fig_height) ** 0.5, 17.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    annot_df = pd.DataFrame(0.0, index=auc_df.index, columns=auc_df.columns)
    for mtype in auc_df.index:
        best_coh = auc_df.loc[mtype][:-1].idxmax()
        annot_df.loc[mtype, best_coh] = auc_df.loc[mtype, best_coh]
        annot_df.loc[mtype, 'All'] = auc_df.loc[mtype, 'All']

    annot_df = annot_df.applymap('{:.2f}'.format).applymap(
        lambda x: ('' if x == '0.00' else '1.0' if x == '1.00'
                   else x.lstrip('0'))
        )

    ax = sns.heatmap(auc_df, cmap=auc_cmap, vmin=0, vmax=1, center=0.5,
                     linewidth=0, annot=annot_df, fmt='',
                     annot_kws={'size': fig_size * 1.13})

    ax.figure.axes[-1].tick_params(labelsize=fig_size * 1.71)
    ax.figure.axes[-1].set_ylabel("AUC (across 50-fold CVs)",
                                  size=fig_size * 1.9, weight='semibold')

    ax.figure.axes[0].tick_params(
        axis='x', length=fig_size * 0.53, width=fig_size * 0.17)
    plt.xticks(size=fig_size * 1.37, rotation=34, ha='right')
    plt.yticks(size=fig_size * 1.51)

    ax.axvline(auc_df.shape[1] - 1, ymin=-0.5, ymax=1.5,
               c='0.37', linewidth=2.3)

    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__auc-heatmap.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_label_stability(corr_df, auc_df, auc_vals, stat_dict, args):
    fig, ax = plt.subplots(figsize=(14, 11))

    plot_clrs = dict(zip(['all', 'random', 'fivefold'],
                         ['black', cv_clrs['random'], cv_clrs['fivefold']]))

    for mtype, auc_vals in auc_df.iterrows():
        for coh, auc_val in auc_vals.iteritems():
            if coh != 'All' and auc_val == auc_val and auc_val > 0.5:
                corr_dict = corr_df.loc[mtype, coh]

                corr_vals = {
                    cv_mth: np.median(corr_mat.values[np.triu_indices(
                        corr_mat.shape[0], k=1)])
                    for cv_mth, corr_mat in corr_dict.items()
                    }

                for cv_mth, corr_val in corr_vals.items():
                    ax.scatter(auc_val, corr_val, s=53, c=plot_clrs[cv_mth],
                               alpha=0.23, edgecolors='none')

                sort_vals = sorted(corr_vals.items(), key=itemgetter(1))
                ax.plot([auc_val] * 2, [sort_vals[0][1], sort_vals[1][1]],
                        color=plot_clrs[sort_vals[1][0]],
                        alpha=0.19, linewidth=2.1)
                ax.plot([auc_val] * 2, [sort_vals[1][1], sort_vals[2][1]],
                        color=plot_clrs[sort_vals[2][0]],
                        alpha=0.19, linewidth=2.1)

    ax.set_xlabel("Transfer AUC", size=23, weight='semibold')
    ax.set_ylabel("Label Stability", size=23, weight='semibold')
    ax.set_xlim([0.48, 1.01])

    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     args.model_name.split('__')[0],
                     "{}__label-stability.svg".format(
                         args.model_name.split('__')[1])),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the performance of a model in predicting the presence of "
        "mutations in cohorts other than the one it was trained on."
        )

    parser.add_argument('temp_dir', type=str)
    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source was used")

    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort]),
        args.model_name.split('__')[0]
        ), exist_ok=True)
 
    use_ctf = min(
        int(out_file.parts[-2].split('__samps-')[1])
        for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-data__{}.p.gz".format(
                args.expr_source, args.cohort, args.model_name)
            )
        )

    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, use_ctf)
    cdata = merge_cohort_data(os.path.join(base_dir, out_tag))

    with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                  "out-data__{}.p.gz".format(
                                      args.model_name)),
                     'r') as fl:
        out_dict = pickle.load(fl)

    auc_vals = out_dict['Fit']['test'].AUC.quantile(q=0.25, axis=1)
    use_mtypes = auc_vals[auc_vals >= 0.7].index

    stat_dict = dict()
    for coh, trnsf_df in out_dict['Trnsf'].items():
        stat_dict[coh] = dict()

        with open(os.path.join(args.temp_dir, 'variant_baseline',
                               args.expr_source, 'setup',
                               "{}__cohort-data.p".format(coh)), 'rb') as f:
            trnsf_cdata = pickle.load(f)

        if coh in args.cohort:
            sub_stat = trnsf_cdata.train_data()[0].index.isin(
                cdata.get_train_samples())

            if (~sub_stat).any():
                out_dict['Trnsf'][coh][mtype] = out_dict[
                    'Trnsf'][coh][mtype].iloc[~sub_stat, :]

                for mtype in use_mtypes:
                    stat_dict[coh][mtype] = np.array(
                        trnsf_cdata.train_pheno(mtype))[~sub_stat]

            else:
                del(out_dict['Trnsf'][coh])

        else:
            for mtype in use_mtypes:
                stat_dict[coh][mtype] = np.array(
                    trnsf_cdata.train_pheno(mtype))

    corr_df = pd.DataFrame.from_records({
        coh: {
            mtype: {
                'random': trnsf_vals[mtype].iloc[:, :25].corr(
                    method='spearman'),
                'fivefold': trnsf_vals[mtype].iloc[:, 25:50].corr(
                    method='spearman'),
                'all': trnsf_vals[mtype].corr(method='spearman'),
                }
            for mtype, mut_stat in stat_dict[coh].items()
            if mut_stat.sum() >= 20
            }
        for coh, trnsf_vals in out_dict['Trnsf'].items()
        })

    auc_df = pd.DataFrame.from_records({
        coh: {
            mtype: (
                np.greater.outer(
                    trnsf_vals[mtype].iloc[mut_stat, :-1],
                    trnsf_vals[mtype].iloc[~mut_stat, :-1]
                    ).mean()
                + np.equal.outer(
                    trnsf_vals[mtype].iloc[mut_stat, :-1],
                    trnsf_vals[mtype].iloc[~mut_stat, :-1]
                    ).mean()
                / 2
                )
            for mtype, mut_stat in stat_dict[coh].items()
            if mut_stat.sum() >= 20
            }
        for coh, trnsf_vals in out_dict['Trnsf'].items()
        })

    auc_df = auc_df.iloc[:, ~auc_df.isna().all().values]
    auc_df['All'] = -1.

    for mtype in auc_df.index:
        mut_arr = [trnsf_vals[mtype].iloc[stat_dict[coh][mtype], :-1]
                   for coh, trnsf_vals in out_dict['Trnsf'].items()]
        mut_vals = np.concatenate([vals.values.flatten() for vals in mut_arr])

        wt_arr = [trnsf_vals[mtype].iloc[~stat_dict[coh][mtype], :-1]
                  for coh, trnsf_vals in out_dict['Trnsf'].items()]
        wt_vals = np.concatenate([vals.values.flatten() for vals in wt_arr])

        auc_df.loc[mtype, 'All'] = np.greater.outer(mut_vals, wt_vals).mean()
        auc_df.loc[mtype, 'All'] += np.equal.outer(
            mut_vals, wt_vals).mean() / 2

    plot_transfer_aucs(auc_df, auc_vals, stat_dict, args)
    plot_label_stability(corr_df, auc_df, auc_vals, stat_dict, args)


if __name__ == "__main__":
    main()

