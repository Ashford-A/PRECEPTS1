
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'variant_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'transfer')

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.utilities import auc_cmap

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


def plot_transfer_aucs(trnsf_dict, auc_vals, stat_dict, args, cdata):
    auc_df = pd.DataFrame.from_records({coh: {
        mtype: (
            np.greater.outer(trnsf_vals[mtype].iloc[mut_stat, :-1],
                             trnsf_vals[mtype].iloc[~mut_stat, :-1]).mean()
            + np.equal.outer(trnsf_vals[mtype].iloc[mut_stat, :-1],
                             trnsf_vals[mtype].iloc[~mut_stat, :-1]).mean()
            / 2
            )
        for mtype, mut_stat in stat_dict[coh].items() if mut_stat.sum() >= 20
        } for coh, trnsf_vals in trnsf_dict.items()})

    auc_df = auc_df.iloc[:, ~auc_df.isna().all().values]
    auc_df['All'] = -1.

    for mtype in auc_df.index:
        mut_arr = [trnsf_vals[mtype].iloc[stat_dict[coh][mtype], :-1]
                   for coh, trnsf_vals in trnsf_dict.items()]
        mut_vals = np.concatenate([vals.values.flatten() for vals in mut_arr])

        wt_arr = [trnsf_vals[mtype].iloc[~stat_dict[coh][mtype], :-1]
                  for coh, trnsf_vals in trnsf_dict.items()]
        wt_vals = np.concatenate([vals.values.flatten() for vals in wt_arr])

        auc_df.loc[mtype, 'All'] = np.greater.outer(mut_vals, wt_vals).mean()
        auc_df.loc[mtype, 'All'] += np.equal.outer(
            mut_vals, wt_vals).mean() / 2

    fig_width = 2.9 + auc_df.shape[1] * 0.41
    fig_height = 1.7 + auc_df.shape[0] * 0.23
    fig_size = 23 - min((fig_width * fig_height) ** 0.61, 28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax = sns.heatmap(auc_df, cmap=auc_cmap, vmin=0, vmax=1, center=0.5,
                     linewidth=0)

    ax.figure.axes[-1].tick_params(labelsize=fig_size * 1.73)
    ax.figure.axes[-1].set_ylabel("AUC (across 50-fold CVs)",
                                  size=fig_size * 1.9, weight='semibold')

    ax.figure.axes[0].tick_params(
        axis='x', length=fig_size * 0.53, width=fig_size * 0.17)
    plt.xticks(size=fig_size * 1.37, rotation=34, ha='right')
    plt.yticks(size=fig_size * 1.51)

    fig.savefig(
        os.path.join(plot_dir,
                     "{}__{}__samps-{}".format(args.expr_source, args.cohort,
                                               args.samp_cutoff),
                     args.model_name.split('__')[0],
                     "{}__auc-heatmap.svg".format(
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

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('model_name', type=str,
                        help="which mutation classifier was tested")

    args = parser.parse_args()
    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, args.samp_cutoff)

    os.makedirs(os.path.join(plot_dir, out_tag,
                             args.model_name.split('__')[0]),
                exist_ok=True)

    cdata = merge_cohort_data(os.path.join(base_dir, out_tag))
    with open(os.path.join(base_dir, out_tag,
                           "out-data__{}.p".format(args.model_name)),
              'rb') as fl:
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

    plot_transfer_aucs(out_dict['Trnsf'], auc_vals, stat_dict, args, cdata)


if __name__ == "__main__":
    main()

