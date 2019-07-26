
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'variant_baseline')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'dataset')

from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.utilities.colour_maps import cor_cmap

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from itertools import product
from itertools import combinations as combn
from scipy.stats import spearmanr

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_model_correlation(out_dict, args, cdata_dict):
    fig, axarr = plt.subplots(figsize=(16, 9), nrows=2, ncols=3)

    auc_vals = pd.DataFrame({
        mdl: ols['Fit']['test'].AUC.quantile(q=0.25, axis=1)
        for mdl, ols in out_dict.items()
        }).round(4)
    auc_bins = pd.qcut(auc_vals.values.flatten(),
                       q=[0, 0.5, 0.75, 1], precision=4).categories

    fold_dicts = {auc_bin: {(mdl1, mdl2): []
                            for mdl1, mdl2 in product(out_dict.keys(),
                                                      repeat=2)}
                  for auc_bin in auc_bins}

    infr_dicts = {auc_bin: {(mdl1, mdl2): []
                            for mdl1, mdl2 in product(out_dict.keys(),
                                                      repeat=2)}
                  for auc_bin in auc_bins}

    for mdl, ols in out_dict.items():
        for mtype in ols['Scores']['fivefold']:
            use_bin = auc_bins.get_loc(auc_vals.loc[mtype, mdl])

            val_ar = np.vstack(ols['Scores']['fivefold'][mtype].apply(
                lambda x: [y for y in x if y == y]))
            fold_dicts[auc_bins[use_bin]][mdl, mdl] += [
                np.median([spearmanr(val_ar[:, i], val_ar[:, j]).correlation
                           for i, j in combn(range(5), 2)])
                ]

    for (mdl1, ols1), (mdl2, ols2) in combn(out_dict.items(), 2):
        for mtype in ols1['Scores']['fivefold']:
            mdls_lbl = tuple(sorted([mdl1, mdl2]))

            use_bin = min(auc_bins.get_loc(auc_vals.loc[mtype, mdl1]),
                          auc_bins.get_loc(auc_vals.loc[mtype, mdl2]))

            val_ar1 = np.vstack(ols1['Scores']['fivefold'][mtype].apply(
                lambda x: [y for y in x if y == y]))
            val_ar2 = np.vstack(ols2['Scores']['fivefold'][mtype].apply(
                lambda x: [y for y in x if y == y]))

            fold_dicts[auc_bins[use_bin]][mdls_lbl] += [
                np.median([spearmanr(val_ar1[:, i], val_ar2[:, i]).correlation
                           for i in range(5)])
                ]

            fold_dicts[auc_bins[use_bin]][mdls_lbl[::-1]] += [
                np.median([spearmanr(val_ar1[:, i], val_ar2[:, j]).correlation
                           for i, j in combn(range(5), 2)])
                ]

            infr_dicts[auc_bins[use_bin]][mdls_lbl] += [
                spearmanr(ols1['Scores']['infer'][mtype],
                          ols2['Scores']['infer'][mtype]).correlation
                ]

    fold_dicts = {
        auc_bin: pd.Series({mdls: np.mean(cor_vals)
                            for mdls, cor_vals in bin_dict.items()}).unstack()
        for auc_bin, bin_dict in fold_dicts.items()
        }

    infr_dicts = {
        auc_bin: pd.Series({mdls: np.mean(cor_vals)
                            for mdls, cor_vals in bin_dict.items()}).unstack()
        for auc_bin, bin_dict in infr_dicts.items()
        }

    for j, auc_bin in enumerate(auc_bins):
        fold_annt = fold_dicts[auc_bin].copy().applymap(
            '{:.2f}'.format).applymap(lambda x: x.lstrip('0'))

        sns.heatmap(fold_dicts[auc_bin], cmap=cor_cmap, vmin=-1, vmax=1,
                    center=0, ax=axarr[0, j], square=True, cbar=False,
                    annot=fold_annt, fmt='', annot_kws={'size': 9})

        sns.heatmap(infr_dicts[auc_bin], cmap=cor_cmap, vmin=-1, vmax=1,
                    center=0, ax=axarr[1, j], square=True, cbar=False)

        axarr[0, j].set_title(
            "{:.3f} - {:.3f}".format(auc_bin.left, auc_bin.right))

        axarr[0, j].set_xticklabels([])
        axarr[1, j].set_xticklabels(labels=axarr[1, j].get_xticklabels(),
                                    rotation=37, ha='right', size=11)

        if auc_bin != auc_bins[0]:
            for i in [0, 1]:
                axarr[i, j].set_yticklabels([])

    fig.tight_layout(w_pad=1.3, h_pad=2.1)
    fig.savefig(
        os.path.join(plot_dir, args.expr_source,
                     "{}__model-correlation.svg".format(args.cohort)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser("Plots the relationships between the "
                                     "outputs of mutation prediction models "
                                     "tested in a given cohort's dataset.")

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source was used")
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-data__*.p.gz".format(
                args.expr_source, args.cohort))
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Model': out_data[1].split('out-data__')[1].split('.p')[0]}
        for out_data in out_datas
        ]).groupby(['Model'])['Samps'].min()

    cdata_dict = {
        ctf: merge_cohort_data(os.path.join(
            base_dir, "{}__{}__samps-{}".format(
                args.expr_source, args.cohort, ctf)
            ))
        for ctf in set(out_use)
        }

    out_dict = {
        mdl: pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(
                args.expr_source, args.cohort, ctf),
            "out-data__{}.p.gz".format(mdl)
            ), 'r'))
        for mdl, ctf in out_use.iteritems()
        }

    # create the plots
    plot_model_correlation(out_dict.copy(), args, cdata_dict)


if __name__ == "__main__":
    main()

