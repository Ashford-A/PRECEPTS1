
from dryadic.features.mutations import MuType
from ..utilities.labels import get_fancy_label
from ..AML_scRNA_analysis import base_dir
from .utils import load_scRNA_expr
from ..utilities.transformers import OmicUMAP

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from itertools import combinations as combn
from functools import reduce
from operator import and_, itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Added by Andrew on 6/2/2023 to point the script to the correct run location. Instead of "base_dir", it was saved in "temp_dir" global variable
# base_dir points the script toward outputs in the temporary directory location where the script stored the intermediate files, it requires the creation of
# a plots/cluster directory within, for instance "dryads-research/AML_scRNA_analysis/default__default/
# For example: dryads-research/AML_scRNA_analysis/default__default/plots/cluster
#base_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/AML556-D0/Temp_Files/dryads-research/AML_scRNA_analysis/default__default'
#base_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/AML556-D0/dryads-research/AML_scRNA_analysis'
base_dir = '/home/groups/precepts/ashforda/single-cell_precepts/custom_precepts_inputs/vanGalen_D0_AML_samples_and_4_healthy_BM_samples/Temp_Files/dryads-research/AML_scRNA_analysis/default__default'

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'cluster')


def plot_score_clustering(auc_vec, pred_data, comp_info, pheno_dict, args):
    fig, (gene_ax, subt_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    comp_tag, comp_data = comp_info
    cur_gene = {tuple(mtype.label_iter())[0] for mtype in auc_vec.index}
    assert len(cur_gene) == 1, ("Score comparison plots can only be made for "
                                "one mutated gene at a time!")
    cur_gene = tuple(cur_gene)[0]

    base_mtype = MuType({('Gene', cur_gene): None})
    pred_vals = {mtype: pred_data.loc[mtype].apply(np.mean)
                 for mtype in auc_vec.index}
    corr_vals = {mtype: spearmanr(pred_vals[base_mtype], vals)[0]
                 for mtype, vals in pred_vals.items() if mtype != base_mtype}

    divg_scrs = pd.Series({mtype: (1 - corr_val) * (auc_vec[mtype] - 0.7)
                           for mtype, corr_val in corr_vals.items()})
    best_subtype = divg_scrs.sort_values().index[-1]
    #use_cmap = sns.diverging_palette(13, 131, s=91, l=31, sep=10, as_cmap=True)
    use_cmap = sns.diverging_palette(13, 131, s=91, l=31, sep=50, as_cmap=True)
    #use_cmap = use_cmap = sns.diverging_palette(13, 131, s=91, l=31, sep=10, center="dark", as_cmap=True)
    
    for ax, mtype in [(gene_ax, base_mtype), (subt_ax, best_subtype)]:
        ax.scatter(comp_data.loc[pred_data.columns].iloc[:, 0],
                   comp_data.loc[pred_data.columns].iloc[:, 1],
                   marker='o', s=4, c=pred_vals[mtype], cmap=use_cmap,
                   alpha=0.13, edgecolor='none')

        ax.text(0.99, 0.01,
                "{} muts in beatAML\nAUC: {:.3f}".format(
                    np.sum(pheno_dict[mtype]), auc_vec[mtype]),
                size=15, ha='right', va='bottom', transform=ax.transAxes)

        ax.grid(alpha=0.47, linewidth=1.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    gene_ax.set_title("Any Mutation of {}".format(cur_gene),
                      size=21, weight='bold')

    subt_lbl = get_fancy_label(
        MuType({('Scale', 'Point'): tuple(mtype.subtype_iter())[0][1]}),
        pnt_link='\n'
        )
    subt_ax.set_title(subt_lbl, size=21, weight='bold')

    fig.savefig(
        os.path.join(plot_dir, "{}_score-clustering.svg".format(cur_gene)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_comparison(trans_expr, comp_info, args):
    fig, (comp_ax, main_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    comp_tag, comp_data = comp_info
    print(comp_data.shape)
    for ax, umap_df in [(comp_ax, comp_data), (main_ax, trans_expr)]:
        ax.scatter(umap_df.iloc[:, 0], umap_df.iloc[:, 1], marker='o',
                   s=5, c=['black'], alpha=0.11, edgecolor='none')

        ax.grid(alpha=0.47, linewidth=1.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.savefig(
        os.path.join(plot_dir, "comparison_{}.svg".format(comp_tag)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_clustering(trans_expr, args):
    fig, ax = plt.subplots(figsize=(9, 8))

    type_stat = trans_expr.index.map(itemgetter(0))
    type_clrs = dict(zip(type_stat.unique(),
                         sns.color_palette('bright',
                                           n_colors=len(type_stat.unique()))))

    lgnd_lbls = []
    lgnd_marks = []

    for sub_type, type_clr in type_clrs.items():
        type_indx = type_stat == sub_type

        ax.scatter(trans_expr.iloc[type_indx, 0],
                   trans_expr.iloc[type_indx, 1],
                   marker='o', s=13, c=[type_clr],
                   alpha=0.13, edgecolor='none')

        lgnd_lbls += ["{} ({})".format(sub_type, np.sum(type_indx))]
        lgnd_marks += [Line2D([], [], marker='o', linestyle='None',
                              markersize=23, alpha=0.43,
                              markerfacecolor=type_clr,
                              markeredgecolor='none')]

    ax.set_xlabel("1st Component", size=17, weight='semibold')
    ax.set_xticklabels([])
    ax.set_ylabel("2nd Component", size=17, weight='semibold')
    ax.set_yticklabels([])

    ax.legend(lgnd_marks, lgnd_lbls, bbox_to_anchor=(0.5, -0.05),
              frameon=False, fontsize=21, ncol=3, loc=9, handletextpad=0.3)
    ax.grid(alpha=0.47, linewidth=1.3)

    fig.savefig(
        os.path.join(plot_dir, "clustering.svg"),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_cluster',
        description="Plots the output of unsupervised learning on a cohort."
        )

    parser.add_argument('classif')
    parser.add_argument('--feats_file')
    parser.add_argument('--comp_files', nargs='+')
    parser.add_argument('--seed', type=int, default=9087)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Added by Andrew 6/2/2023 to check what base_dir is set to
    #print('base_dir: ', base_dir)

    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "out-pheno.p.gz"),
                     'r') as f:
        pheno_dict = pickle.load(f)

        # Added by Andrew on 6/2/2023 to check the pheno_dict values
        #print('pheno_dict: ', pheno_dict)

    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "out-aucs.p.gz"),
                     'r') as f:
        auc_df = pickle.load(f)

        # Added by Andrew on 6/2/2023 to check the auc_df values
        #print('auc_df: ', auc_df)

    comp_dict = {
        Path(comp_file).stem: pd.read_csv(comp_file, sep='\t',
                                          index_col=0, header=None)
        for comp_file in args.comp_files
        }

    with bz2.BZ2File(os.path.join(base_dir, args.classif,
                                  "out-sc.p.gz"),
                     'r') as f:
        sc_preds = pickle.load(f)
        
        # Added by Andrew on 6/2/2023 to check the sc_preds values
        #print('sc_preds: ', sc_preds)

    for gene, auc_vec in auc_df['mean'].groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        if len(auc_vec) > 1:
            for comp_info in comp_dict.items():
                plot_score_clustering(auc_vec, sc_preds, comp_info,
                                      pheno_dict, args)

    sc_expr = load_scRNA_expr().sparse.to_dense()
    if args.feats_file:
        with open(args.feats_file, 'rb') as fl:
            use_feats = pickle.load(fl)
        sc_expr = sc_expr[use_feats] + 2.

    sc_means = sc_expr.mean()
    sc_stds = sc_expr.std()
    sc_expr = sc_expr.loc[:, sc_means >= sc_means.quantile(q=0.5)]

    trans_expr = OmicUMAP().fit_transform(sc_expr, low_memory=True)
    trans_expr = pd.DataFrame(trans_expr, index=sc_expr.index)
    os.makedirs(plot_dir, exist_ok=True)

    for comp_info in comp_dict.items():
        plot_comparison(trans_expr.copy(), comp_info, args)

    plot_clustering(trans_expr.copy(), args)


if __name__ == "__main__":
    main()

