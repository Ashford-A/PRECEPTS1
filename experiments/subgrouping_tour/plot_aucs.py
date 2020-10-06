
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.subvariant_test.utils import get_fancy_label
from HetMan.experiments.subvariant_test.plot_aucs import (
    choose_gene_colour, place_labels)
from HetMan.experiments.subvariant_test import pnt_mtype
from dryadic.features.mutations import MuType

import argparse
import bz2
import dill as pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import average_precision_score as aupr_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from colorsys import hls_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_sub_comparisons(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    pnt_dict = dict()
    clr_dict = dict()

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for gene, auc_vec in auc_vals.groupby(
            lambda mtype: mtype.get_labels()[0]):

        # if there were subgroupings tested for the gene, find the results
        # for the mutation representing all point mutations for this gene...
        if len(auc_vec) > 1:
            import pdb; pdb.set_trace()
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            # ...as well as the results for the best subgrouping of
            # mutations found for this gene
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            # if the AUC for the optimal subgrouping is good enough, plot it
            # against the AUC for all point mutations of the gene...
            if auc_vec[best_indx] > 0.6:
                clr_dict[gene] = choose_gene_colour(gene)
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                # ...and if we are sure that the optimal subgrouping AUC is
                # better than the point mutation AUC then add a label with the
                # gene name and a description of the best found subgrouping...
                if conf_sc > 0.9:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene,
                                            get_fancy_label(best_subtype))
                        )

                # ...if we are not sure but the respective AUCs are still
                # pretty great then add a label with just the gene name...
                elif auc_vec[base_indx] > 0.7 or auc_vec[best_indx] > 0.7:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, (gene, ''))

                # ...otherwise plot the point with no label
                else:
                    pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                        base_size ** 0.53, ('', ''))

                pie_ax = inset_axes(
                    ax, width=base_size ** 0.5, height=base_size ** 0.5,
                    bbox_to_anchor=(auc_vec[base_indx], auc_vec[best_indx]),
                    bbox_transform=ax.transData, loc=10,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                           colors=[clr_dict[gene] + (0.77, ),
                                   clr_dict[gene] + (0.29, )])

    # figure out where to place the labels for each point, and plot them
    lbl_pos = place_labels(pnt_dict)
    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=13, ha=pos[1], va='bottom')
        ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][1],
                size=9, ha=pos[1], va='top')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.plot([0.48, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [0.48, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])
    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found subgrouping", size=23, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_aupr_comparisons(auc_vals, pred_df, pheno_dict, conf_vals, args):
    fig, (base_ax, subg_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    pnt_dict = {'Base': dict(), 'Subg': dict()}
    clr_dict = dict()
    plt_max = 0.53

    for gene, auc_vec in auc_vals.groupby(
            lambda mtype: mtype.get_labels()[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            if auc_vec[best_indx] > 0.6:
                clr_dict[gene] = choose_gene_colour(gene)

                base_size = 0.47 * np.mean(pheno_dict[base_mtype])
                best_prop = 0.47 * np.mean(
                    pheno_dict[best_subtype]) / base_size

                base_infr = pred_df.loc[base_mtype].apply(np.mean)
                best_infr = pred_df.loc[best_subtype].apply(np.mean)

                base_auprs = (aupr_score(pheno_dict[base_mtype], base_infr),
                              aupr_score(pheno_dict[base_mtype], best_infr))
                subg_auprs = (aupr_score(pheno_dict[best_subtype], base_infr),
                              aupr_score(pheno_dict[best_subtype], best_infr))

                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                base_lbl = '', ''
                subg_lbl = '', ''
                min_diff = np.log2(1.25)

                if conf_sc > 0.9:
                    base_lbl = gene, get_fancy_label(best_subtype)
                    subg_lbl = gene, get_fancy_label(best_subtype)

                elif auc_vec[base_indx] > 0.75 or auc_vec[best_indx] > 0.75:
                    base_lbl = gene, ''
                    subg_lbl = gene, ''

                elif auc_vec[base_indx] > 0.6 or auc_vec[best_indx] > 0.6:
                    if abs(np.log2(base_auprs[1] / base_auprs[0])) > min_diff:
                        base_lbl = gene, ''
                    if abs(np.log2(subg_auprs[1] / subg_auprs[0])) > min_diff:
                        subg_lbl = gene, ''

                pnt_dict['Base'][base_auprs] = base_size ** 0.53, base_lbl
                pnt_dict['Subg'][subg_auprs] = base_size ** 0.53, subg_lbl

                for ax, (base_aupr, subg_aupr) in zip(
                        [base_ax, subg_ax], [base_auprs, subg_auprs]):
                    plt_max = min(1.005,
                                  max(plt_max,
                                      base_aupr + 0.11, subg_aupr + 0.11))

                    pie_ax = inset_axes(
                        ax, width=base_size ** 0.5, height=base_size ** 0.5,
                        bbox_to_anchor=(base_aupr, subg_aupr),
                        bbox_transform=ax.transData, loc=10,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.29, 0],
                               colors=[clr_dict[gene] + (0.77,),
                                       clr_dict[gene] + (0.29,)])

    for ax, lbl in zip([base_ax, subg_ax], ['Base', 'Subg']):
        lbl_pos = place_labels(pnt_dict[lbl],
                               lims=(0, plt_max), lbl_dens=0.63)

        for (pnt_x, pnt_y), pos in lbl_pos.items():
            ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                    pnt_dict[lbl][pnt_x, pnt_y][1][0],
                    size=11, ha=pos[1], va='bottom')
            ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                    pnt_dict[lbl][pnt_x, pnt_y][1][1],
                    size=7, ha=pos[1], va='top')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]
            ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

            # if the label is sufficiently far away from its point...
            if ln_lngth > (0.043 + pnt_dict[lbl][pnt_x, pnt_y][0] / 19):
                use_clr = clr_dict[pnt_dict[lbl][pnt_x, pnt_y][1][0]]
                pnt_gap = pnt_dict[lbl][pnt_x, pnt_y][0] / (13 * ln_lngth)
                lbl_gap = 0.006 / ln_lngth

                ax.plot([pnt_x - pnt_gap * x_delta,
                         pos[0][0] + lbl_gap * x_delta],
                        [pnt_y - pnt_gap * y_delta,
                         pos[0][1] + lbl_gap * y_delta
                         + 0.008 + 0.004 * np.sign(y_delta)],
                        c=use_clr, linewidth=1.1, alpha=0.23)

        ax.plot([0, plt_max], [0, 0],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([0, 0], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [1, 1],
                color='black', linewidth=1.5, alpha=0.89)
        ax.plot([1, 1], [0, plt_max],
                color='black', linewidth=1.5, alpha=0.89)

        ax.plot([0, plt_max], [0, plt_max],
                color='#550000', linewidth=1.7, linestyle='--', alpha=0.37)

        ax.set_xlim([-0.01, plt_max])
        ax.set_ylim([-0.01, plt_max])

        ax.set_xlabel("using all point mutation predicted scores",
                      size=19, weight='semibold')
        ax.set_ylabel("using best found subgrouping predicted scores",
                      size=19, weight='semibold')

    base_ax.set_title("AUPR on all point mutations",
                      size=21, weight='semibold')
    subg_ax.set_title("AUPR on best subgrouping mutations",
                      size=21, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "aupr-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the AUCs for a particular classifier on the mutations "
        "enumerated for a given cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('search_params', type=str)
    parser.add_argument('mut_lvls', type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    args = parser.parse_args()
    out_dir = os.path.join(base_dir,
                           '__'.join([args.expr_source, args.cohort]))

    out_files = {
        out_lbl: os.path.join(
            out_dir, "out-{}__{}__{}__{}.p.gz".format(
                out_lbl, args.search_params, args.mut_lvls, args.classif)
            )
        for out_lbl in ['pred', 'pheno', 'aucs', 'conf']
        }

    if not os.path.isfile(out_files['conf']):
        raise ValueError("No experiment output found for these parameters!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    with bz2.BZ2File(out_files['pred'], 'r') as f:
        pred_df = pickle.load(f)['Chrm']
    with bz2.BZ2File(out_files['pheno'], 'r') as f:
        phn_dict = pickle.load(f)
    with bz2.BZ2File(out_files['aucs'], 'r') as f:
        auc_vals = pickle.load(f).Chrm
    with bz2.BZ2File(out_files['conf'], 'r') as f:
        conf_vals = pickle.load(f)['Chrm']

    assert auc_vals.index.isin(phn_dict).all()
    plot_sub_comparisons(auc_vals, phn_dict, conf_vals, args)
    plot_aupr_comparisons(auc_vals, pred_df, phn_dict, conf_vals, args)


if __name__ == '__main__':
    main()

