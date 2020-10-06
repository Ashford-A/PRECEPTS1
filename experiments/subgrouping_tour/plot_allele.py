
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'allele')

from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from HetMan.experiments.subvariant_tour.utils import (
    get_fancy_label, RandomType)
from HetMan.experiments.subvariant_tour.plot_aucs import place_labels
from dryadic.features.mutations import MuType

import argparse
from glob import glob
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def plot_sub_comparisons(allc_df, infer_df, auc_vals, pheno_dict, args):
    fig, (subt_ax, allt_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)
    np.random.seed(3742)

    auc_vals = auc_vals[[
        not isinstance(mtype, RandomType)
        and not (mtype.subtype_list()[0][1] != pnt_mtype
                 and pheno_dict[mtype].sum() == pheno_dict[MuType(
                     {('Gene', mtype.get_labels()[0]): pnt_mtype})].sum())
        for mtype in auc_vals.index
        ]]

    subt_pnts = dict()
    allt_pnts = dict()
    clr_dict = dict()

    for gene, auc_vec in auc_vals.groupby(
            lambda mtype: mtype.subtype_list()[0][0]):
        clr_dict[gene] = hls_to_rgb(
            h=np.random.uniform(size=1)[0], l=0.5, s=0.8)

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            if auc_vec[best_indx] > 0.65:
                base_infr = infer_df.loc[base_mtype].apply(np.mean)
                base_allc = allc_df[base_mtype][pheno_dict[base_mtype]]

                subt_infr = infer_df.loc[best_subtype].apply(np.mean)
                subt_allc = allc_df[best_subtype][pheno_dict[best_subtype]]
                allt_allc = allc_df[best_subtype][pheno_dict[base_mtype]]

                base_corr = spearmanr(base_infr[pheno_dict[base_mtype]],
                                      base_allc).correlation
                subt_corr = spearmanr(subt_infr[pheno_dict[best_subtype]],
                                      subt_allc).correlation
                allt_corr = spearmanr(subt_infr[pheno_dict[base_mtype]],
                                      allt_allc).correlation

                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                if auc_vec[best_indx] > 0.75:
                    use_lbl = gene, get_fancy_label(best_subtype)
                else:
                    use_lbl = '', ''

                subt_pnts[base_corr, subt_corr] = base_size ** 0.53, use_lbl
                allt_pnts[base_corr, allt_corr] = base_size ** 0.53, use_lbl

                for ax, corr in zip([subt_ax, allt_ax],
                                    [subt_corr, allt_corr]):
                    pie_ax = inset_axes(
                        ax, width=base_size ** 0.5, height=base_size ** 0.5,
                        bbox_to_anchor=(base_corr, corr),
                        bbox_transform=ax.transData, loc=10,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.21, 0],
                               colors=[clr_dict[gene] + (0.77,),
                                       clr_dict[gene] + (0.29,)])

    plt_lims = (min(min(xval for xval, _ in subt_pnts),
                    min(yval for _, yval in subt_pnts),
                    min(yval for _, yval in allt_pnts)) - 0.11,
                max(max(xval for xval, _ in subt_pnts),
                    max(yval for _, yval in subt_pnts),
                    max(yval for _, yval in allt_pnts)) + 0.11)

    for ax, pnt_dict in zip([subt_ax, allt_ax], [subt_pnts, allt_pnts]):
        lbl_pos = place_labels(pnt_dict, lims=plt_lims, lbl_dens=0.61)

        for (pnt_x, pnt_y), pos in lbl_pos.items():
            ax.text(pos[0][0], pos[0][1] + 500 ** -1,
                    pnt_dict[pnt_x, pnt_y][1][0],
                    size=11, ha=pos[1], va='bottom')
            ax.text(pos[0][0], pos[0][1] - 500 ** -1,
                    pnt_dict[pnt_x, pnt_y][1][1],
                    size=7, ha=pos[1], va='top')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]
            ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

            # if the label is sufficiently far away from its point...
            if ln_lngth > (0.031 + pnt_dict[pnt_x, pnt_y][0] / 19):
                use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
                pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (11 * ln_lngth)
                lbl_gap = 0.008 / ln_lngth

                ax.plot([pnt_x - pnt_gap * x_delta,
                         pos[0][0] + lbl_gap * x_delta],
                        [pnt_y - pnt_gap * y_delta,
                         pos[0][1] + lbl_gap * y_delta
                         + 0.01 + 0.006 * np.sign(y_delta)],
                        c=use_clr, linewidth=1.9, alpha=0.27)

        ax.set_xlim(plt_lims)
        ax.set_ylim(plt_lims)

        ax.plot(plt_lims, [0, 0],
                color='black', linewidth=1.1, linestyle=':', alpha=0.71)
        ax.plot([0, 0], plt_lims,
                color='black', linewidth=1.1, linestyle=':', alpha=0.71)
 
        ax.plot(plt_lims, plt_lims,
                color='#550000', linewidth=1.5, linestyle='--', alpha=0.43)

    subt_ax.set_xlabel("correlation between inferred\nmutation score and VAF",
                       size=18, weight='semibold')
    subt_ax.set_ylabel("correlation between inferred best subgrouping\n"
                       "score and VAF of best subgrouping mutation",
                       size=17, weight='semibold')

    allt_ax.set_xlabel("correlation between inferred\nmutation score and VAF",
                       size=18, weight='semibold')
    allt_ax.set_ylabel("correlation between inferred best subgrouping\n"
                       "score and VAF of all point mutations",
                       size=17, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots how the allelic frequency of mutations in a cohort compare "
        "against the characteristics of the classifiers used to predict them."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-data__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif)
            )
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-data__')[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby(['Levels'])['Samps'].min()

    out_tag = "{}__{}__samps-{}".format(
        args.expr_source, args.cohort, out_use.min())
    cdata = merge_cohort_data(os.path.join(base_dir, out_tag), use_seed=8713)

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    phn_dict = dict()
    infer_dict = dict()
    auc_dict = dict()

    for lvls, ctf in out_use.iteritems():
        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, args.cohort, ctf),
                "out-pheno__{}__{}.p.gz".format(lvls, args.classif)
                ), 'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, args.cohort, ctf),
                "out-data__{}__{}.p.gz".format(lvls, args.classif)
                ), 'r') as f:
            infer_dict[lvls] = pickle.load(f)['Infer']['Chrm']

        with bz2.BZ2File(os.path.join(
                base_dir, "{}__{}__samps-{}".format(
                    args.expr_source, args.cohort, ctf),
                "out-aucs__{}__{}.p.gz".format(lvls, args.classif)
                ), 'r') as f:
            auc_dict[lvls] = pickle.load(f)['Chrm']

    infer_df = pd.concat(infer_dict.values())
    auc_vals = pd.concat(auc_dict.values())

    allc_dict = {
        mtype: mtype.get_leaf_annot(cdata.mtrees[cdata.choose_mtree(mtype)],
                                    ['ref_count', 'alt_count'])
        for mtype in auc_vals.index if not isinstance(mtype, RandomType)
        }

    allc_df = pd.DataFrame(
        {mtype: {samp: (sum(vals['alt_count']) / (sum(vals['alt_count'])
                                                  + sum(vals['ref_count'])))
                 for samp, vals in allcs.items()}
         for mtype, allcs in allc_dict.items()},
        index=sorted(cdata.get_train_samples())
        ).fillna(0.0)

    plot_sub_comparisons(allc_df, infer_df, auc_vals, phn_dict, args)


if __name__ == '__main__':
    main()

