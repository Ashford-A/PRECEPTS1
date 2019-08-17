
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

from HetMan.experiments.subvariant_tour import *
from HetMan.experiments.subvariant_tour.utils import calculate_aucs
from HetMan.experiments.subvariant_tour.merge_tour import merge_cohort_data
from dryadic.features.mutations import MuType

import argparse
from glob import glob
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def get_fancy_label(mtype):
    sub_keys = mtype.subkeys()

    if len(sub_keys) <= 2:
        use_lbls = []

        for lbl_key in sub_keys:
            sub_lbls = str(MuType(lbl_key)).split(':')[1:]

            for i in range(len(sub_lbls)):
                if sub_lbls[i].count('|') >= 2:
                    sub_lbls[i] = "3+ sub-mutations"

                sub_lbls[i] = sub_lbls[i].replace("None", "no domain")
                sub_lbls[i] = sub_lbls[i].replace("_Mutation", "")
                sub_lbls[i] = sub_lbls[i].replace("_", "")
                sub_lbls[i] = sub_lbls[i].replace("|", " or ")

                dom_mtch = re.search("(^[:]|^)SM[0-9]", sub_lbls[i])
                while dom_mtch:
                    sub_lbls[i] = "{}Domain:SMART-{}".format(
                        sub_lbls[i][:dom_mtch.span()[0]],
                        sub_lbls[i][(dom_mtch.span()[1] - 1):]
                        )
                    dom_mtch = re.search("(^[:]|^)SM[0-9]", sub_lbls[i])

                exn_mtch = re.search("([0-9]+)/([0-9]+)", sub_lbls[i])
                while exn_mtch:
                    sub_lbls[i] = "Exon:{}{}".format(
                        exn_mtch.groups()[0],
                        sub_lbls[i][exn_mtch.span()[1]:]
                        )
                    exn_mtch = re.search("([0-9]+)/([0-9]+)", sub_lbls[i])

            use_lbls += ['\nwith '.join(sub_lbls)]
        use_lbl = '\nand '.join(use_lbls)

    else:
        use_lbl = "grouping of\n3+ mutation types"

    return use_lbl


def place_labels(pnt_dict):
    lbl_pos = {pnt: None for pnt in pnt_dict}

    for pnt, (sz, _) in pnt_dict.items():
        use_sz = (sz ** 0.53) / 1843

        if not any(((pnt[0] - 0.13 - use_sz) < pnt2[0] <= pnt[0]
                    and ((pnt[1] - 0.03 - use_sz) < pnt2[1]
                         < (pnt[1] + 0.03 + use_sz)))
                   for pnt2 in pnt_dict if pnt2 != pnt):
            lbl_pos[pnt] = ((pnt[0] - use_sz, pnt[1]), 'right')

        elif not any((pnt[0] <= pnt2[0] < (pnt[0] + 0.13 + use_sz))
                      and ((pnt[1] - 0.03 - use_sz) < pnt2[1]
                           < (pnt[1] + 0.03 + use_sz))
                     for pnt2 in pnt_dict if pnt2 != pnt):
            lbl_pos[pnt] = ((pnt[0] + use_sz, pnt[1]), 'left')

    while any(pos is None for pos in lbl_pos.values()):
        old_pnts = tuple(pnt_dict.keys())

        for pnt in old_pnts:
            if (pnt in lbl_pos and lbl_pos[pnt] is None
                    and pnt_dict[pnt][1] is not None):
                new_pos = 0.05 * np.random.randn(2) + [pnt[0], pnt[1]]
                new_pos = new_pos.round(5).clip(0.55, 0.95).tolist()

                if not any((((new_pos[0] - 0.1)
                             < pnt2[0] <= (new_pos[0] + 0.1))
                            and ((new_pos[1] - 0.04) < pnt2[1]
                                 < (new_pos[1] + 0.04)))
                           for pnt2 in pnt_dict if pnt2 != pnt):
                    lbl_pos[pnt] = new_pos, 'center'
                    pnt_dict[tuple(new_pos)] = (0, None)

    return lbl_pos


def plot_sub_comparisons(auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(11, 10))
    np.random.seed(3742)

    pnt_dict = dict()
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

            if auc_vec[best_indx] > 0.6:
                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size

                pnt_dict[auc_vec[base_indx], auc_vec[best_indx]] = (
                    2119 * base_size, (gene, get_fancy_label(best_subtype)))

                pie_size = base_size ** 0.5
                pie_ax = inset_axes(ax, width=pie_size, height=pie_size,
                                    bbox_to_anchor=(auc_vec[base_indx],
                                                    auc_vec[best_indx]),
                                    bbox_transform=ax.transData, loc=10,
                                    axes_kwargs=dict(aspect='equal'),
                                    borderpad=0)

                pie_ax.pie(x=[best_prop, 1 - best_prop], explode=[0.29, 0],
                           colors=[clr_dict[gene] + (0.77,),
                                   clr_dict[gene] + (0.29,)])

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
        pnt_sz = (pnt_dict[pnt_x, pnt_y][0] ** 0.43) / 1077
        if ln_lngth > 0.01 + pnt_sz:
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1][0]]
            lbl_sz = pnt_dict[pnt_x, pnt_y][1][1].count('\n')

            pnt_gap = pnt_sz / ln_lngth
            lbl_gap = (0.02 + (1 / 117) * lbl_sz ** 0.17) / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta],
                    c=use_clr, linewidth=2.7, alpha=0.27)

    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])
    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found subgrouping", size=23, weight='semibold')

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            linewidth=2.1, linestyle='--', color='#550000', alpha=0.41)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
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
    parser.add_argument('classif', help="a mutation classifier", type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-data__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif))
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-data__')[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby(['Levels'])['Samps'].min()

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    cdata_dict = {
        lvls: merge_cohort_data(os.path.join(
            base_dir, "{}__{}__samps-{}".format(
                args.expr_source, args.cohort, ctf)
            ), lvls, use_seed=8713)
        for lvls, ctf in out_use.iteritems()
        }

    infer_dict = {
        lvls: pickle.load(bz2.BZ2File(os.path.join(
            base_dir, "{}__{}__samps-{}".format(
                args.expr_source, args.cohort, ctf),
            "out-data__{}__{}.p.gz".format(lvls, args.classif)
            ), 'r'))['Infer']
        for lvls, ctf in out_use.iteritems()
        }

    out_dict = {lvls: calculate_aucs(infer_dfs, cdata_dict[lvls])
                for lvls, infer_dfs in infer_dict.items()}

    pheno_dict = {mtype: phn for _, phn_dict in out_dict.values()
                  for mtype, phn in phn_dict.items()}
    auc_dfs = {cis_lbl: pd.concat([auc_df[cis_lbl]
                                   for auc_df, _ in out_dict.values()])
               for cis_lbl in cis_lbls}

    plot_sub_comparisons(auc_dfs['Chrm'], pheno_dict, args)


if __name__ == '__main__':
    main()

