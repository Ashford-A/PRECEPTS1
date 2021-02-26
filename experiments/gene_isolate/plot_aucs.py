
from ..utilities.mutations import (
    pnt_mtype, dup_mtype, loss_mtype, gains_mtype, dels_mtype)

from ..gene_isolate import base_dir
from ..subgrouping_isolate.utils import remove_pheno_dups, get_mut_ex
from ..subgrouping_isolate.plot_gene import choose_subtype_colour
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.colour_maps import variant_clrs, mcomb_clrs
from ..utilities.labels import get_fancy_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from functools import reduce
from operator import or_

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}
cna_mtypes = {'Gain': gains_mtype, 'Loss': dels_mtype}


def plot_size_comparisons(auc_df, pheno_dict, args):
    fig, axarr = plt.subplots(figsize=(9, 10), nrows=2)

    plot_dicts = {ex_lbl: dict() for ex_lbl in ['Iso', 'IsoShal']}
    line_dicts = {ex_lbl: dict() for ex_lbl in ['Iso', 'IsoShal']}

    clr_dict = {variant_clrs['Point']: 'only point mutations',
                mcomb_clrs['Point+Gain']: 'point or gains',
                mcomb_clrs['Point+Loss']: 'point or losses'}
    lgnd_dicts = {ex_lbl: {clr: 0 for clr in clr_dict}
                  for ex_lbl in ['Iso', 'IsoShal']}

    size_lims = list()
    auc_lims = list()

    for ax, ex_lbl in zip(axarr, ['Iso', 'IsoShal']):
        auc_vals = auc_df.loc[[mut for mut in auc_df.index
                               if get_mut_ex(mut) == ex_lbl], ex_lbl]

        plt_df = pd.DataFrame({
            mut: {'Size': np.sum(pheno_dict[mut]), 'AUC': auc_val}
            for mut, auc_val in auc_vals.iteritems()
            }).transpose().astype({'Size': int})
        plt_df = plt_df.loc[remove_pheno_dups(plt_df.index, pheno_dict)]

        for mut, (size_val, auc_val) in plt_df.iterrows():
            plt_mut = reduce(or_, mut.mtypes)
            plt_clr = choose_subtype_colour(plt_mut)

            if (plt_mut.is_supertype(pnt_mtype)
                    or plt_mut in {dels_mtype, gains_mtype,
                                   dup_mtype, loss_mtype}):
                plt_lbl = get_fancy_label(plt_mut)

                plt_sz = 347
                lbl_gap = 0.31
                edg_clr = plt_clr

            else:
                lgnd_dicts[ex_lbl][plt_clr] += 1

                plt_lbl = ''
                plt_sz = 31
                lbl_gap = 0.13
                edg_clr = 'none'

            if plt_lbl is not None:
                line_dicts[ex_lbl][size_val, auc_val] = dict(c=plt_clr)
                plot_dicts[ex_lbl][size_val, auc_val] = lbl_gap, (plt_lbl, '')

            ax.scatter(size_val, auc_val,
                       c=[plt_clr], s=plt_sz, alpha=0.23, edgecolor=edg_clr)

        size_lims += [plt_df.Size.quantile(q=[0, 1]).tolist()]
        auc_lims += [plt_df.AUC.quantile(q=[0, 1]).tolist()]

    x_min = min(min_size for min_size, _ in size_lims)
    x_max = max(max_size for _, max_size in size_lims)
    x_rng = x_max - x_min
    y_min = min(min_auc for min_auc, _ in auc_lims)
    y_max = max(max_auc for _, max_auc in auc_lims)
    y_rng = y_max - y_min

    x_min, x_max = max(x_min - x_rng / 41, 0), x_max + x_rng / 8.3
    y_min, y_max = y_min - (1 - y_min) / 31, 1 + (1 - y_min) / 113
    if 0.4 < y_min < 0.48:
        y_min = 0.389

    for ax, ex_lbl in zip(axarr, ['Iso', 'IsoShal']):
        ax.set_title(ex_lbl, size=24, weight='semibold')
        ax.grid(linewidth=0.83, alpha=0.41)
        ax.tick_params(labelsize=13)

        ax.plot([x_min, x_max], [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
        ax.plot([x_min, x_max], [1, 1],
                color='black', linewidth=1.9, alpha=0.89)
        ax.plot([0, 0], [y_min, y_max],
                color='black', linewidth=1.9, alpha=0.89)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if plot_dicts[ex_lbl]:
            lbl_pos = place_scatter_labels(plot_dicts[ex_lbl], ax,
                                           font_size=11,
                                           line_dict=line_dicts[ex_lbl],
                                           linewidth=0.71, alpha=0.61)

        if ex_lbl == 'IsoShal':
            ax.set_xlabel("# of Mutated Samples",
                          size=21, weight='semibold')
        else:
            ax.set_xticklabels([])

    plt.tight_layout(h_pad=1.1)
    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}_size-comparison_{}.svg".format(
                         args.cohort, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of classification task AUCs."
        )

    parser.add_argument('gene', help="a mutated gene")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = Path(base_dir, args.gene)

    out_files = tuple(out_dir.glob(
        os.path.join("out-conf__{}__*_*_{}.p.gz".format(
            args.cohort, args.classif))
        ))

    if len(out_files) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    out_aucs = list()
    out_confs = list()
    out_preds = {'Iso': list(), 'IsoShal': list()}

    phn_dict = dict()
    auc_lists = {ex_lbl: pd.Series([]) for ex_lbl in ['Iso', 'IsoShal']}
    conf_lists = {ex_lbl: pd.Series([]) for ex_lbl in ['Iso', 'IsoShal']}

    for out_file in out_files:
        out_tag = '_'.join(out_file.parts[-1].split('_')[1:])

        with bz2.BZ2File(Path(out_dir, '_'.join(["out-pheno", out_tag])),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(Path(out_dir, '_'.join(["out-aucs", out_tag])),
                         'r') as f:
            out_aucs += [pickle.load(f)]

        with bz2.BZ2File(Path(out_dir, '_'.join(["out-conf", out_tag])),
                         'r') as f:
            out_confs += [pickle.load(f)]

    mtypes_comp = np.greater_equal.outer(
        *([[set(auc_vals['Iso']['mean'].index)
            for auc_vals in out_aucs]] * 2)
        )
    super_list = np.apply_along_axis(all, 1, mtypes_comp)

    if super_list.any():
        super_indx = super_list.argmax()

        for ex_lbl in ['Iso', 'IsoShal']:
            auc_lists[ex_lbl] = pd.concat([
                auc_lists[ex_lbl], out_aucs[super_indx][ex_lbl]['mean']])
            conf_lists[ex_lbl] = pd.concat([
                conf_lists[ex_lbl], out_confs[super_indx][ex_lbl]['mean']])

    auc_df = pd.DataFrame(auc_lists)
    conf_df = pd.DataFrame(conf_lists)

    plot_size_comparisons(auc_df, phn_dict, args)


if __name__ == '__main__':
    main()

