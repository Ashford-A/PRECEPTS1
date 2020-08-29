
from ..utilities.mutations import (
    pnt_mtype, dup_mtype, loss_mtype, gains_mtype, dels_mtype)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate.plot_gene import choose_subtype_colour
from ..subgrouping_isolate.utils import remove_pheno_dups
from ..utilities.labels import get_fancy_label
from ..subvariant_test.utils import get_cohort_label
from ..utilities.label_placement import place_scatterpie_labels

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


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'gene_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'experiment')


def plot_size_comparisons(auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    plot_dict = dict()
    clr_dict = dict()

    plt_df = pd.DataFrame({
        mut: {'Size': np.sum(pheno_dict[mut]), 'AUC': auc_val}
        for mut, auc_val in auc_vals.iteritems() if isinstance(mut, MuType)
        }).transpose().astype({'Size': int})
    plt_df = plt_df.loc[remove_pheno_dups(plt_df.index, pheno_dict)]

    for mtype, (size_val, auc_val) in plt_df.iterrows():
        plt_clr = choose_subtype_colour(mtype)

        if (mtype.is_supertype(pnt_mtype)
                or mtype in {dels_mtype, gains_mtype,
                             dup_mtype, loss_mtype}):
            plt_lbl = get_fancy_label(mtype)

            plt_sz = 347
            lbl_gap = 0.31
            edg_clr = plt_clr

        else:
            plt_lbl = ''
            plt_sz = 31
            lbl_gap = 0.13
            edg_clr = 'none'

        if plt_lbl is not None:
            clr_dict[size_val, auc_val] = plt_clr
            plot_dict[size_val, auc_val] = lbl_gap, (plt_lbl, '')

        ax.scatter(size_val, auc_val,
                   c=[plt_clr], s=plt_sz, alpha=0.23, edgecolor=edg_clr)

    size_min, size_max = plt_df.Size.quantile(q=[0, 1])
    auc_min, auc_max = plt_df.AUC.quantile(q=[0, 1])

    x_min = max(size_min - (size_max - size_min) / 29, 0)
    x_max = size_max + (size_max - size_min) / 6.1
    y_min, y_max = auc_min - (1 - auc_min) / 17, 1 + (1 - auc_min) / 113

    if plot_dict:
        lbl_pos = place_scatterpie_labels(plot_dict, clr_dict, fig, ax,
                                          [(x_min, x_max), (y_min, y_max)])

    ax.plot([x_min, x_max], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([x_min, x_max], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0, 0], [y_min, y_max], color='black', linewidth=1.9, alpha=0.89)

    coh_lbl = get_cohort_label(args.cohort)
    ax.set_xlabel("# of Mutated Samples in {}".format(coh_lbl),
                  size=25, weight='bold')
    ax.set_ylabel("Classification Task\nAccuracy in {}".format(coh_lbl),
                  size=25, weight='bold')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__size-comparison_{}.svg".format(
                         args.cohort, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_experiment',
        description="Makes plots specific to one iteration of the pipeline."
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
    phn_dict = dict()

    auc_dfs = {ex_lbl: pd.DataFrame([])
               for ex_lbl in ['All', 'Iso', 'IsoShal']}
    conf_dfs = {ex_lbl: pd.DataFrame([])
                for ex_lbl in ['All', 'Iso', 'IsoShal']}

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
        *([[set(auc_vals['All']['mean'].index)
            for auc_vals in out_aucs]] * 2)
        )
    super_list = np.apply_along_axis(all, 1, mtypes_comp)

    if super_list.any():
        super_indx = super_list.argmax()

        for ex_lbl in ['All', 'Iso', 'IsoShal']:
            auc_dfs[ex_lbl] = pd.concat([auc_dfs[ex_lbl],
                                         out_aucs[super_indx][ex_lbl]])
            conf_dfs[ex_lbl] = pd.concat([conf_dfs[ex_lbl],
                                          out_confs[super_indx][ex_lbl]])

    auc_dfs = {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
               for ex_lbl, auc_df in auc_dfs.items()}
    conf_dfs = {ex_lbl: conf_df.loc[~conf_df.index.duplicated()]
                for ex_lbl, conf_df in conf_dfs.items()}

    plot_size_comparisons(auc_dfs['All']['mean'], phn_dict, args)


if __name__ == '__main__':
    main()

