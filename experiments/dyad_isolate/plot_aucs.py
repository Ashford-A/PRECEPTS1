
from ..utilities.mutations import pnt_mtype, copy_mtype
from dryadic.features.mutations import MuType, MutComb

from ..dyad_isolate import base_dir
from ..utilities.labels import get_fancy_label
from ..utilities.label_placement import place_scatterpie_labels
from ..utilities.misc import choose_label_colour

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')


def plot_sub_comparisons(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    clr_dict = dict()

    use_aucs = auc_vals[[
        not isinstance(mtype, MutComb)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            if auc_vec[best_subtype] > 0.6:
                auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
                clr_dict[auc_tupl] = choose_label_colour(gene)

                base_size = np.mean(pheno_dict[base_mtype])
                best_prop = np.mean(pheno_dict[best_subtype]) / base_size
                conf_sc = np.greater.outer(conf_vals[best_subtype],
                                           conf_vals[base_mtype]).mean()

                if conf_sc > 0.8:
                    mtype_lbl = get_fancy_label(
                        tuple(best_subtype.subtype_iter())[0][1],
                        pnt_link='\n', phrase_link=' '
                        )

                    plot_dict[auc_tupl] = base_size ** 0.53, (gene, mtype_lbl)

                elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                    plot_dict[auc_tupl] = base_size ** 0.53, (gene, '')
                else:
                    plot_dict[auc_tupl] = base_size ** 0.53, ('', '')

                pie_ax = inset_axes(
                    ax, width=base_size ** 0.5, height=base_size ** 0.5,
                    bbox_to_anchor=auc_tupl, bbox_transform=ax.transData,
                    loc=10, axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop],
                           colors=[clr_dict[auc_tupl] + (0.77, ),
                                   clr_dict[auc_tupl] + (0.29, )],
                           explode=[0.29, 0], startangle=90)

    ax.plot([0.48, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [0.48, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([0.48, 1.0005], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [0.48, 1.0005], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0.49, 0.997], [0.49, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=23, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=23, weight='semibold')

    if plot_dict:
        lbl_pos = place_scatterpie_labels(plot_dict, clr_dict, fig, ax)

    ax.grid(alpha=0.43, linewidth=1.07)
    ax.set_xlim([0.48, 1.01])
    ax.set_ylim([0.48, 1.01])

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(Path(out_dir).glob(
        "out-conf_*_*_{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort])), exist_ok=True)

    out_use = pd.DataFrame([{'Levels': out_file.parts[-1].split('_')[2],
                             'File': out_file}
                            for out_file in out_list])

    out_iter = out_use.groupby('Levels')['File']
    out_aucs = {lvls: list() for lvls, _ in out_iter}
    out_confs = {lvls: list() for lvls, _ in out_iter}
    phn_dict = dict()

    auc_dfs = {ex_lbl: pd.DataFrame([])
               for ex_lbl in ['All', 'Iso', 'IsoShal']}
    conf_dfs = {ex_lbl: pd.DataFrame([])
                for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for lvls, out_files in out_iter:
        for out_file in out_files:
            out_tag = '_'.join(out_file.parts[-1].split('_')[1:])

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-aucs", out_tag])),
                             'r') as f:
                out_aucs[lvls] += [pickle.load(f)]

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-conf", out_tag])),
                             'r') as f:
                out_confs[lvls] += [pickle.load(f)]

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[ex_lbl] = pd.concat([
                    auc_dfs[ex_lbl],
                    pd.DataFrame(out_aucs[lvls][super_indx][ex_lbl])
                    ])

                conf_dfs[ex_lbl] = pd.concat([
                    conf_dfs[ex_lbl],
                    pd.DataFrame(out_confs[lvls][super_indx][ex_lbl])
                    ])

    auc_dfs = {ex_lbl: auc_df.loc[~auc_df.index.duplicated()]
               for ex_lbl, auc_df in auc_dfs.items()}
    conf_dfs = {ex_lbl: conf_df.loc[~conf_df.index.duplicated()]
                for ex_lbl, conf_df in conf_dfs.items()}

    plot_sub_comparisons(auc_dfs['All']['mean'], phn_dict,
                         conf_dfs['All']['mean'], args)


if __name__ == '__main__':
    main()

