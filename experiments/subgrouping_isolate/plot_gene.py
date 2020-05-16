
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subgrouping_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'gene')

from HetMan.experiments.subvariant_test import (
    variant_clrs, pnt_mtype, copy_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_isolate import cna_mtypes
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.utilities.mutations import Mcomb, ExMcomb
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_isolate.utils import get_fancy_label
from HetMan.experiments.utilities.label_placement import (
    place_scatterpie_labels)
from HetMan.experiments.subvariant_test.utils import get_cohort_label
from HetMan.experiments.subvariant_isolate import mcomb_clrs

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_size_comparisons(auc_vals, pheno_dict, conf_vals,
                          use_coh, args, add_lgnd=False):
    fig, ax = plt.subplots(figsize=(13, 8))

    pnt_dict = dict()
    clr_dict = dict()

    plt_df = pd.DataFrame({
        mut: {'Size': np.sum(pheno_dict[mut]), 'AUC': auc_val}
        for mut, auc_val in auc_vals.iteritems()
        if not isinstance(mut, RandomType)
        }).transpose().astype({'Size': int})

    for mut, (size_val, auc_val) in plt_df.iterrows():
        if isinstance(mut, MuType):
            sub_mut = mut.subtype_list()[0][1]
            plt_mrk = 'o'

            if (copy_mtype & sub_mut).is_empty():
                plt_clr = variant_clrs['Point']

            elif dict(cna_mtypes)['Gain'].is_supertype(sub_mut):
                plt_clr = variant_clrs['Gain']
            elif dict(cna_mtypes)['Loss'].is_supertype(sub_mut):
                plt_clr = variant_clrs['Loss']

            elif not (dict(cna_mtypes)['Gain'] & sub_mut).is_empty():
                plt_clr = mcomb_clrs['Point+Gain']
            elif not (dict(cna_mtypes)['Loss'] & sub_mut).is_empty():
                plt_clr = mcomb_clrs['Point+Loss']

            if sub_mut.is_supertype(pnt_mtype):
                plt_sz = 413
                lbl_gap = 0.31

                if sub_mut == pnt_mtype:
                    plt_lbl = "Any Point"

                elif sub_mut.is_supertype(dict(cna_mtypes)['Gain']):
                    plt_lbl = "Any Point + Any Gain"
                elif sub_mut.is_supertype(dict(cna_mtypes)['Loss']):
                    plt_lbl = "Any Point + Any Loss"

                elif sub_mut.is_supertype(gain_mtype):
                    plt_lbl = "Any Point + Deep Gains"
                elif sub_mut.is_supertype(loss_mtype):
                    plt_lbl = "Any Point + Deep Losses"

            else:
                plt_lbl = ''
                plt_sz = 31
                lbl_gap = 0.13

        elif len(mut.mtypes) == 1:
            iso_mtype = tuple(mut.mtypes)[0].subtype_list()[0][1]
            plt_mrk = 'D'

            if (copy_mtype & iso_mtype).is_empty():
                plt_clr = variant_clrs['Point']

            elif dict(cna_mtypes)['Gain'].is_supertype(iso_mtype):
                plt_clr = variant_clrs['Gain']
            elif dict(cna_mtypes)['Loss'].is_supertype(iso_mtype):
                plt_clr = variant_clrs['Loss']

            elif not (dict(cna_mtypes)['Gain'] & iso_mtype).is_empty():
                plt_clr = mcomb_clrs['Point+Gain']
            elif not (dict(cna_mtypes)['Loss'] & iso_mtype).is_empty():
                plt_clr = mcomb_clrs['Point+Loss']

            if iso_mtype.is_supertype(pnt_mtype):
                plt_sz = 413
                lbl_gap = 0.31

                if iso_mtype == pnt_mtype:
                    plt_lbl = "Only: Any Point"

                elif iso_mtype.is_supertype(dict(cna_mtypes)['Gain']):
                    plt_lbl = "Only: Any Point + Any Gain"
                elif iso_mtype.is_supertype(dict(cna_mtypes)['Loss']):
                    plt_lbl = "Only: Any Point + Any Loss"

                elif iso_mtype.is_supertype(gain_mtype):
                    plt_lbl = "Only: Any Point + Deep Gains"
                elif iso_mtype.is_supertype(loss_mtype):
                    plt_lbl = "Only: Any Point + Deep Losses"

            else:
                plt_lbl = ''
                plt_sz = 37
                lbl_gap = 0.19

        else:
            plt_mrk = 'D'

            if (size_val, auc_val) not in clr_dict:
                plt_clr = 'black'

            plt_lbl = ''
            plt_sz = 47
            lbl_gap = 0.29

        if (plt_lbl is not None
                and not ((size_val, auc_val) in pnt_dict
                         and pnt_dict[size_val, auc_val][0] > lbl_gap)):
            clr_dict[size_val, auc_val] = plt_clr
            pnt_dict[size_val, auc_val] = lbl_gap, (plt_lbl, '')

        ax.scatter(size_val, auc_val, marker=plt_mrk,
                   c=[plt_clr], s=plt_sz, alpha=0.19, edgecolor='none')

    size_min, size_max = plt_df.Size.quantile(q=[0, 1])
    auc_min, auc_max = plt_df.AUC.quantile(q=[0, 1])

    x_min = max(size_min - (size_max - size_min) / 29, 0)
    x_max = size_max + (size_max - size_min) / 29
    y_min, y_max = auc_min - (1 - auc_min) / 17, 1 + (1 - auc_min) / 113
    x_rng, y_rng = x_max - x_min, y_max - y_min

    ax.plot([x_min, x_max], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([x_min, x_max], [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([0, 0], [y_min, y_max], color='black', linewidth=1.9, alpha=0.89)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    coh_lbl = get_cohort_label(use_coh)
    ax.set_xlabel("# of Mutated Samples in {}".format(coh_lbl),
                  size=25, weight='semibold')
    ax.set_ylabel("Classification Task\nAccuracy in {}".format(coh_lbl),
                  size=25, weight='semibold')

    if pnt_dict:
        lbl_pos = place_scatterpie_labels(pnt_dict, fig, ax, seed=args.seed)

        for (pnt_x, pnt_y), pos in lbl_pos.items():
            ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                    pnt_dict[pnt_x, pnt_y][1][0],
                    size=11, ha=pos[1], va='bottom')

            x_delta = pnt_x - pos[0][0]
            y_delta = pnt_y - pos[0][1]

            if abs(x_delta) > x_rng / 23 or abs(y_delta) > y_rng / 23:
                end_x = pos[0][0] + np.sign(x_delta) * x_rng / 203
                end_y = pos[0][1] + np.heaviside(y_delta, 0) * y_rng / 29

                ln_x, ln_y = (pnt_x - end_x) / x_rng, (pnt_y - end_y) / y_rng
                ln_mag = (ln_x ** 2 + ln_y ** 2) ** 0.5
                ln_cos, ln_sin = ln_x / ln_mag, ln_y / ln_mag

                ax.plot([pnt_x - ln_cos * x_rng / 11
                         * pnt_dict[pnt_x, pnt_y][0], end_x],
                        [pnt_y - ln_sin * y_rng / 11
                         * pnt_dict[pnt_x, pnt_y][0], end_y],
                        c=clr_dict[pnt_x, pnt_y], linewidth=1.7, alpha=0.31)

    plt.savefig(
        os.path.join(plot_dir, args.gene,
                     "{}__size-comparison_{}_{}.svg".format(
                         use_coh, args.classif, args.expr_source)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Creates assorted plots for the output related to one particular "
        "mutated gene across all tested cohorts."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('gene', help="a mutated gene")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="print info about created plots")
    args = parser.parse_args()

    out_list = tuple(Path(base_dir).glob(
        os.path.join("{}__*".format(args.expr_source),
                     "out-siml__*__*__{}.p.gz".format(args.classif))
        ))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_use = pd.DataFrame(
        [{'Cohort': out_file.parts[-2].split('__')[1],
          'Levels': '__'.join(out_file.parts[-1].split('__')[1:-2]),
          'File': out_file}
         for out_file in out_list]
        )

    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    out_iter = out_use.groupby(['Cohort', 'Levels'])['File']
    out_aucs = {(coh, lvls): list() for coh, lvls in out_iter.groups}
    out_confs = {(coh, lvls): list() for coh, lvls in out_iter.groups}

    phn_dicts = {coh: dict() for coh in set(out_use.Cohort)}
    auc_dfs = {ex_lbl: {coh: pd.DataFrame([]) for coh in set(out_use.Cohort)}
               for ex_lbl in ['All', 'Iso', 'IsoShal']}
    conf_dfs = {ex_lbl: {coh: pd.DataFrame([]) for coh in set(out_use.Cohort)}
                for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for (coh, lvls), out_files in out_iter:
        for out_file in out_files:
            out_dir = os.path.join(base_dir,
                                   '__'.join([args.expr_source, coh]))
            out_tag = '__'.join(out_file.parts[-1].split('__')[1:])

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_vals = pickle.load(f)

                phn_dicts[coh].update({
                    mut: phns for mut, phns in phn_vals.items()
                    if mut.get_labels()[0] == args.gene
                    })

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-aucs", out_tag])),
                             'r') as f:
                auc_vals = pickle.load(f)

                out_aucs[coh, lvls] += [
                    {ex_lbl: {
                        cv_k: auc_list[[mut for mut in auc_list.index
                                        if mut.get_labels()[0] == args.gene]]
                        for cv_k, auc_list in auc_dict.items()
                        } for ex_lbl, auc_dict in auc_vals.items()}
                    ]

            with bz2.BZ2File(Path(out_dir, '__'.join(["out-conf", out_tag])),
                             'r') as f:
                conf_vals = pickle.load(f)

                out_confs[coh, lvls] += [
                    {ex_lbl: {
                        cv_k: conf_list[[mut for mut in conf_list.index
                                        if mut.get_labels()[0] == args.gene]]
                        for cv_k, conf_list in conf_dict.items()
                        } for ex_lbl, conf_dict in conf_vals.items()}
                    ]

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[coh, lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[ex_lbl][coh] = pd.concat([
                    auc_dfs[ex_lbl][coh],
                    pd.DataFrame(out_aucs[coh, lvls][super_indx][ex_lbl])
                    ])

                conf_dfs[ex_lbl][coh] = pd.concat([
                    conf_dfs[ex_lbl][coh],
                    pd.DataFrame(out_confs[coh, lvls][super_indx][ex_lbl])
                    ])

    for coh, coh_use in out_use.groupby('Cohort')['Levels']:
        if phn_dicts[coh]:
            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[ex_lbl][coh] = auc_dfs[ex_lbl][coh].loc[
                    ~auc_dfs[ex_lbl][coh].index.duplicated()]
                conf_dfs[ex_lbl][coh] = conf_dfs[ex_lbl][coh].loc[
                    ~conf_dfs[ex_lbl][coh].index.duplicated()]

            plot_size_comparisons(auc_dfs['All'][coh]['mean'], phn_dicts[coh],
                                  conf_dfs['All'][coh]['mean'], coh, args)

            if 'Consequence__Exon' not in set(coh_use.tolist()):
                if args.verbose:
                    print("Cannot compare AUCs until this experiment is run "
                          "with mutation levels `Consequence__Exon` "
                          "which tests genes' base mutations!")


if __name__ == '__main__':
    main()

