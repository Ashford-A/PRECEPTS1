
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'dyad_isolate')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'interaction')

from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_isolate import cna_mtypes
from HetMan.experiments.utilities.mutations import ExMcomb
from dryadic.features.mutations import MuType

from HetMan.experiments.subgrouping_isolate.utils import calculate_pair_siml
from HetMan.experiments.utilities.misc import create_twotone_circle
from HetMan.experiments.subgrouping_isolate.plot_gene import (
    choose_subtype_colour)

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from itertools import combinations as combn
from itertools import product
from functools import reduce
from operator import or_
from scipy.stats import fisher_exact

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


AUC_CUTOFF = 0.8
SAMP_CUTOFF = 20


def plot_mutual_similarity(siml_dict, pheno_dict, auc_vals, pred_df,
                           ex_lbl, cdata, args):
    fig_size = (13, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    use_mtree = tuple(cdata.mtrees.values())[0]

    if ex_lbl == 'Iso':
        use_combs = {mcomb for mcomb in auc_vals.index
                     if not (mcomb.all_mtype
                             & dict(cna_mtypes)['Shal']).is_empty()}

    elif ex_lbl == 'IsoShal':
        use_combs = {mcomb for mcomb in auc_vals.index
                     if ((mcomb.all_mtype
                          & dict(cna_mtypes)['Shal']).is_empty()
                         and all((mtype & dict(cna_mtypes)['Shal']).is_empty()
                                 for mtype in mcomb.mtypes))}

    use_pairs = {(mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
                 if (set(mcomb1.get_labels()) == set(mcomb2.get_labels())
                     and (np.sum(pheno_dict[mcomb1] & ~pheno_dict[mcomb2])
                          >= SAMP_CUTOFF)
                     and (np.sum(~pheno_dict[mcomb1] & pheno_dict[mcomb2])
                          >= SAMP_CUTOFF)
                     and (all((mtype1 & mtype2).is_empty()
                              for mtype1, mtype2 in product(mcomb1.mtypes,
                                                            mcomb2.mtypes))
                          or not (pheno_dict[mcomb1]
                                  & pheno_dict[mcomb2]).any()))}

    if not use_pairs:
        return None

    mutex_dict = {mcombs: None for mcombs in use_pairs}
    pair_simls = {mcombs: tuple() for mcombs in use_pairs}

    for mcomb1, mcomb2 in use_pairs:
        ovlp_odds, ovlp_pval = fisher_exact(table=pd.crosstab(
            pheno_dict[mcomb1], pheno_dict[mcomb2]))

        mutex_dict[mcomb1, mcomb2] = -np.log10(ovlp_pval)
        if ovlp_odds < 1:
            mutex_dict[mcomb1, mcomb2] *= -1

        all_mtype = reduce(
            or_, [MuType({('Gene', gene): use_mtree[gene].allkey()})
                  for gene in mcomb1.get_labels()]
            )

        if ex_lbl == 'IsoShal':
            all_mtype -= MuType({('Gene', tuple(mcomb1.get_labels())): dict(
                cna_mtypes)['Shal']})

        pair_simls[mcomb1, mcomb2] = [
            calculate_pair_siml(mcomb1, mcomb2, all_mtype, siml_dict,
                                pheno_dict, pred_df, cdata),
            calculate_pair_siml(mcomb2, mcomb1, all_mtype, siml_dict,
                                pheno_dict, pred_df, cdata)
            ]

    plot_df = pd.DataFrame({'Occur': pd.Series(mutex_dict),
                            'Simil': pd.Series(pair_simls).apply(np.mean)})
    plot_df = plot_df.loc[~plot_df.isnull().any(axis=1)]
    plot_df = plot_df.sort_index()

    plot_lims = plot_df.quantile(q=[0, 1])
    plot_diff = plot_lims.diff().iloc[1]
    plot_lims.Occur += plot_diff.Occur * np.array([-7., 17.]) ** -1
    plot_lims.Simil += plot_diff.Simil * np.array([-17., 7.]) ** -1

    plot_rngs = plot_lims.diff().iloc[1]
    plot_lims.Occur[0] = min(plot_lims.Occur[0], -plot_rngs.Occur / 3.41)
    plot_lims.Occur[1] = max(plot_lims.Occur[1], plot_rngs.Occur / 3.41)
    plot_lims.Simil[0] = min(plot_lims.Simil[0], -plot_rngs.Simil / 2.23)
    plot_lims.Simil[1] = max(plot_lims.Simil[1], plot_rngs.Simil / 2.23)
    plot_rngs = plot_lims.diff().iloc[1]

    xy_scale = np.array([1, 2 ** np.log2(plot_rngs).diff()[-1]
                         * 2 ** -np.diff(np.log2(fig_size))])
    xy_scale /= (np.prod(plot_rngs) ** -0.76) * 11

    for (mcomb1, mcomb2), (occur_val, simil_val) in plot_df.iterrows():
        plt_clrs = [
            choose_subtype_colour(
                reduce(or_, mcomb.mtypes).subtype_list()[0][1])
            for mcomb in (mcomb1, mcomb2)
            ]

        plt_size = (pheno_dict[mcomb1].mean() * pheno_dict[mcomb2].mean())
        plt_size = (plt_size ** 0.25) * (plot_df.shape[0] ** -0.13)
 
        for ptch in create_twotone_circle((occur_val, simil_val),
                                          plt_clrs, scale=xy_scale * plt_size,
                                          alpha=0.23, edgecolor='none'):
            ax.add_artist(ptch)

    ax.text(plot_rngs.Occur / -97, plot_lims.Simil[1] - plot_rngs.Simil / 41,
            '\u2190', size=23, ha='right', va='center', weight='bold')
    ax.text(plot_rngs.Occur / -23, plot_lims.Simil[1] - plot_rngs.Simil / 41,
            "significant exclusivity", size=13, ha='right', va='center')

    ax.text(plot_rngs.Occur / 97, plot_lims.Simil[1] - plot_rngs.Simil / 41,
            '\u2192', size=23, ha='left', va='center', weight='bold')
    ax.text(plot_rngs.Occur / 23, plot_lims.Simil[1] - plot_rngs.Simil / 41,
            "significant overlap", size=13, ha='left', va='center')

    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / -71,
            '\u2190', size=23, rotation=90, ha='center', va='top',
            weight='bold')
    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / -13,
            "opposite\ndownstream\neffects",
            size=13, rotation=90, ha='center', va='top')

    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / 71,
            '\u2192', size=23, rotation=90, ha='center', va='bottom',
            weight='bold')
    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / 13,
            "similar\ndownstream\neffects",
            size=13, rotation=90, ha='center', va='bottom')

    plt.xticks(size=11)
    plt.yticks(size=11)
    ax.axhline(0, color='black', linewidth=1.7, linestyle='--', alpha=0.41)
    ax.axvline(0, color='black', linewidth=1.7, linestyle='--', alpha=0.41)

    plt.xlabel("Genomic Co-occurence", size=23, weight='semibold')
    plt.ylabel("Transcriptomic Similarity", size=23, weight='semibold')

    ax.set_xlim(*plot_lims.Occur)
    ax.set_ylim(*plot_lims.Simil)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_mutual-simil_{}.svg".format(ex_lbl, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the relationships between pairs of mutations as inferred "
        "from when they are isolated against one another in a given cohort."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(Path(out_dir).glob(
        "out-siml_*_*_{}.p.gz".format(args.classif)))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    os.makedirs(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort])), exist_ok=True)

    out_use = pd.DataFrame([{'Levels': out_file.parts[-1].split('_')[2],
                             'File': out_file}
                            for out_file in out_list])

    out_iter = out_use.groupby('Levels')['File']
    out_aucs = {lvls: list() for lvls in out_iter.groups}
    out_simls = {lvls: list() for lvls in out_iter.groups}
    out_preds = {lvls: list() for lvls in out_iter.groups}

    phn_dict = dict()
    cdata = None

    auc_lists = {ex_lbl: pd.Series([]) for ex_lbl in ['Iso', 'IsoShal']}
    siml_dicts = {ex_lbl: {lvls: None for lvls, _ in out_iter}
                  for ex_lbl in ['Iso', 'IsoShal']}
    pred_dfs = {ex_lbl: pd.DataFrame([]) for ex_lbl in ['Iso', 'IsoShal']}

    for lvls, out_files in out_iter:
        for out_file in out_files:
            out_tag = '_'.join(out_file.parts[-1].split('_')[1:])

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_vals = pickle.load(f)

                phn_dict.update({mut: phns for mut, phns in phn_vals.items()
                                 if isinstance(mut, ExMcomb)})

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-aucs", out_tag])),
                             'r') as f:
                auc_vals = pickle.load(f)

                out_aucs[lvls] += [
                    {ex_lbl: auc_dict['mean'][
                        [mut for mut in auc_dict['mean'].index
                         if isinstance(mut, ExMcomb)]
                        ]
                     for ex_lbl, auc_dict in auc_vals.items()}
                    ]

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-siml", out_tag])),
                             'r') as f:
                out_simls[lvls] += [pickle.load(f)]

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-pred", out_tag])),
                             'r') as f:
                pred_vals = pickle.load(f)

                out_preds[lvls] += [
                    {ex_lbl: pred_vals[ex_lbl].loc[
                        [mut for mut in pred_vals[ex_lbl].index
                         if isinstance(mut, ExMcomb)]
                        ]
                     for ex_lbl in ['Iso', 'IsoShal']}
                    ]

            with bz2.BZ2File(Path(out_dir,
                                  '_'.join(["cohort-data", out_tag])),
                             'r') as f:
                new_cdata = pickle.load(f)

                if cdata is None:
                    cdata = new_cdata
                else:
                    cdata.merge(new_cdata)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['Iso'].index)
                for auc_vals in out_aucs[lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['Iso', 'IsoShal']:
                auc_lists[ex_lbl] = pd.concat([
                    auc_lists[ex_lbl], out_aucs[lvls][super_indx][ex_lbl]
                    ], sort=False)

                pred_dfs[ex_lbl] = pd.concat([
                    pred_dfs[ex_lbl], out_preds[lvls][super_indx][ex_lbl]
                    ], sort=False)
                siml_dicts[ex_lbl][lvls] = out_simls[lvls][super_indx][ex_lbl]

    auc_lists = {ex_lbl: auc_list[(auc_list >= AUC_CUTOFF)
                                  & ~auc_list.index.duplicated()]
                 for ex_lbl, auc_list in auc_lists.items()}

    for ex_lbl in ['Iso', 'IsoShal']:
        plot_mutual_similarity(siml_dicts[ex_lbl], phn_dict,
                               auc_lists[ex_lbl], pred_dfs[ex_lbl],
                               ex_lbl, cdata, args)
        #TODO: plots in the same vein for synergy x occurence,
        # synergy x simil, simil1+simil2


if __name__ == '__main__':
    main()

