
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, shal_mtype,
    dup_mtype, loss_mtype, gains_mtype, dels_mtype, Mcomb, ExMcomb
    )
from dryadic.features.mutations import MuType

from ..subgrouping_isolate.utils import calculate_mean_siml, calculate_ks_siml
from ..subgrouping_isolate.plot_gene import choose_subtype_colour

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

import numpy as np
import pandas as pd

from itertools import combinations as combn
from itertools import product
from functools import reduce
from operator import or_, add
from scipy.stats import fisher_exact

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'dyad_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'interaction')
SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}


def remove_pair_dups(mut_pairs, pheno_dict):
    pair_infos = set()
    pair_list = set()

    for mut1, mut2 in mut_pairs:
        pair_info = tuple(sorted([tuple(pheno_dict[mut1]),
                                  tuple(pheno_dict[mut2])]))
        pair_info += tuple(sorted(set(mut1.label_iter())
                                  | set(mut2.label_iter())))

        if pair_info not in pair_infos:
            pair_infos |= {pair_info}
            pair_list |= {(mut1, mut2)}

    return pair_list


def plot_mutual_similarity(pred_df, pheno_dict, auc_vals,
                           cdata, args, ex_lbl, siml_metric):
    use_mtree = tuple(cdata.mtrees.values())[0]
    use_combs = {mcomb for mcomb in auc_vals.index.tolist()
                 if len(mcomb.mtypes) == 1}

    if ex_lbl == 'Iso':
        use_combs = {mcomb for mcomb in use_combs
                     if not (mcomb.all_mtype & shal_mtype).is_empty()}

    elif ex_lbl == 'IsoShal':
        use_combs = {mcomb for mcomb in use_combs
                     if ((mcomb.all_mtype & shal_mtype).is_empty()
                         and all((mtype & shal_mtype).is_empty()
                                 for mtype in mcomb.mtypes))}

    base_phns = {mcomb: pheno_dict[Mcomb(*mcomb.mtypes)]
                 for mcomb in use_combs}

    use_pairs = [(mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
                 if (set(mcomb1.label_iter()) == set(mcomb2.label_iter())
                     and (all((mtype1 & mtype2).is_empty()
                              for mtype1, mtype2 in product(mcomb1.mtypes,
                                                            mcomb2.mtypes))
                          or not (pheno_dict[mcomb1]
                                  & pheno_dict[mcomb2]).any()))]

    use_pairs = remove_pair_dups(use_pairs, pheno_dict)
    pair_combs = set(reduce(add, use_pairs))

    if args.verbose:
        print("{}:   {} pairs containing {} unique mutation types were "
              "produced from {} possible types".format(
                  ex_lbl, len(use_pairs), len(pair_combs), len(use_combs)))

    if not use_pairs:
        return None

    fig, ax = plt.subplots(figsize=(13, 8))
    train_samps = cdata.get_train_samples()
    use_preds = pred_df.loc[pair_combs, train_samps].applymap(np.mean)
    mutex_dict = {mcombs: None for mcombs in use_pairs}
    map_args = list()

    for mcomb1, mcomb2 in use_pairs:
        ovlp_odds, ovlp_pval = fisher_exact(table=pd.crosstab(
            base_phns[mcomb1], base_phns[mcomb2]))

        mutex_dict[mcomb1, mcomb2] = -np.log10(ovlp_pval)
        if ovlp_odds < 1:
            mutex_dict[mcomb1, mcomb2] *= -1

        all_mtype = reduce(
            or_, [MuType({('Gene', gene): use_mtree[gene].allkey()})
                  for gene in mcomb1.label_iter()]
            )

        if ex_lbl == 'IsoShal':
            all_mtype -= MuType({
                ('Gene', tuple(mcomb1.label_iter())): shal_mtype})

        all_phn = np.array(cdata.train_pheno(all_mtype))
        wt_vals = {mcomb: use_preds.loc[mcomb, ~all_phn]
                   for mcomb in (mcomb1, mcomb2)}
        mut_vals = {mcomb: use_preds.loc[mcomb, pheno_dict[mcomb]]
                    for mcomb in (mcomb1, mcomb2)}

        map_args += [(wt_vals[mcomb1], mut_vals[mcomb1],
                      use_preds.loc[mcomb1, pheno_dict[mcomb2]]),
                     (wt_vals[mcomb2], mut_vals[mcomb2],
                      use_preds.loc[mcomb2, pheno_dict[mcomb1]])]

    pool = mp.Pool(args.cores)
    siml_list = pool.starmap(SIML_FXS[siml_metric], map_args, chunksize=1)
    pool.close()
    siml_vals = dict(zip(use_pairs, zip(siml_list[::2], siml_list[1::2])))

    plot_df = pd.DataFrame({'Occur': pd.Series(mutex_dict),
                            'Simil': pd.Series(siml_vals).apply(np.mean)})

    plot_lims = plot_df.quantile(q=[0, 1])
    plot_diff = plot_lims.diff().iloc[1]
    plot_lims.Occur += plot_diff.Occur * np.array([-4.3, 17.]) ** -1
    plot_lims.Simil += plot_diff.Simil * np.array([-17., 4.3]) ** -1
    plot_rngs = plot_lims.diff().iloc[1]

    plot_lims.Occur[0] = min(plot_lims.Occur[0], -plot_rngs.Occur / 3.41,
                             -1.07)
    plot_lims.Occur[1] = max(plot_lims.Occur[1], plot_rngs.Occur / 3.41,
                             1.07)

    plot_lims.Simil[0] = min(plot_lims.Simil[0], -plot_rngs.Simil / 2.23,
                             -0.53)
    plot_lims.Simil[1] = max(plot_lims.Simil[1], plot_rngs.Simil / 2.23,
                             0.53)

    plot_rngs = plot_lims.diff().iloc[1]
    size_mult = 20103 * len(map_args) ** (-3 / 7)

    for (mcomb1, mcomb2), (occur_val, simil_val) in plot_df.iterrows():
        plt_sz = size_mult * (pheno_dict[mcomb1].mean()
                              * pheno_dict[mcomb2].mean()) ** 0.5

        for i, (plt_half, mcomb) in enumerate(zip(['left', 'right'],
                                                  [mcomb1, mcomb2])):
            plt_clr = choose_subtype_colour(
                tuple(reduce(or_, mcomb.mtypes).subtype_iter())[0][1])

            if (set(tuple(mcomb1.mtypes)[0].label_iter())
                    == set(tuple(mcomb2.mtypes)[0].label_iter())):
                mrk_style = MarkerStyle('D', fillstyle=plt_half)
            else:
                mrk_style = MarkerStyle('o', fillstyle=plt_half)

            ax.scatter(occur_val, simil_val, s=plt_sz, facecolor=plt_clr,
                       marker=mrk_style, alpha=11 / 79, edgecolor='none')

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
    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / -17,
            "opposite\ndownstream\neffects",
            size=13, ha='center', va='top')

    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / 71,
            '\u2192', size=23, rotation=90, ha='center', va='bottom',
            weight='bold')
    ax.text(plot_lims.Occur[0] + plot_rngs.Occur / 25, plot_rngs.Simil / 17,
            "similar\ndownstream\neffects",
            size=13, ha='center', va='bottom')

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
                     "{}_{}-mutual-simil_{}.svg".format(
                         ex_lbl, siml_metric, args.classif)),
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

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.7)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})
    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    out_dir = Path(base_dir, '__'.join([args.expr_source, args.cohort]))
    out_list = tuple(out_dir.glob(
        "out-aucs_*_*_{}.p.gz".format(args.classif)))

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
    out_preds = {lvls: list() for lvls in out_iter.groups}

    phn_dict = dict()
    cdata = None

    auc_lists = {ex_lbl: pd.Series([]) for ex_lbl in ['Iso', 'IsoShal']}
    pred_dfs = {ex_lbl: pd.DataFrame([]) for ex_lbl in ['Iso', 'IsoShal']}

    for lvls, out_files in out_iter:
        for out_file in out_files:
            out_tag = '_'.join(out_file.parts[-1].split('_')[1:])

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-pheno", out_tag])),
                             'r') as f:
                phn_dict.update(pickle.load(f))

            with bz2.BZ2File(Path(out_dir, '_'.join(["out-aucs", out_tag])),
                             'r') as f:
                auc_vals = pickle.load(f)

            out_aucs[lvls] += [{
                ex_lbl: auc_dict['mean'][
                    [mut for mut in auc_dict['mean'].index
                     if isinstance(mut, ExMcomb)]
                    ]
                for ex_lbl, auc_dict in auc_vals.items()
                if ex_lbl in ['Iso', 'IsoShal']
                }]

            # TODO: this is responsible for more than half of the time needed
            # to load output data, can we make it more efficient?
            with bz2.BZ2File(Path(out_dir, '_'.join(["out-pred", out_tag])),
                             'r') as f:
                pred_vals = pickle.load(f)

            out_preds[lvls] += [{
                ex_lbl: pred_vals[ex_lbl].loc[
                    out_aucs[lvls][-1][ex_lbl].index]
                for ex_lbl in ['Iso', 'IsoShal']
                }]

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

    auc_lists = {ex_lbl: auc_list[(auc_list >= args.auc_cutoff)
                                  & ~auc_list.index.duplicated()]
                 for ex_lbl, auc_list in auc_lists.items()}

    for ex_lbl in ['Iso', 'IsoShal']:
        for siml_metric in args.siml_metrics:
            plot_mutual_similarity(
                pred_dfs[ex_lbl], phn_dict, auc_lists[ex_lbl],
                cdata, args, ex_lbl, siml_metric
                )

        # TODO: plots in the same vein for synergy x occurence,
        # synergy x simil, simil1+simil2


if __name__ == '__main__':
    main()

