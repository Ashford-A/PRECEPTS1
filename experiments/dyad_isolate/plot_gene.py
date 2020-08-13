"""
Creates assorted plots for the output related to one particular
mutated gene across all tested cohorts.
"""

from .plot_interaction import remove_pair_dups
from ..utilities.mutations import shal_mtype, Mcomb, ExMcomb
from ..subgrouping_isolate.utils import calculate_mean_siml, calculate_ks_siml
from ..subgrouping_isolate.plot_gene import choose_subtype_colour
from dryadic.features.mutations import MuType

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from itertools import combinations as combn
from itertools import product
from functools import reduce
from operator import or_, add

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'


base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'dyad_isolate')
plot_dir = os.path.join(base_dir, 'plots', 'gene')
SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}


def plot_mutual_similarity(pred_df, pheno_dict, auc_vals,
                           cdata, args, use_coh, ex_lbl, siml_metric):
    use_mtree = tuple(cdata.mtrees.values())[0]
    use_combs = {mut for mut in auc_vals.index.tolist()
                 if isinstance(mut, ExMcomb) and len(mut.mtypes) == 1}

    if ex_lbl == 'Iso':
        use_combs = {mcomb for mcomb in use_combs
                     if not (mcomb.all_mtype & shal_mtype).is_empty()}

    elif ex_lbl == 'IsoShal':
        use_combs = {mcomb for mcomb in use_combs
                     if ((mcomb.all_mtype & shal_mtype).is_empty()
                         and all((mtype & shal_mtype).is_empty()
                                 for mtype in mcomb.mtypes))}

    base_phns = {mcomb: (pheno_dict[Mcomb(*mcomb.mtypes)]
                         if Mcomb(*mcomb.mtypes) in pheno_dict
                         else np.array(cdata.get_pheno(Mcomb(*mcomb.mtypes))))
                 for mcomb in use_combs}

    use_pairs = [(mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
                 if (set(mcomb1.label_iter()) == set(mcomb2.label_iter())
                     and (all((mtype1 & mtype2).is_empty()
                              for mtype1, mtype2 in product(mcomb1.mtypes,
                                                            mcomb2.mtypes))
                          or not (pheno_dict[mcomb1]
                                  & pheno_dict[mcomb2]).any()))]

    use_pairs = remove_pair_dups(use_pairs, pheno_dict)
    pair_combs = set(reduce(add, use_pairs, tuple()))

    if args.verbose:
        print("{}({}):   {} pairs containing {} unique mutation types were "
              "produced from {} possible types".format(
                  use_coh, ex_lbl,
                  len(use_pairs), len(pair_combs), len(use_combs)
                ))

    if not use_pairs:
        return None

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.grid(alpha=0.53, linewidth=0.53)

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
    plot_lims.Occur += plot_diff.Occur * np.array([-17., 4.3]) ** -1
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

        if (set(tuple(mcomb1.mtypes)[0].label_iter())
                == set(tuple(mcomb2.mtypes)[0].label_iter())):
            use_mrk = 'D'
        else:
            use_mrk = 'o'

        gene_stat = [args.gene in tuple(mcomb.mtypes)[0].label_iter()
                     for mcomb in (mcomb1, mcomb2)]

        plt_clrs = [
            choose_subtype_colour(
                tuple(tuple(mcomb.mtypes)[0].subtype_iter())[0][1])
            if gene_stat[i] else None
            for i, mcomb in enumerate((mcomb1, mcomb2))
            ]

        if gene_stat[0] ^ gene_stat[1] or plt_clrs[0] != plt_clrs[1]:
            for i, (plt_half, mcomb) in enumerate(zip(['left', 'right'],
                                                      [mcomb1, mcomb2])):
                if gene_stat[i]:
                    mrk_style = MarkerStyle(use_mrk, fillstyle=plt_half)

                    ax.scatter(occur_val, simil_val,
                               s=plt_sz, marker=mrk_style,
                               facecolor=plt_clrs[i], edgecolor='none',
                               alpha=13 / 79)

        else:
            if all(gene_stat):
                fc_clr = plt_clrs[0]
                eg_clr = 'none'
                lw = 0

            else:
                fc_clr = 'none'
                eg_clr = '0.31'
                lw = 1.9

            ax.scatter(occur_val, simil_val, s=plt_sz, marker=use_mrk,
                       facecolor=fc_clr, edgecolor=eg_clr, linewidth=lw,
                       alpha=13 / 79)

    x_plcs = plot_rngs.Occur / 97, plot_rngs.Occur / 23
    y_plc = plot_lims.Simil[1] - plot_rngs.Simil / 41

    ax.text(-x_plcs[0], y_plc, '\u2190',
            size=23, ha='right', va='center', weight='bold')
    ax.text(-x_plcs[1], y_plc, "significant exclusivity",
            size=13, ha='right', va='center')

    ax.text(x_plcs[0], y_plc, '\u2192',
            size=23, ha='left', va='center', weight='bold')
    ax.text(x_plcs[1], y_plc, "significant overlap",
            size=13, ha='left', va='center')

    x_plc = plot_lims.Occur[1] - plot_rngs.Occur / 17
    y_plcs = plot_rngs.Simil / 71, plot_rngs.Simil / 17

    ax.text(x_plc, -y_plcs[0], '\u2190',
            size=23, rotation=90, ha='center', va='top', weight='bold')
    ax.text(x_plc, -y_plcs[1], "opposite\ndownstream\neffects",
            size=13, ha='center', va='top')

    ax.text(x_plc, y_plcs[0], '\u2192',
            size=23, rotation=90, ha='center', va='bottom', weight='bold')
    ax.text(x_plc, y_plcs[1], "similar\ndownstream\neffects",
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
        os.path.join(plot_dir, args.gene,
                     "{}__{}__{}-mutual-simil_{}_{}.svg".format(
                         use_coh, ex_lbl, siml_metric,
                         args.classif, args.expr_source
                         )
                     ),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_gene',
        description="Plots gene-specific experiment output across cohorts."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('gene', help="a mutated gene")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument('--cohorts', nargs='+')
    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.7)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="print info about created plots")

    args = parser.parse_args()
    out_list = tuple(Path(base_dir).glob(
        os.path.join("{}__*".format(args.expr_source),
                     "out-conf_*_*_{}.p.gz".format(args.classif))
        ))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_df = pd.DataFrame([{'Cohort': out_file.parts[-2].split('__')[1],
                            'Levels': out_file.parts[-1].split('_')[2],
                            'File': out_file}
                           for out_file in out_list])

    if args.cohorts:
        out_df = out_df.loc[out_df.Cohort.isin(args.cohorts)]

        if out_df.shape[0] == 0:
            raise ValueError("No completed experiments found for given "
                             "cohort(s) {} !".format(set(args.cohorts)))

    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    out_iter = out_df.groupby(['Cohort', 'Levels'])['File']
    phn_dicts = {coh: dict() for coh in out_df.Cohort.unique()}
 
    out_dirs = {coh: Path(base_dir, '__'.join([args.expr_source, coh]))
                for coh in out_df.Cohort.values}
    out_tags = {fl: '_'.join(fl.parts[-1].split('_')[1:])
                for fl in out_df.File}

    for (coh, lvls), out_files in out_iter:
        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[coh],
                                  '_'.join(["out-pheno",
                                             out_tags[out_file]])),
                             'r') as f:
                phn_vals = pickle.load(f)

            phn_dicts[coh].update({
                mut: phns for mut, phns in phn_vals.items()
                if args.gene in mut.label_iter()
                })

    use_cohs = {coh for coh, phn_dict in phn_dicts.items() if phn_dict}
    if not use_cohs:
        raise ValueError("No completed experiments found having tested "
                         "mutations of the gene {} for the given "
                         "parameters!".format(args.gene))

    out_use = out_df.loc[out_df.Cohort.isin(use_cohs)]
    use_iter = out_use.groupby(['Cohort', 'Levels'])['File']

    out_aucs = {(coh, lvls): list() for coh, lvls in use_iter.groups}
    out_confs = {(coh, lvls): list() for coh, lvls in use_iter.groups}
    out_preds = {(coh, lvls): list() for coh, lvls in use_iter.groups}
    cdata_dict = {coh: None for coh, _ in use_iter.groups}

    auc_dfs = {coh: {ex_lbl: pd.DataFrame([])
                     for ex_lbl in ['All', 'Iso', 'IsoShal']}
               for coh in use_cohs}
    conf_dfs = {coh: {ex_lbl: pd.DataFrame([])
                      for ex_lbl in ['All', 'Iso', 'IsoShal']}
                for coh in use_cohs}
    pred_dfs = {coh: {ex_lbl: pd.DataFrame([])
                      for ex_lbl in ['All', 'Iso', 'IsoShal']}
                for coh in use_cohs}

    for (coh, lvls), out_files in use_iter:
        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[coh],
                                  '_'.join(["out-aucs", out_tags[out_file]])),
                             'r') as f:
                auc_vals = pickle.load(f)

            out_aucs[coh, lvls] += [
                {ex_lbl: auc_df.loc[[mut for mut in auc_df.index
                                     if args.gene in mut.label_iter()]]
                 for ex_lbl, auc_df in auc_vals.items()}
                ]

            with bz2.BZ2File(Path(out_dirs[coh],
                                  '_'.join(["out-conf", out_tags[out_file]])),
                             'r') as f:
                conf_vals = pickle.load(f)

            out_confs[coh, lvls] += [{
                ex_lbl: pd.DataFrame(conf_dict).loc[
                    out_aucs[coh, lvls][-1][ex_lbl].index]
                for ex_lbl, conf_dict in conf_vals.items()
                }]

            with bz2.BZ2File(Path(out_dirs[coh],
                                  '_'.join(["out-pred", out_tags[out_file]])),
                             'r') as f:
                pred_vals = pickle.load(f)

            out_preds[coh, lvls] += [{
                ex_lbl: pd.DataFrame(pred_dict).loc[
                    out_aucs[coh, lvls][-1][ex_lbl].index]
                for ex_lbl, pred_dict in pred_vals.items()
                }]

            with bz2.BZ2File(Path(out_dirs[coh],
                                  '_'.join(["cohort-data",
                                             out_tags[out_file]])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[coh] is None:
                cdata_dict[coh] = new_cdata
            else:
                cdata_dict[coh].merge(new_cdata, use_genes=[args.gene])

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals['All']['mean'].index)
                for auc_vals in out_aucs[coh, lvls]]] * 2)
            )
        super_list = np.apply_along_axis(all, 1, mtypes_comp)

        if super_list.any():
            super_indx = super_list.argmax()

            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                auc_dfs[coh][ex_lbl] = pd.concat([
                    auc_dfs[coh][ex_lbl],
                    out_aucs[coh, lvls][super_indx][ex_lbl]
                    ], sort=False)

                conf_dfs[coh][ex_lbl] = pd.concat([
                    conf_dfs[coh][ex_lbl],
                    out_confs[coh, lvls][super_indx][ex_lbl]
                    ], sort=False)

                pred_dfs[coh][ex_lbl] = pd.concat([
                    pred_dfs[coh][ex_lbl],
                    out_preds[coh, lvls][super_indx][ex_lbl]
                    ], sort=False)

    for coh, coh_lvls in out_use.groupby('Cohort')['Levels']:
        for ex_lbl in ['Iso', 'IsoShal']:
            auc_lists = auc_dfs[coh][ex_lbl]['mean'].loc[
                (auc_dfs[coh][ex_lbl]['mean'] >= args.auc_cutoff)
                & ~auc_dfs[coh][ex_lbl].index.duplicated()
                ]
            conf_dfs[coh][ex_lbl] = conf_dfs[coh][ex_lbl]['mean'].loc[
                ~conf_dfs[coh][ex_lbl].index.duplicated()]

            for siml_metric in args.siml_metrics:
                plot_mutual_similarity(pred_dfs[coh][ex_lbl], phn_dicts[coh],
                                       auc_lists, cdata_dict[coh],
                                       args, coh, ex_lbl, siml_metric)


if __name__ == '__main__':
    main()

