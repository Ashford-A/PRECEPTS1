
from ..utilities.mutations import pnt_mtype, shal_mtype, ExMcomb
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir
from ..subgrouping_test import train_cohorts
from .utils import remove_pheno_dups
from ..utilities.metrics import calculate_mean_siml, calculate_ks_siml
from ..utilities.labels import get_fancy_label, get_cohort_label
from ..utilities.misc import choose_label_colour
from ..utilities.colour_maps import simil_cmap
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import multiprocessing as mp

from functools import reduce
from operator import add
from itertools import combinations as combn
from itertools import permutations as permt
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'point')

SIML_FXS = {'mean': calculate_mean_siml, 'ks': calculate_ks_siml}


def plot_divergent_pairs(pred_dfs, pheno_dicts, auc_lists,
                         cdata_dict, args, siml_metric):
    fig, ax = plt.subplots(figsize=(12, 7))

    divg_lists = dict()
    for (src, coh), auc_list in auc_lists.items():
        use_combs = remove_pheno_dups({
            mut for mut, auc_val in auc_list.iteritems()
            if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                and all(
                    pnt_mtype.is_supertype(tuple(mtype.subtype_iter())[0][1])
                    for mtype in mut.mtypes
                    ))
            }, pheno_dicts[src, coh])

        if args.ex_lbl == 'Iso':
            use_combs = {mcomb for mcomb in use_combs
                         if not (mcomb.all_mtype & shal_mtype).is_empty()}

        else:
            use_combs = {mcomb for mcomb in use_combs
                         if (mcomb.all_mtype & shal_mtype).is_empty()}

        use_pairs = {
            (mcomb1, mcomb2) for mcomb1, mcomb2 in combn(use_combs, 2)
            if ((tuple(mcomb1.label_iter())[0]
                 == tuple(mcomb2.label_iter())[0])
                and (all((mtp1 & mtp2).is_empty()
                         for mtp1, mtp2 in product(mcomb1.mtypes,
                                                   mcomb2.mtypes))
                     or not (pheno_dicts[src, coh][mcomb1]
                             & pheno_dicts[src, coh][mcomb2]).any()))
            }

        if not use_pairs:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {tuple(mcomb.label_iter())[0]
                     for comb_pair in use_pairs for mcomb in comb_pair}

        all_mtypes = {
            gene: MuType({('Gene', gene): base_mtree[gene].allkey()})
            for gene in use_genes
            }

        if args.ex_lbl == 'IsoShal':
            for gene in use_genes:
                all_mtypes[gene] -= MuType({('Gene', gene): shal_mtype})

        all_phns = {
            gene: np.array(cdata_dict[src, coh].train_pheno(all_mtype))
            for gene, all_mtype in all_mtypes.items()
            }

        train_samps = cdata_dict[src, coh].get_train_samples()
        pair_combs = set(reduce(add, use_pairs))
        use_preds = pred_dfs[src, coh].loc[pair_combs, train_samps]
        map_args = []

        wt_vals = {
            mcomb: use_preds.loc[mcomb][~all_phns[
                tuple(mcomb.label_iter())[0]]]
            for mcomb in pair_combs
            }

        mut_vals = {mcomb: use_preds.loc[mcomb, pheno_dicts[src, coh][mcomb]]
                    for mcomb in pair_combs}

        if siml_metric == 'mean':
            wt_means = {mcomb: vals.mean() for mcomb, vals in wt_vals.items()}
            mut_means = {mcomb: vals.mean()
                         for mcomb, vals in mut_vals.items()}

            map_args += [
                (wt_vals[mcomb1], mut_vals[mcomb1],
                 use_preds.loc[mcomb1, pheno_dicts[src, coh][mcomb2]],
                 wt_means[mcomb1], mut_means[mcomb1], None)
                for mcombs in use_pairs for mcomb1, mcomb2 in permt(mcombs)
                ]

        elif siml_metric == 'ks':
            base_dists = {
                mcomb: ks_2samp(wt_vals[mcomb], mut_vals[mcomb],
                                alternative='greater').statistic
                for mcomb in pair_combs
                }

            map_args += [
                (wt_vals[mcomb1], mut_vals[mcomb1],
                 use_preds.loc[mcomb1, pheno_dicts[src, coh][mcomb2]],
                 base_dists[mcomb1])
                for mcombs in use_pairs for mcomb1, mcomb2 in permt(mcombs)
                ]

        if siml_metric == 'mean':
            chunk_size = int(len(map_args) / (3 * args.cores)) + 1
        elif siml_metric == 'ks':
            chunk_size = int(len(map_args) / (7 * args.cores)) + 1

        pool = mp.Pool(args.cores)
        siml_list = pool.starmap(SIML_FXS[siml_metric], map_args, chunk_size)
        pool.close()
        siml_df = pd.DataFrame(dict(
            zip(use_pairs, zip(siml_list[::2], siml_list[1::2])))).transpose()

        divg_list = siml_df.abs().max(axis=1).sort_values()
        divg_indx = {mcomb: None for mcomb in pair_combs}
        divg_pairs = set()

        for mcomb1, mcomb2 in divg_list.index:
            if divg_indx[mcomb1] is None and divg_indx[mcomb2] is None:
                divg_indx[mcomb1] = mcomb2
                divg_indx[mcomb2] = mcomb1
                divg_pairs |= {(mcomb1, mcomb2)}

            if not any(v is None for v in divg_indx.values()):
                break

        divg_lists[src, coh] = divg_list.loc[divg_pairs]

    size_mult = 537 * sum(len(divg_list)
                          for divg_list in divg_lists.values()) ** 0.31

    for (src, coh), divg_list in divg_lists.items():
        for (mcomb1, mcomb2), divg_val in divg_list.iteritems():
            cur_gene = tuple(mcomb1.label_iter())[0]

            plt_sz = (np.mean(pheno_dicts[src, coh][mcomb1])
                      * np.mean(pheno_dicts[src, coh][mcomb2])) ** 0.5
            plt_sz *= size_mult
            use_clr = choose_label_colour('+'.join([cur_gene, coh]))

            ax.scatter(auc_lists[src, coh][[mcomb1, mcomb2]].min(), divg_val,
                       s=plt_sz, c=[use_clr], alpha=0.17, edgecolor='none')

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 181]
    ymax = max(divg_list.max() for divg_list in divg_lists.values()) * 1.13
    ylims = [ymax / -173, ymax]

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot(xlims, [0, 0], color='black', linewidth=1.7, alpha=0.83)
    ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

    ax.set_xlabel("Minimum Classification Accuracy", size=21, weight='bold')
    ax.set_ylabel("Maximum Absolute Similarity", size=21, weight='bold')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    plt.savefig(os.path.join(plot_dir,
                             "{}_{}-siml-symmetry_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_similarities',
        description="Compares pairs of genes' subgroupings with a cohort."
        )

    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('ex_lbl', help="a classification mode",
                        choices={'Iso', 'IsoShal'})

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.8)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    out_datas = tuple(Path(base_dir).glob(
        os.path.join("*", "out-aucs__*__*__{}.p.gz".format(args.classif))))

    os.makedirs(plot_dir, exist_ok=True)
    out_list = pd.DataFrame(
        [{'Source': '__'.join(out_data.parts[-2].split('__')[:-1]),
          'Cohort': out_data.parts[-2].split('__')[-1],
          'Levels': '__'.join(out_data.parts[-1].split('__')[1:-2]),
          'File': out_data}
         for out_data in out_datas]
        ).groupby('Cohort').filter(
            lambda outs: 'Consequence__Exon' in set(outs.Levels))

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_list = out_list[out_list.Cohort.isin(train_cohorts)]
    use_iter = out_list.groupby(['Source', 'Cohort', 'Levels'])['File']

    out_dirs = {(src, coh): Path(base_dir, '__'.join([src, coh]))
                for src, coh, _ in use_iter.groups}
    out_tags = {fl: '__'.join(fl.parts[-1].split('__')[1:])
                for fl in out_list.File}
    pred_tag = "out-pred_{}".format(args.ex_lbl)

    phn_dicts = {(src, coh): dict() for src, coh, _ in use_iter.groups}
    cdata_dict = {(src, coh): None for src, coh, _ in use_iter.groups}

    auc_lists = {(src, coh): pd.Series() for src, coh, _ in use_iter.groups}
    pred_dfs = {(src, coh): pd.DataFrame() for src, coh, _ in use_iter.groups}

    for (src, coh, lvls), out_files in use_iter:
        out_aucs = list()
        out_preds = list()

        for out_file in out_files:
            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["out-pheno",
                                             out_tags[out_file]])),
                             'r') as f:
                phn_dicts[src, coh].update(pickle.load(f))

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["out-aucs",
                                             out_tags[out_file]])),
                             'r') as f:
                out_aucs += [pickle.load(f)[args.ex_lbl]['mean']]

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join([pred_tag, out_tags[out_file]])),
                             'r') as f:
                pred_vals = pickle.load(f)

            out_preds += [pred_vals.applymap(np.mean)]

            with bz2.BZ2File(Path(out_dirs[src, coh],
                                  '__'.join(["cohort-data",
                                             out_tags[out_file]])),
                             'r') as f:
                new_cdata = pickle.load(f)

            if cdata_dict[src, coh] is None:
                cdata_dict[src, coh] = new_cdata
            else:
                cdata_dict[src, coh].merge(new_cdata)

        mtypes_comp = np.greater_equal.outer(
            *([[set(auc_vals.index) for auc_vals in out_aucs]] * 2))
        super_indx = np.apply_along_axis(all, 1, mtypes_comp).argmax()

        auc_lists[src, coh] = auc_lists[src, coh].append(out_aucs[super_indx])
        pred_dfs[src, coh] = pd.concat([
            pred_dfs[src, coh], out_preds[super_indx]], sort=False)

    for src, coh, _ in use_iter.groups:
        auc_lists[src, coh].sort_index(inplace=True)
        auc_lists[src, coh] = auc_lists[src, coh].loc[
            ~auc_lists[src, coh].index.duplicated()]
        pred_dfs[src, coh] = pred_dfs[src, coh].loc[auc_lists[src, coh].index]

    for siml_metric in args.siml_metrics:
        if args.auc_cutoff < 1:
            plot_divergent_pairs(pred_dfs, phn_dicts, auc_lists,
                                 cdata_dict, args, siml_metric)


if __name__ == '__main__':
    main()

