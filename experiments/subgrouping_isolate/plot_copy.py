
from ..utilities.mutations import (pnt_mtype, shal_mtype, deep_mtype,
                                   copy_mtype, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import siml_fxs, cna_mtypes, remove_pheno_dups, get_mut_ex
from ..utilities.labels import get_cohort_label
from ..utilities.misc import choose_label_colour
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'copy')


def plot_point_similarity(pred_dfs, pheno_dicts, auc_lists,
                          cdata_dict, args, cna_lbl, siml_metric):
    fig, (pnt_ax, cpy_ax) = plt.subplots(figsize=(12, 14), nrows=2)
    cna_mtype = cna_mtypes[cna_lbl]

    copy_dict = dict()
    gn_dict = dict()
    siml_dicts = {k: {(src, coh): dict() for src, coh in auc_lists}
                  for k in ['Pnt', 'Cpy']}
    annt_lists = {k: {(src, coh): set() for src, coh in auc_lists}
                  for k in ['Pnt', 'Cpy']}

    plot_dicts = {'Pnt': dict(), 'Cpy': dict()}
    line_dicts = {'Pnt': dict(), 'Cpy': dict()}
    clr_dict = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined and subsets of point mutations...
    for (src, coh), auc_list in auc_lists.items():
        use_aucs = auc_list[
            list(remove_pheno_dups({
                mut for mut, auc_val in auc_list.iteritems()
                if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                    and get_mut_ex(mut) == args.ex_lbl)
                }, pheno_dicts[src, coh]))
            ]

        if len(use_aucs) == 0:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {tuple(mcomb.label_iter())[0] for mcomb in use_aucs.index}
        train_samps = cdata_dict[src, coh].get_train_samples()
        coh_lbl = get_cohort_label(coh)

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

        for gene, auc_vals in use_aucs.groupby(
                lambda mcomb: tuple(mcomb.label_iter())[0]):
            pnt_comb = {mcomb for mcomb in auc_vals.index
                        if all(pnt_mtype == tuple(mtype.subtype_iter())[0][1]
                               for mtype in mcomb.mtypes)}

            cpy_combs = {
                mcomb for mcomb in auc_vals.index
                if all(
                    cna_mtype.is_supertype(tuple(mtype.subtype_iter())[0][1])
                    for mtype in mcomb.mtypes
                    )
                }

            if args.ex_lbl == 'IsoShal':
                cpy_combs = {
                    mcomb for mcomb in cpy_combs
                    if all(
                        deep_mtype.is_supertype(
                            tuple(mtype.subtype_iter())[0][1])
                        for mtype in mcomb.mtypes
                        )
                    }

            if len(pnt_comb) == 1 or len(cpy_combs) > 0:
                clr_dict[gene] = None

                if args.ex_lbl == 'Iso':
                    ex_all = ExMcomb(MuType({('Gene', gene): copy_mtype}),
                                     MuType({('Gene', gene): pnt_mtype}))

                else:
                    ex_all = ExMcomb(MuType({('Gene', gene): deep_mtype}),
                                     MuType({('Gene', gene): pnt_mtype}))

                if (src, coh, gene) not in copy_dict:
                    copy_dict[src, coh, gene] = np.array(
                        cdata_dict[src, coh].train_pheno(
                            ExMcomb(MuType({('Gene', gene): pnt_mtype}),
                                    MuType({('Gene', gene): cna_mtype}))
                            )
                        )

                    gn_dict[src, coh, gene] = np.array(
                        cdata_dict[src, coh].train_pheno(ex_all))

            if len(pnt_comb) == 1:
                pnt_comb = tuple(pnt_comb)[0]
                assert not (pheno_dicts[src, coh][pnt_comb]
                            & copy_dict[src, coh, gene]).any()

                if copy_dict[src, coh, gene].sum() >= 10:
                    use_preds = pred_dfs[src, coh].loc[pnt_comb, train_samps]

                    siml_dicts['Pnt'][src, coh][pnt_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][pnt_comb]],
                            use_preds.loc[copy_dict[src, coh, gene]]
                            )

                    plt_tupl = (auc_vals[pnt_comb],
                                siml_dicts['Pnt'][src, coh][pnt_comb])

                    if (siml_dicts['Pnt'][src, coh][pnt_comb] >= 0.5
                            or gene in {'TP53', 'PIK3CA', 'GATA3'}):
                        annt_lists['Pnt'][src, coh] |= {pnt_comb}

                        plot_dicts['Pnt'][plt_tupl] = [
                            None, ("{} in {}".format(gene, coh_lbl), '')]
                        line_dicts['Pnt'][plt_tupl] = gene

                    else:
                        plot_dicts['Pnt'][plt_tupl] = [None, ('', '')]

            elif len(pnt_comb) > 1:
                raise ValueError

            for cpy_comb in cpy_combs:
                if gn_dict[src, coh, gene].sum() >= 10:
                    use_preds = pred_dfs[src, coh].loc[cpy_comb, train_samps]

                    siml_dicts['Cpy'][src, coh][cpy_comb] = siml_fxs[
                        siml_metric](
                            use_preds.loc[~all_phns[gene]],
                            use_preds.loc[pheno_dicts[src, coh][cpy_comb]],
                            use_preds.loc[gn_dict[src, coh, gene]]
                            )

                    plt_tupl = (auc_vals[cpy_comb],
                                siml_dicts['Cpy'][src, coh][cpy_comb])

                    if (siml_dicts['Cpy'][src, coh][cpy_comb] >= 0.75
                            or gene in {'TP53', 'PIK3CA', 'GATA3'}):
                        annt_lists['Cpy'][src, coh] |= {cpy_comb}

                        plot_dicts['Cpy'][plt_tupl] = [
                            None, ("{} in {}".format(gene, coh_lbl), '')]
                        line_dicts['Cpy'][plt_tupl] = gene

                    else:
                        plot_dicts['Cpy'][plt_tupl] = [None, ('', '')]

    if len(clr_dict) > 8:
        for gene in clr_dict:
            clr_dict[gene] = choose_label_colour(gene)

    else:
        use_clrs = sns.color_palette(palette='bright', n_colors=len(clr_dict))
        clr_dict = dict(zip(clr_dict, use_clrs))

    size_mult = sum(len(siml_vals) for siml_dict in siml_dicts.values()
                    for siml_vals in siml_dict.values()) ** -0.23

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]

    ymin = min(min(siml_vals.values()) for siml_dict in siml_dicts.values()
               for siml_vals in siml_dict.values() if siml_vals)
    ymax = max(max(siml_vals.values()) for siml_dict in siml_dicts.values()
               for siml_vals in siml_dict.values() if siml_vals)
    yrng = ymax - ymin
    ylims = [ymin - yrng / 23, ymax + yrng / 23]

    ylbls = {'Pnt': ("Inferred {} Similarity"
                     "\nto Point Mutations").format(cna_lbl),
             'Cpy': ("Inferred Point Mutation Similarity"
                     "\nto {} Alterations").format(cna_lbl)}

    for k, ax in zip(['Pnt', 'Cpy'], [pnt_ax, cpy_ax]):
        for (src, coh), siml_vals in siml_dicts[k].items():
            for mcomb, siml_val in siml_vals.items():
                cur_gene = tuple(mcomb.label_iter())[0]

                auc_val = auc_lists[src, coh][mcomb]
                plt_size = size_mult * np.mean(pheno_dicts[src, coh][mcomb])
                plot_dicts[k][auc_val, siml_val][0] = plt_size * 2.1

                ax.scatter(auc_val, siml_val,
                           c=[clr_dict[cur_gene]], s=1473 * plt_size,
                           alpha=0.37, edgecolor='none')
                
        ax.grid(alpha=0.47, linewidth=0.9)
        ax.plot(xlims, [0, 0],
                color='black', linewidth=1.11, linestyle='--', alpha=0.67)
        ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

        ax.set_ylabel(ylbls[k], size=21, weight='bold')
        if k == 'Cpy':
            ax.set_xlabel("Subgrouping\nClassification Accuracy",
                          size=21, weight='bold')

        ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))

        for tupl in line_dicts[k]:
            line_dicts[k][tupl] = {'c': clr_dict[line_dicts[k][tupl]]}

        for val in np.linspace(args.auc_cutoff, 0.99, 200):
            if (val, 0) not in plot_dicts[k]:
                plot_dicts[k][val, 0] = [1 / 11, ('', '')]

        lbl_pos = place_scatter_labels(plot_dicts[k], ax,
                                       plt_lims=[xlims, ylims],
                                       font_size=9, line_dict=line_dicts[k],
                                       linewidth=1.19, alpha=0.37)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}-point-similarity_{}.svg".format(
                         args.ex_lbl, cna_lbl, siml_metric, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()
    return annt_lists


def main():
    parser = argparse.ArgumentParser(
        'plot_copy',
        description="Compares copy # alterations subgroupings with a cohort."
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

    # parse command line arguments, find completed runs for this classifier
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
    #out_list = out_list[out_list.Cohort.isin({'METABRIC_LumA'})]
    use_iter = out_list.groupby(['Source', 'Cohort', 'Levels'])['File']

    out_dirs = {(src, coh): Path(base_dir, '__'.join([src, coh]))
                for src, coh, _ in use_iter.groups}
    out_tags = {fl: '__'.join(fl.parts[-1].split('__')[1:])
                for fl in out_list.File}
    pred_tag = "out-pred_{}".format(args.ex_lbl)

    phn_dicts = {(src, coh): dict() for src, coh, _ in use_iter.groups}
    cdata_dict = {(src, coh): None for src, coh, _ in use_iter.groups}

    auc_lists = {(src, coh): pd.Series(dtype='float')
                 for src, coh, _ in use_iter.groups}
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
        super_comp = np.apply_along_axis(all, 1, mtypes_comp)

        # if there is not a subgrouping set that contains all the others,
        # concatenate the output of all sets...
        if not super_comp.any():
            auc_lists[src, coh] = auc_lists[src, coh].append(
                pd.concat(out_aucs, sort=False))
            pred_dfs[src, coh] = pd.concat(
                [pred_dfs[src, coh], *out_preds], sort=False)

        # ...otherwise, use the "superset"
        else:
            super_indx = super_comp.argmax()

            auc_lists[src, coh] = auc_lists[src, coh].append(
                out_aucs[super_indx])
            pred_dfs[src, coh] = pd.concat(
                [pred_dfs[src, coh], out_preds[super_indx]], sort=False)

    # filter out duplicate subgroupings due to overlapping search criteria
    for src, coh, _ in use_iter.groups:
        auc_lists[src, coh].sort_index(inplace=True)
        pred_dfs[src, coh].sort_index(inplace=True)
        assert (auc_lists[src, coh].index == pred_dfs[src, coh].index).all()

        auc_lists[src, coh] = auc_lists[src, coh].loc[
            ~auc_lists[src, coh].index.duplicated()]
        pred_dfs[src, coh] = pred_dfs[src, coh].loc[
            ~pred_dfs[src, coh].index.duplicated()]

    for siml_metric in args.siml_metrics:
        if args.auc_cutoff < 1:
            for cna_lbl in ['Gain', 'Loss']:
                annt_types = plot_point_similarity(
                    pred_dfs, phn_dicts, auc_lists,
                    cdata_dict, args, cna_lbl, siml_metric
                    )


if __name__ == '__main__':
    main()

