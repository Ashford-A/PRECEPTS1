
from ..utilities.mutations import (pnt_mtype, copy_mtype,
                                   deep_mtype, shal_mtype, ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import load_cohorts_data, siml_fxs, remove_pheno_dups, get_mut_ex
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path

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
plot_dir = os.path.join(base_dir, 'plots', 'points')


def plot_overlap_divergence(pred_dfs, pheno_dicts, auc_lists,
                            cdata_dict, args, siml_metric):
    fig, (sngl_ax, mult_ax) = plt.subplots(figsize=(12, 14), nrows=2)

    siml_dicts = {(src, coh): dict() for src, coh in auc_lists}
    gn_dict = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined and subsets of point mutations...
    for (src, coh), auc_list in auc_lists.items():
        use_combs = remove_pheno_dups({
            mut for mut, auc_val in auc_list.iteritems()
            if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
                and get_mut_ex(mut) == args.ex_lbl
                and all(pnt_mtype.is_supertype(get_subtype(mtype))
                        for mtype in mut.mtypes))
            }, pheno_dicts[src, coh])

        # skip this dataset for plotting if we cannot find any such pairs
        if not use_combs:
            continue

        # get sample order used in the cohort and a breakdown of mutations
        # in which each individual mutation can be uniquely identified
        train_samps = cdata_dict[src, coh].get_train_samples()
        use_mtree = cdata_dict[src, coh].mtrees['Gene', 'Scale', 'Copy',
                                                'Exon', 'Position', 'HGVSp']
        use_genes = {get_label(mcomb) for mcomb in use_combs}
        cmp_phns = {gene: {'Sngl': None, 'Mult': None} for gene in use_genes}

        # get the samples carrying a single point mutation or multiple
        # mutations of each gene with at least one mutation in the cohort
        for gene in use_genes:
            gene_tree = use_mtree[gene]['Point']

            if args.ex_lbl == 'Iso':
                gene_cpy = MuType({('Gene', gene): copy_mtype})
            else:
                gene_cpy = MuType({('Gene', gene): deep_mtype})

            cpy_samps = gene_cpy.get_samples(use_mtree)
            samp_counts = {samp: 0 for samp in (gene_tree.get_samples()
                                                - cpy_samps)}

            for subk in MuType(gene_tree.allkey()).leaves():
                for samp in MuType(subk).get_samples(gene_tree):
                    if samp in samp_counts:
                        samp_counts[samp] += 1

            for samp in train_samps:
                if samp not in samp_counts:
                    samp_counts[samp] = 0

            cmp_phns[gene]['Sngl'] = np.array([samp_counts[samp] == 1
                                               for samp in train_samps])
            cmp_phns[gene]['Mult'] = np.array([samp_counts[samp] > 1
                                               for samp in train_samps])

        all_mtypes = {
            gene: MuType({('Gene', gene): use_mtree[gene].allkey()})
            for gene in use_genes
            }

        if args.ex_lbl == 'IsoShal':
            for gene in use_genes:
                all_mtypes[gene] -= MuType({('Gene', gene): shal_mtype})

        all_phns = {
            gene: np.array(cdata_dict[src, coh].train_pheno(all_mtype))
            for gene, all_mtype in all_mtypes.items()
            }

        # for each subgrouping, find the subset of point mutations that
        # defines it, the gene it's associated with, and its task predictions
        for mcomb in use_combs:
            cur_gene = get_label(mcomb)
            use_preds = pred_dfs[src, coh].loc[mcomb, train_samps]

            # get the samples that carry any point mutation of this gene
            if (src, coh, cur_gene) not in gn_dict:
                gn_dict[src, coh, cur_gene] = np.array(
                    cdata_dict[src, coh].train_pheno(
                        MuType({('Gene', cur_gene): pnt_mtype}))
                    )

            # find the samples carrying one or multiple point mutations of
            # this gene not belonging to this subgrouping
            cmp_phn = ~pheno_dicts[src, coh][mcomb]
            if len(mcomb.mtypes) == 1:
                cmp_phn &= cmp_phns[cur_gene]['Mult']
            else:
                cmp_phn &= cmp_phns[cur_gene]['Sngl']

            if cmp_phn.sum() >= 10:
                siml_dicts[src, coh][mcomb] = siml_fxs[siml_metric](
                    use_preds.loc[~all_phns[cur_gene]],
                    use_preds.loc[pheno_dicts[src, coh][mcomb]],
                    use_preds.loc[cmp_phn]
                    )

    plt_df = pd.DataFrame(
        {'Siml': pd.DataFrame.from_records(siml_dicts).stack()})
    plt_df['AUC'] = [auc_lists[src, coh][mcomb]
                     for mcomb, (src, coh) in plt_df.index]

    gene_means = plt_df.groupby(
        lambda x: (get_label(x[0]), len(x[0].mtypes))).mean()
    clr_dict = {gene: choose_label_colour(gene)
                for gene, _ in gene_means.index}
    size_mult = plt_df.groupby(
        lambda x: len(x[0].mtypes)).Siml.count().max() ** -0.23

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]
    ymin, ymax = plt_df.Siml.quantile(q=[0, 1])
    yrng = ymax - ymin
    ylims = [ymin - yrng / 23, ymax + yrng / 23]

    plot_dicts = {mcomb_i: {(auc_val, siml_val): [0.0001, (gene, '')]
                            for (gene, mcomb_indx), (siml_val, auc_val)
                            in gene_means.iterrows() if mcomb_indx == mcomb_i}
                  for mcomb_i in [1, 2]}

    for (mcomb, (src, coh)), (siml_val, auc_val) in plt_df.iterrows():
        cur_gene = get_label(mcomb)

        plt_size = size_mult * np.mean(pheno_dicts[src, coh][mcomb])
        plot_dicts[(len(mcomb.mtypes) == 2) + 1][auc_val, siml_val] = [
            0.19 * plt_size, ('', '')]

        if len(mcomb.mtypes) == 1:
            use_ax = sngl_ax
        else:
            use_ax = mult_ax

        use_ax.scatter(auc_val, siml_val, s=3751 * plt_size,
                       c=[clr_dict[cur_gene]], alpha=0.25, edgecolor='none')

    for ax, mcomb_i in zip([sngl_ax, mult_ax], [1, 2]):
        ax.grid(alpha=0.47, linewidth=0.9)
        ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)

        for yval in [0, 1]:
            if xlims[0] < yval < xlims[1]:
                ax.plot(xlims, [yval, yval], color='black',
                        linewidth=1.11, linestyle='--', alpha=0.67)

                for k in np.linspace(args.auc_cutoff, 0.99, 200):
                    if (k, yval) not in plot_dicts[mcomb_i]:
                        plot_dicts[mcomb_i][k, yval] = [1 / 703, ('', '')]

        line_dict = {k: {'c': clr_dict[v[1][0]]}
                     for k, v in plot_dicts[mcomb_i].items() if v[1][0]}
        font_dict = {k: {'c': v['c'], 'weight': 'bold'}
                     for k, v in line_dict.items()}

        lbl_pos = place_scatter_labels(plot_dicts[mcomb_i], ax,
                                       plt_lims=[xlims, ylims],
                                       line_dict=line_dict,
                                       font_dict=font_dict, font_size=19)

        ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    mult_ax.set_xlabel("Subgrouping Classification Accuracy",
                       size=21, weight='bold')
    sngl_ax.set_ylabel("Overlaps' Similarity to Singletons",
                       size=21, weight='bold')
    mult_ax.set_ylabel("Singletons' Similarity to Overlaps",
                       size=21, weight='bold')

    plt.savefig(os.path.join(plot_dir,
                             "{}_{}-overlap-divergence_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_point',
        description="Compares combinations of point mutations across cohorts."
        )

    parser.add_argument('classif', help="a mutation classifier")
    parser.add_argument('ex_lbl', help="a classification mode",
                        choices={'Iso', 'IsoShal'})

    parser.add_argument('--auc_cutoff', '-a', type=float, default=0.8)
    parser.add_argument('--siml_metrics', '-s', nargs='+',
                        default=['ks'], choices={'mean', 'ks'})

    parser.add_argument('--cores', '-c', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data_cache')
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
            lambda outs: ('Consequence__Exon' in set(outs.Levels)
                          and 'Exon__Position__HGVSp' in set(outs.Levels))
            )

    if len(out_list) == 0:
        raise ValueError("No completed experiments found for this "
                         "combination of parameters!")

    out_list = out_list[out_list.Cohort.isin(train_cohorts)]
    pred_dfs, phn_dicts, auc_lists, cdata_dict = load_cohorts_data(
        out_list, args.ex_lbl, args.data_cache)

    for siml_metric in args.siml_metrics:
        if args.auc_cutoff < 1:
            plot_overlap_divergence(pred_dfs, phn_dicts, auc_lists,
                                    cdata_dict, args, siml_metric)


if __name__ == '__main__':
    main()

