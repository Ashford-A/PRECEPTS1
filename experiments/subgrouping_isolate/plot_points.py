
from ..utilities.mutations import (pnt_mtype, copy_mtype, deep_mtype,
                                   shal_mtype, gains_mtype, dels_mtype,
                                   ExMcomb)
from dryadic.features.mutations import MuType

from ..subgrouping_isolate import base_dir, train_cohorts
from .utils import (load_cohorts_data, siml_fxs, remove_pheno_dups,
                    get_mut_ex, get_mcomb_lbl, choose_subtype_colour)
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.labels import get_cohort_label
from ..utilities.colour_maps import variant_clrs
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import multiprocessing as mp

from itertools import product
from itertools import permutations as permt
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

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
    test = dict()

    # for each dataset, find the subgroupings meeting the minimum task AUC
    # that are exclusively defined and subsets of point mutations...
    for (src, coh), auc_list in auc_lists.items():
        test[src, coh] = dict()
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

            if cmp_phn.sum() >= 1:
                siml_dicts[src, coh][mcomb] = siml_fxs[siml_metric](
                    use_preds.loc[~all_phns[cur_gene]],
                    use_preds.loc[pheno_dicts[src, coh][mcomb]],
                    use_preds.loc[cmp_phn]
                    )
                test[src, coh][mcomb] = sum(
                    pheno_dicts[src, coh][mcomb] & cmp_phns[cur_gene]['Mult'])

    plt_df = pd.DataFrame(
        {'Siml': pd.DataFrame.from_records(siml_dicts).stack()})
    plt_df['AUC'] = [auc_lists[src, coh][mcomb]
                     for mcomb, (src, coh) in plt_df.index]
    """
    plt_df['Phn'] = [test[src, coh][mcomb] 
                     for mcomb, (src, coh) in plt_df.index]
    import ipdb; ipdb.set_trace()
    """

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


def plot_overlap_synergy(pred_dfs, pheno_dicts, auc_lists,
                         cdata_dict, args, siml_metric):
    fig, ax = plt.subplots(figsize=(11, 11))
    siml_dfs = dict()
    annt_lists = dict()

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

        use_scombs = {mcomb for mcomb in use_combs if len(mcomb.mtypes) == 1}
        use_dcombs = {mcomb for mcomb in use_combs if len(mcomb.mtypes) > 1}

        # find all pairs of subgroupings from the same gene that are disjoint
        # either by construction or by phenotype
        use_pairs = {(scomb, dcomb)
                     for scomb, dcomb in product(use_scombs, use_dcombs)
                     if (get_label(scomb) == get_label(dcomb)
                         and not (pheno_dicts[src, coh][scomb]
                                  & pheno_dicts[src, coh][dcomb]).any())}

        # skip this dataset for plotting if we cannot find any such pairs
        if not use_pairs:
            continue

        base_mtree = tuple(cdata_dict[src, coh].mtrees.values())[0]
        use_genes = {get_label(mcomb)
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

        wt_vals = {mcomb: use_preds.loc[mcomb][~all_phns[get_label(mcomb)]]
                   for mcomb in pair_combs}
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
            chunk_size = int(len(map_args) / (11 * args.cores)) + 1
        elif siml_metric == 'ks':
            chunk_size = int(len(map_args) / (3 * args.cores)) + 1

        pool = mp.Pool(args.cores)
        siml_list = pool.starmap(siml_fxs[siml_metric], map_args, chunk_size)
        pool.close()

        siml_dfs[src, coh] = pd.concat({(src, coh): pd.DataFrame(
            dict(zip(use_pairs, zip(siml_list[::2], siml_list[1::2])))
            ).transpose()})
        siml_dfs[src, coh].index.set_names(
            ['Source', 'Cohort', 'Mcomb1', 'Mcomb2'], inplace=True)

        annt_lists[src, coh] = dict()
        for gene, siml_df in siml_dfs[src, coh].groupby(
                lambda x: get_label(x[2])):
            annt_mat = pd.DataFrame.from_dict({
                (mcomb1, mcomb2): {'Siml1': siml1, 'Siml2': siml2,
                                   'AUC1': auc_list[mcomb1],
                                   'AUC2': auc_list[mcomb2]}
                for (_, _, mcomb1, mcomb2), (siml1, siml2)
                in siml_df.iterrows()
                }, orient='index')

            annt_lists[src, coh][gene] = (
                (annt_mat.AUC1 - args.auc_cutoff)
                * (annt_mat.AUC2 - args.auc_cutoff)
                ).sort_values().index[-10:].tolist()

    plt_df = pd.concat(siml_dfs.values())
    gene_counts = plt_df.groupby(
        lambda x: (x[0], x[1], get_label(x[2]), get_label(x[3]))).count()

    plt_genes = {gene for _, _, gene, _ in gene_counts.index}
    clr_dict = {gene: choose_label_colour(gene) for gene in plt_genes}
    size_mult = plt_df.shape[0] ** -0.23

    lgnd_lbls = ["{} in {}{:>7} pairs".format(gene, get_cohort_label(coh),
                                              count[0])
                 for (_, coh, gene, _), count in gene_counts.iterrows()]

    lgnd_mrks = [Line2D([], [], marker='o', linestyle='None',
                        markersize=25, alpha=0.61,
                        markerfacecolor=clr_dict[gene],
                        markeredgecolor='none')
                 for _, _, gene, _ in gene_counts.index]

    for (src, coh, scomb, dcomb), siml_vals in plt_df.iterrows():
        cur_gene = get_label(scomb)

        plt_size = (np.mean(pheno_dicts[src, coh][scomb])
                    * np.mean(pheno_dicts[src, coh][dcomb])) ** 0.5
        plt_size *= size_mult

        ax.scatter(*siml_vals, s=2307 * plt_size,
                   c=[clr_dict[cur_gene]], alpha=0.23, edgecolor='none')

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))

    plt_min, plt_max = np.percentile(plt_df.values, q=[0, 100])
    plt_rng = plt_max - plt_min
    plt_lims = [plt_min - plt_rng / 31, plt_max + plt_rng / 31]

    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=1.3, linestyle='--', alpha=0.41)

    ax.set_xlabel("Singletons' Similarity to Overlaps",
                  size=25, weight='bold')
    ax.set_ylabel("Overlaps' Similarity to Singletons",
                  size=25, weight='bold')

    ax.legend(lgnd_mrks, lgnd_lbls, bbox_to_anchor=(0.5, -0.083),
              frameon=False, fontsize=21, ncol=1, loc=9, handletextpad=0.3)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(os.path.join(plot_dir,
                             "{}_{}-overlap-synergy_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()
    return annt_lists


def plot_overlap_aucs(pheno_dict, auc_list,
                      cdata, data_tag, args, siml_metric, use_gene):
    fig, ax = plt.subplots(figsize=(11, 8))

    use_combs = remove_pheno_dups({
        mut for mut, auc_val in auc_list.iteritems()
        if (isinstance(mut, ExMcomb) and auc_val >= args.auc_cutoff
            and get_mut_ex(mut) == args.ex_lbl
            and len(mut.mtypes) == 1 and get_label(mut) == use_gene
            and all(pnt_mtype.is_supertype(get_subtype(mtype))
                    for mtype in mut.mtypes))
        }, pheno_dict)

    train_samps = cdata.get_train_samples()
    use_mtree = cdata.mtrees['Gene', 'Scale', 'Copy',
                             'Exon', 'Position', 'HGVSp']
    gene_tree = use_mtree[use_gene]['Point']

    if args.ex_lbl == 'Iso':
        gene_cpy = MuType({('Gene', use_gene): copy_mtype})
    else:
        gene_cpy = MuType({('Gene', use_gene): deep_mtype})

    cpy_samps = gene_cpy.get_samples(use_mtree)
    samp_counts = {samp: 0 for samp in (gene_tree.get_samples() - cpy_samps)}

    for subk in MuType(gene_tree.allkey()).leaves():
        for samp in MuType(subk).get_samples(gene_tree):
            if samp in samp_counts:
                samp_counts[samp] += 1

    for samp in train_samps:
        if samp not in samp_counts:
            samp_counts[samp] = 0

    plt_df = pd.DataFrame({'AUC': auc_list[use_combs]})
    plt_df['Muts'] = [np.mean([samp_counts[samp]
                               for samp, phn in zip(train_samps,
                                                    pheno_dict[mcomb])
                               if phn]) for mcomb in plt_df.index]

    plt_df['Size'] = [np.mean(pheno_dict[mcomb]) for mcomb in plt_df.index]
    plt_clr = choose_label_colour(use_gene)
    size_mult = plt_df.shape[0] ** -0.23

    xlims = [args.auc_cutoff - (1 - args.auc_cutoff) / 47,
             1 + (1 - args.auc_cutoff) / 277]
    ymin, ymax = [1, plt_df.Muts.max()]
    yrng = ymax - ymin
    ylims = [ymin - yrng / 23, ymax + yrng / 23]

    for mcomb, (auc_val, mut_val, size_val) in plt_df.iterrows():
        ax.scatter(auc_val, mut_val, s=3751 * size_mult * size_val,
                   c=[plt_clr], alpha=0.31, edgecolor='none')

    ax.grid(alpha=0.47, linewidth=0.9)
    ax.plot([1, 1], ylims, color='black', linewidth=1.7, alpha=0.83)
    ax.plot(xlims, [1, 1], color='black', linewidth=1.3, alpha=0.83)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7, steps=[1, 2, 5]))
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_xlabel("Subgrouping Classification Accuracy",
                  size=21, weight='bold')
    ax.set_ylabel("Average # of {} Point Mutations"
                  "\nper Subgrouping Sample".format(use_gene),
                  size=21, weight='bold')

    ax.text(0.97, 0.07, get_cohort_label(data_tag.split('__')[1]),
            size=17, style='italic', ha='right', va='bottom',
            transform=ax.transAxes)

    plt.savefig(os.path.join(plot_dir, data_tag, use_gene,
                             "{}_{}-overlap-aucs_{}.svg".format(
                                 args.ex_lbl, siml_metric, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_synergy_pair(pred_vals, auc_vals, pheno_dict, cdata,
                      data_tag, args, siml_metric, file_lbl=None):
    fig, ((mcomb2_ax, sctr_ax), (crnr_ax, mcomb1_ax)) = plt.subplots(
        figsize=(13, 12), nrows=2, ncols=2,
        gridspec_kw=dict(height_ratios=[3.1, 1], width_ratios=[1, 3.1])
        )

    plt_mcomb1, plt_mcomb2 = pred_vals.index
    assert len(plt_mcomb1.mtypes) == 1 and len(plt_mcomb2.mtypes) == 2
    mcomb1_vals, mcomb2_vals = pred_vals[cdata.get_train_samples()].values

    use_gene = {get_label(mtype) for mcomb in [plt_mcomb1, plt_mcomb2]
                for mtype in mcomb.mtypes}
    assert len(use_gene) == 1
    use_gene = tuple(use_gene)[0]

    train_samps = cdata.get_train_samples()
    gene_tree = cdata.mtrees['Gene', 'Scale', 'Copy',
                             'Exon', 'Position', 'HGVSp'][use_gene]

    all_mtype = MuType({('Gene', use_gene): gene_tree.allkey()})
    if args.ex_lbl == 'IsoShal':
        all_mtype -= MuType({('Gene', use_gene): shal_mtype})

    x_min, x_max = np.percentile(mcomb1_vals, q=[0, 100])
    y_min, y_max = np.percentile(mcomb2_vals, q=[0, 100])
    x_rng, y_rng = x_max - x_min, y_max - y_min
    xlims = x_min - x_rng / 31, x_max + x_rng / 11
    ylims = y_min - y_rng / 31, y_max + y_rng / 11

    if args.ex_lbl == 'Iso':
        cpy_samps = copy_mtype.get_samples(gene_tree)
    else:
        cpy_samps = deep_mtype.get_samples(gene_tree)

    samp_counts = {samp: 0 for samp in (gene_tree.get_samples() - cpy_samps)}
    for subk in MuType(gene_tree['Point'].allkey()).leaves():
        for samp in MuType(subk).get_samples(gene_tree['Point']):
            if samp in samp_counts:
                samp_counts[samp] += 1

    for samp in train_samps:
        if samp not in samp_counts:
            samp_counts[samp] = 0

    sngl_phn = np.array([samp_counts[samp] == 1 for samp in train_samps])
    mult_phn = np.array([samp_counts[samp] > 1 for samp in train_samps])

    use_phns = {
        'WT': ~np.array(cdata.train_pheno(all_mtype)),
        'Mut1': pheno_dict[plt_mcomb1], 'Mut2': pheno_dict[plt_mcomb2],
        'Sngl': sngl_phn & ~pheno_dict[plt_mcomb1] & ~pheno_dict[plt_mcomb2],
        'Mult': mult_phn & ~pheno_dict[plt_mcomb2] & ~pheno_dict[plt_mcomb1],
        'Gain': np.array(cdata.train_pheno(
            ExMcomb(all_mtype, MuType({('Gene', use_gene): gains_mtype})))),
        'Del': np.array(cdata.train_pheno(
            ExMcomb(all_mtype, MuType({('Gene', use_gene): dels_mtype}))))
        }

    use_fclrs = {'WT': variant_clrs['WT'], 'Mut1': '#080097',
                 'Mut2': '#00847F', 'Sngl': 'none', 'Mult': 'none',
                 'Gain': 'none', 'Del': 'none'}

    use_eclrs = {'WT': 'none', 'Mut1': 'none', 'Mut2': 'none',
                 'Sngl': '#080097', 'Mult': '#00847F',
                 'Gain': choose_subtype_colour(gains_mtype),
                 'Del': choose_subtype_colour(dels_mtype)}

    use_sizes = {'WT': 5, 'Mut1': 7, 'Mut2': 7,
                 'Sngl': 5, 'Mult': 5, 'Gain': 5, 'Del': 5}
    use_alphas = {'WT': 0.21, 'Mut1': 0.31, 'Mut2': 0.31,
                  'Sngl': 0.35, 'Mult': 0.35, 'Gain': 0.35, 'Del': 0.35}

    for lbl, phn in use_phns.items():
        sctr_ax.plot(mcomb1_vals[phn], mcomb2_vals[phn],
                     marker='o', linewidth=0,
                     markersize=use_sizes[lbl], alpha=use_alphas[lbl],
                     mfc=use_fclrs[lbl], mec=use_eclrs[lbl], mew=1.9)

    mcomb_lbls = [get_mcomb_lbl(mcomb, pnt_link='\nor ')
                  for mcomb in [plt_mcomb1, plt_mcomb2]]
    subg_lbls = ["only {} mutation\nis {}".format(use_gene, mcomb_lbl)
                 for mcomb_lbl in mcomb_lbls]

    sctr_ax.text(0.98, 0.03, subg_lbls[0], size=16, c=use_fclrs['Mut1'],
                 ha='right', va='bottom', transform=sctr_ax.transAxes)
    sctr_ax.text(0.03, 0.98, subg_lbls[1], size=16, c=use_fclrs['Mut2'],
                 ha='left', va='top', transform=sctr_ax.transAxes)

    sctr_ax.text(0.97, 0.98, get_cohort_label(data_tag.split('__')[1]),
                 size=20, style='italic', ha='right', va='top',
                 transform=sctr_ax.transAxes)

    sctr_ax.set_xticklabels([])
    sctr_ax.set_yticklabels([])

    for ax in sctr_ax, mcomb1_ax, mcomb2_ax:
        ax.grid(alpha=0.47, linewidth=0.9)

    use_preds = pd.DataFrame({
        'Mut1': mcomb1_vals, 'Mut2': mcomb2_vals, 'Phn': 'WT'})
    for lbl in ['Mut1', 'Mut2', 'Sngl', 'Mult', 'Gain', 'Del']:
        use_preds.loc[use_phns[lbl], 'Phn'] = lbl

    sns.violinplot(data=use_preds, y='Phn', x='Mut1', ax=mcomb1_ax,
                   order=['WT', 'Mut1', 'Mut2',
                          'Sngl', 'Mult', 'Gain', 'Del'],
                   palette=use_fclrs, orient='h', linewidth=0, cut=0)

    sns.violinplot(data=use_preds, x='Phn', y='Mut2', ax=mcomb2_ax,
                   order=['WT', 'Mut2', 'Mut1',
                          'Mult', 'Sngl', 'Gain', 'Del'],
                   palette=use_fclrs, orient='v', linewidth=0, cut=0)

    for mcomb_ax, lbls in zip([mcomb1_ax, mcomb2_ax],
                              [['Sngl', 'Mult', 'Gain', 'Del'],
                               ['Mult', 'Sngl', 'Gain', 'Del']]):
        for i in range(3):
            mcomb_ax.get_children()[i * 2].set_alpha(0.61)
            mcomb_ax.get_children()[i * 2].set_linewidth(0)

        i = 0
        for lbl in lbls:
            if use_phns[lbl].sum() > 1:
                i += 1

                mcomb_ax.get_children()[4 + i * 2].set_edgecolor(
                    use_eclrs[lbl])
                mcomb_ax.get_children()[4 + i * 2].set_facecolor('white')
                mcomb_ax.get_children()[4 + i * 2].set_linewidth(2.9)
                mcomb_ax.get_children()[4 + i * 2].set_alpha(0.71)

    mcomb1_ax.set_xlabel("Subgrouping Isolation Task 1\nPredicted Scores",
                         size=23, weight='semibold')
    mcomb1_ax.yaxis.label.set_visible(False)

    mcomb1_ax.set_yticklabels(
        ["Wild-Type", "Subg1", "Subg2",
         "Other {}\nSingletons".format(use_gene),
         "Other {}\nOverlaps".format(use_gene), 'Gains', 'Dels'],
        size=11
        )

    mcomb2_ax.set_ylabel("Subgrouping Isolation Task 2\nPredicted Scores",
                         size=23, weight='semibold')
    mcomb2_ax.xaxis.label.set_visible(False)

    mcomb2_ax.set_xticklabels(
        ['WT', 'Subg2', 'Subg1', 'Overlps', 'Singls', 'Gains', 'Dels'],
        rotation=31, size=11, ha='right'
        )

    for i, lbl in enumerate(['WT', 'Mut1', 'Mut2',
                             'Sngl', 'Mult', 'Gain', 'Del']):
        mcomb1_ax.text(1, 0.9 - i / 7, "n={}".format(use_phns[lbl].sum()),
                       size=12, ha='left', transform=mcomb1_ax.transAxes,
                       clip_on=False)

    for i, lbl in enumerate(['WT', 'Mut2', 'Mut1',
                             'Mult', 'Sngl', 'Gain', 'Del']):
        mcomb2_ax.text(0.04 + i / 7, 1, "n={}".format(use_phns[lbl].sum()),
                       size=12, rotation=31, ha='left',
                       transform=mcomb2_ax.transAxes, clip_on=False)

    crnr_ax.text(0.77, 0.76, "(AUC1: {:.3f})".format(auc_vals[plt_mcomb1]),
                 size=10, ha='right', transform=crnr_ax.transAxes,
                 clip_on=False)
    mcomb2_ax.text(-0.01, -0.14,
                   "(AUC2: {:.3f})".format(auc_vals[plt_mcomb2]),
                   size=10, rotation=31, ha='right',
                   transform=mcomb2_ax.transAxes, clip_on=False)

    wt_vals = use_preds.loc[use_preds.Phn == 'WT', ['Mut1', 'Mut2']]
    siml_vals = {
        'mut': {
            subg: format(siml_fxs[siml_metric](
                wt_vals[subg], use_preds.loc[use_preds.Phn == subg, subg],
                use_preds.loc[use_preds.Phn == oth_subg, subg]
            ), '.3f')
            for subg, oth_subg in permt(['Mut1', 'Mut2'])
            }
        }

    for lbl in ['Sngl', 'Mult', 'Gain', 'Del']:
        if use_phns[lbl].sum() >= 10:
            siml_vals[lbl] = {
                subg: format(siml_fxs[siml_metric](
                    wt_vals[subg], use_preds.loc[use_preds.Phn == subg, subg],
                    use_preds.loc[use_preds.Phn == lbl, subg]
                ), '.3f')
                for subg in ['Mut1', 'Mut2']
            }

        else:
            siml_vals[lbl] = {'Mut1': "    n/a", 'Mut2': "    n/a"}

    for i, lbl in enumerate(['mut', 'Sngl', 'Mult', 'Gain', 'Del']):
        crnr_ax.text(0.77, 0.62 - i / 7,
                     "(Siml1: {})".format(siml_vals[lbl]['Mut1']),
                     size=10, ha='right', transform=crnr_ax.transAxes,
                     clip_on=False)

    for i, lbl in enumerate(['mut', 'Mult', 'Sngl', 'Gain', 'Del']):
        mcomb2_ax.text(0.133 + i / 7, -0.14,
                       "(Siml2: {})".format(siml_vals[lbl]['Mut2']),
                       size=10, rotation=31, ha='right',
                       transform=mcomb2_ax.transAxes, clip_on=False)

    crnr_ax.axis('off')
    sctr_ax.set_xlim(xlims)
    sctr_ax.set_ylim(ylims)
    mcomb1_ax.set_xlim(xlims)
    mcomb2_ax.set_ylim(ylims)

    if file_lbl is None:
        file_lbl = '__'.join(['__'.join([mtype.get_filelabel()[:30]
                                         for mtype in sorted(mcomb.mtypes)])
                              for mcomb in [plt_mcomb1, plt_mcomb2]])

    fig.tight_layout(w_pad=-2, h_pad=-0.3)
    plt.savefig(os.path.join(
        plot_dir, data_tag, use_gene,
        "{}_{}-synergy-scores_{}__{}.svg".format(args.ex_lbl, siml_metric,
                                                 args.classif, file_lbl)
        ), bbox_inches='tight', format='svg')

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

            annt_pairs = plot_overlap_synergy(pred_dfs, phn_dicts, auc_lists,
                                              cdata_dict, args, siml_metric)

            for (src, coh), gene_dict in annt_pairs.items():
                for gene, pair_list in gene_dict.items():
                    os.makedirs(os.path.join(plot_dir,
                                             '__'.join([src, coh]), gene),
                                exist_ok=True)

                    plot_overlap_aucs(
                        phn_dicts[src, coh], auc_lists[src, coh],
                        cdata_dict[src, coh], '__'.join([src, coh]),
                        args, siml_metric, gene
                        )

                    for i, (mcomb1, mcomb2) in enumerate(sorted(pair_list)):
                        plot_synergy_pair(
                            pred_dfs[src, coh].loc[[mcomb1, mcomb2]],
                            auc_lists[src, coh], phn_dicts[src, coh],
                            cdata_dict[src, coh], '__'.join([src, coh]),
                            args, siml_metric, file_lbl=i + 1
                            )


if __name__ == '__main__':
    main()

