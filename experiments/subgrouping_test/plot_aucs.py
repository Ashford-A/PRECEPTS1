
from ..utilities.mutations import (
    pnt_mtype, copy_mtype, dup_mtype, loss_mtype, RandomType)
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.metrics import calc_conf
from ..utilities.misc import choose_label_colour
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')


def add_scatterpie_legend(ax, plot_dict, plt_min, args):
    lgnd_x, lgnd_y = 0.77 + 0.23 * plt_min, 0.07 + 0.93 * plt_min

    lgnd_sz = (1 - plt_min) / 8.3
    plot_dict[lgnd_x, lgnd_y] = lgnd_sz, ('', '')
    lgnd_clr = choose_label_colour('GENE')
    lgnd_bbox = lgnd_x - lgnd_sz / 2, lgnd_y - lgnd_sz / 2, lgnd_sz, lgnd_sz

    pie_ax = inset_axes(ax, width='100%', height='100%',
                        bbox_to_anchor=lgnd_bbox, bbox_transform=ax.transData,
                        loc=10, axes_kwargs=dict(aspect='equal'), borderpad=0)

    pie_ax.pie(x=[0.43, 0.57], explode=[0.19, 0], startangle=90,
               colors=[lgnd_clr + (0.77, ), lgnd_clr + (0.29, )])

    coh_lbl = "% of {} samples with\npoint mutations in gene".format(
        get_cohort_label(args.cohort))
    ax.text(lgnd_x - lgnd_sz / 103, lgnd_y + lgnd_sz * 0.71, coh_lbl,
            size=15, style='italic', ha='center', va='bottom')

    ax.text(lgnd_x - lgnd_sz * 0.7, lgnd_y - lgnd_sz * 0.13,
            "% of gene's point-mutated\nsamples with best subgrouping",
            size=15, style='italic', ha='right', va='center')

    ax.plot([lgnd_x - lgnd_sz / 1.87, lgnd_x - lgnd_sz / 23],
            [lgnd_y + lgnd_sz / 5.3, lgnd_y + lgnd_sz * 0.67],
            c='black', linewidth=1.1)
    ax.plot([lgnd_x - lgnd_sz / 23, lgnd_x + lgnd_sz / 2.21],
            [lgnd_y + lgnd_sz * 0.67, lgnd_y + lgnd_sz / 5.3],
            c='black', linewidth=1.1)

    ax.plot([lgnd_x - lgnd_sz * 0.63, lgnd_x - lgnd_sz / 3.1],
            [lgnd_y - lgnd_sz / 7.3, lgnd_y - lgnd_sz / 18.3],
            c='black', linewidth=1.1)

    return ax, plot_dict


def plot_random_comparison(auc_vals, pheno_dict, args):
    fig, (viol_ax, sctr_ax) = plt.subplots(
        figsize=(11, 7), nrows=1, ncols=2,
        gridspec_kw=dict(width_ratios=[1, 1.51])
        )

    mtype_genes = pd.Series([
        tuple(mtype.label_iter())[0] for mtype in auc_vals.index
        if (not isinstance(mtype, RandomType)
            and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty())
        ])

    sbgp_genes = mtype_genes.value_counts()[
        mtype_genes.value_counts() > 1].index
    lbl_order = ['Random', 'Point w/o Sub', 'Point w/ Sub', 'Subgroupings']

    gene_stat = pd.Series({
        mtype: ('Random' if (isinstance(mtype, RandomType)
                             and mtype.base_mtype is None)
                else 'RandomGene' if isinstance(mtype, RandomType)
                else 'Point w/ Sub'
                if (tuple(mtype.subtype_iter())[0][1] == pnt_mtype
                    and tuple(mtype.label_iter())[0] in sbgp_genes)
                else 'Point w/o Sub'
                if (tuple(mtype.subtype_iter())[0][1] == pnt_mtype
                    and not tuple(mtype.label_iter())[0] in sbgp_genes)
                else 'Copy' if not (tuple(mtype.subtype_iter())[0][1]
                                    & copy_mtype).is_empty()
                else 'Subgroupings')
        for mtype in auc_vals.index
        })

    use_aucs = auc_vals[~gene_stat.isin(['Copy', 'RandomGene'])]
    gene_stat = gene_stat[~gene_stat.isin(['Copy', 'RandomGene'])]

    sns.violinplot(x=gene_stat, y=use_aucs, ax=viol_ax, order=lbl_order,
                   palette=['0.61', *[variant_clrs['Point']] * 3],
                   cut=0, linewidth=0, width=0.93)

    viol_ax.set_xlabel('')
    viol_ax.set_ylabel('AUC', size=23, weight='semibold')
    viol_ax.set_xticklabels(lbl_order, rotation=37, ha='right', size=18)

    viol_ax.get_children()[2].set_linewidth(3.1)
    viol_ax.get_children()[4].set_linewidth(1.7)
    viol_ax.get_children()[2].set_facecolor('white')
    viol_ax.get_children()[2].set_edgecolor(variant_clrs['Point'])
    viol_ylims = viol_ax.get_ylim()

    if 'Point w/o Sub' in gene_stat:
        viol_ax.get_children()[4].set_edgecolor(variant_clrs['Point'])

    for i, lbl in enumerate(lbl_order):
        viol_ax.get_children()[i * 2].set_alpha(0.41)

        viol_ax.text(i - 0.13, viol_ax.get_ylim()[1],
                     "n={}".format((gene_stat == lbl).sum()),
                     size=15, rotation=37, ha='left', va='bottom')

    viol_xlims = viol_ax.get_xlim()
    viol_ax.plot(viol_xlims, [0.5, 0.5],
                 color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    viol_ax.plot(viol_xlims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    viol_ax.set_ylim(viol_ylims)

    size_dict = dict()
    for mtype in use_aucs.index:
        if isinstance(mtype, RandomType):
            if mtype.size_dist in size_dict:
                size_dict[mtype.size_dist] += [mtype]
            else:
                size_dict[mtype.size_dist] = [mtype]

    for mtype_stat, face_clr, edge_clr, ln_wdth in zip(
            lbl_order[1:],
            ['none', variant_clrs['Point'], variant_clrs['Point']],
            [variant_clrs['Point'], variant_clrs['Point'], 'none'],
            [1.7, 1.7, 0]
            ):

        plt_mtypes = gene_stat.index[gene_stat == mtype_stat]
        size_rtypes = [size_dict[np.sum(pheno_dict[mtype])]
                       for mtype in plt_mtypes]
        mean_vals = [701 * np.mean(pheno_dict[mtype]) for mtype in plt_mtypes]

        sctr_ax.scatter([use_aucs[plt_mtype] for plt_mtype in plt_mtypes],
                        [use_aucs[rtype].max() for rtype in size_rtypes],
                        s=mean_vals, alpha=0.11, facecolor=face_clr,
                        edgecolor=edge_clr, linewidth=ln_wdth)

    sctr_ax.plot(viol_ylims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    sctr_ax.plot([0.5, 0.5], viol_ylims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    sctr_ax.plot(viol_ylims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    sctr_ax.plot([1, 1], viol_ylims, color='black', linewidth=1.9, alpha=0.89)
    sctr_ax.plot(viol_ylims, viol_ylims,
            color='#550000', linewidth=1.7, linestyle='--', alpha=0.41)

    sctr_ax.set_xlabel("AUC of Oncogene Mutation", size=19, weight='semibold')
    sctr_ax.set_ylabel("Best AUC of\nSize-Matched Randoms",
                       size=19, weight='semibold')

    sctr_ax.set_xlim(viol_ylims)
    sctr_ax.set_ylim(viol_ylims)

    plt.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "random-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_size_comparison(auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    plot_df = pd.DataFrame({
        'Size': [pheno_dict[mtype].sum() for mtype in use_aucs.index],
        'AUC': use_aucs.values,
        'Gene': [tuple(mtype.label_iter())[0] for mtype in use_aucs.index]
        })

    ax.scatter(plot_df.Size, plot_df.AUC,
               c=[choose_label_colour(gene) for gene in plot_df.Gene],
               s=23, alpha=0.17, edgecolor='none')

    size_lm = smf.ols('AUC ~ C(Gene) + Size', data=plot_df).fit()
    coef_vals = size_lm.params.sort_values()[::-1]
    gene_coefs = coef_vals[coef_vals.index.str.match('^C(Gene)*')]

    coef_lbl = '\n'.join(
        ["Size Coef: {:.3g}".format(coef_vals.Size)]
        + ["{} Coef: {:.3g}".format(gn.split("[T.")[1].split("]")[0], coef)
           for gn, coef in gene_coefs[:4].iteritems()]
        )

    ax.text(0.97, 0.07, coef_lbl,
            size=14, ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel("# of Samples Affected", size=20, weight='semibold')
    ax.set_ylabel("AUC", size=20, weight='semibold')

    ax.set_xlim([10, ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0], 1.01])

    ax.plot(ax.get_xlim(), [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot(ax.get_xlim(), [1.0, 1.0],
            color='black', linewidth=1.9, alpha=0.89)

    plt.tight_layout(w_pad=2.7)
    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "size-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_sub_comparisons(auc_vals, pheno_dict, conf_vals, args, add_lgnd):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.47

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for mtype in auc_vals.index
        ]]

    # for each gene whose mutations were tested, pick a random colour
    # to use for plotting the results for the gene
    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):

        # if there were subgroupings tested for the gene, find the results
        # for the mutation representing all point mutations for this gene...
        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            # ...as well as the results for the best subgrouping of
            # mutations found for this gene
            base_indx = auc_vec.index.get_loc(base_mtype)
            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
            line_dict[auc_tupl] = dict(c=choose_label_colour(gene))
            base_size = np.mean(pheno_dict[base_mtype])
            plt_size = 0.07 * base_size ** 0.5

            plot_dict[auc_tupl] = [plt_size, ('', '')]
            plt_min = min(plt_min, auc_vec[base_indx] - 0.03,
                          auc_vec[best_subtype] - 0.03)

            best_prop = np.mean(pheno_dict[best_subtype]) / base_size
            conf_sc = calc_conf(conf_vals[best_subtype],
                                conf_vals[base_mtype])

            # ...and if we are sure that the optimal subgrouping AUC is
            # better than the point mutation AUC then add a label with the
            # gene name and a description of the best found subgrouping...
            if auc_tupl[1] > 0.7 and conf_sc > 0.8:
                plot_dict[auc_tupl][1] = gene, get_fancy_label(
                    tuple(best_subtype.subtype_iter())[0][1],
                    pnt_link='\nor ', phrase_link=' '
                    )

            # ...if we are not sure but the respective AUCs are still
            # pretty great then add a label with just the gene name...
            elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                plot_dict[auc_tupl][1] = gene, ''

            auc_bbox = (auc_tupl[0] - plt_size / 2,
                        auc_tupl[1] - plt_size / 2, plt_size, plt_size)

            pie_ax = inset_axes(ax, width='100%', height='100%',
                                bbox_to_anchor=auc_bbox,
                                bbox_transform=ax.transData,
                                axes_kwargs=dict(aspect='equal'), borderpad=0)

            pie_ax.pie(x=[best_prop, 1 - best_prop],
                       colors=[line_dict[auc_tupl]['c'] + (0.77,),
                               line_dict[auc_tupl]['c'] + (0.29,)],
                       explode=[0.29, 0], startangle=90)

    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    ax.grid(linewidth=0.83, alpha=0.41)

    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("Accuracy of Gene-Wide Classifier",
                  size=23, weight='semibold')
    ax.set_ylabel("Accuracy of Best Subgrouping Classifier",
                  size=23, weight='semibold')

    if add_lgnd:
        ax, plot_dict = add_scatterpie_legend(ax, plot_dict, plt_min, args)

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[[plt_min + 0.01, 0.99]] * 2,
                                       seed=args.seed, line_dict=line_dict)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_copy_comparisons(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    line_dict = dict()

    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] == pnt_mtype
             or not (tuple(mtype.subtype_iter())[0][1]
                     & copy_mtype).is_empty())
        for mtype in auc_vals.index
        ]]

    plt_min = 0.47
    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = auc_vec.index.get_loc(base_mtype)
            base_gain = base_mtype | MuType({('Gene', gene): dup_mtype})
            base_loss = base_mtype | MuType({('Gene', gene): loss_mtype})

            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            if auc_vec[best_subtype] > 0.6:
                auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
                line_dict[auc_tupl] = dict(c=choose_label_colour(gene))

                if base_gain in pheno_dict and base_loss in pheno_dict:
                    cnv_size = np.mean(pheno_dict[base_gain]
                                       | pheno_dict[base_loss])

                elif base_gain in pheno_dict:
                    cnv_size = np.mean(pheno_dict[base_gain])
                elif base_loss in pheno_dict:
                    cnv_size = np.mean(pheno_dict[base_loss])

                plt_size = 0.07 * cnv_size ** 0.5
                plot_dict[auc_tupl] = [plt_size, ('', '')]
                plt_min = min(plt_min, auc_vec[base_indx] - 0.03,
                              auc_vec[best_subtype] - 0.03)

                best_prop = np.mean(pheno_dict[best_subtype]) / cnv_size
                conf_sc = calc_conf(conf_vals[best_subtype],
                                    conf_vals[base_mtype])

                if conf_sc > 0.9:
                    plot_dict[auc_tupl][1] = gene, get_fancy_label(
                        tuple(best_subtype.subtype_iter())[0][1],
                        pnt_link='\nor ', phrase_link=' '
                        )

                elif auc_tupl[0] > 0.7 or auc_tupl[1] > 0.7:
                    plot_dict[auc_tupl][1] = gene, ''

                auc_bbox = (auc_tupl[0] - plt_size / 2,
                            auc_tupl[1] - plt_size / 2, plt_size, plt_size)

                pie_ax = inset_axes(
                    ax, width='100%', height='100%',
                    bbox_to_anchor=auc_bbox, bbox_transform=ax.transData,
                    axes_kwargs=dict(aspect='equal'), borderpad=0
                    )

                pie_ax.pie(x=[best_prop, 1 - best_prop],
                           colors=[line_dict[auc_tupl]['c'] + (0.77, ),
                                   line_dict[auc_tupl]['c'] + (0.29, )],
                           explode=[0.29, 0], startangle=90)

    # figure out where to place the labels for each point, and plot them
    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot(plt_lims, [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], plt_lims,
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlabel("AUC using all point mutations", size=23, weight='semibold')
    ax.set_ylabel("AUC of best found CNA subgrouping",
                  size=23, weight='semibold')

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[plt_lims, plt_lims],
                                       seed=args.seed, line_dict=line_dict)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "copy-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour sample -omic dataset")
    parser.add_argument('classif', help="a mutation classifier")

    parser.add_argument(
        '--seed', type=int,
        help="random seed for fixing plot elements like label placement"
        )
    parser.add_argument('--legends', action='store_true',
                        help="add plot legends where applicable?")

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_list = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                              'Levels': '__'.join(out_data[1].split(
                                  'out-trnsf__')[1].split('__')[:-1])}
                             for out_data in out_datas])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby('Levels')['Samps'].min()
    if 'Consequence__Exon' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_dict[lvls] = pickle.load(f)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_dict[lvls] = pickle.load(f)

    auc_df = pd.concat(auc_dict.values())
    conf_list = pd.concat(conf_dict.values())
    assert auc_df.index.isin(phn_dict).all()

    # create the plots
    plot_random_comparison(auc_df['mean'], phn_dict, args)
    plot_size_comparison(auc_df['mean'], phn_dict, args)

    plot_sub_comparisons(auc_df['mean'], phn_dict, conf_list,
                         args, add_lgnd=args.legends)
    plot_copy_comparisons(auc_df['mean'], phn_dict, conf_list, args)


if __name__ == '__main__':
    main()

