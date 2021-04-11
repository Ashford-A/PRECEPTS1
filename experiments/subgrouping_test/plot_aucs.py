"""
This module produces plots comparing AUCs of different classification tasks,
and especially subgrouping tasks against their gene-wide counterparts.

Example usages:
    python -m dryads-research.experiments.subgrouping_test.plot_aucs \
        microarray METABRIC_LumA Ridge
    python -m dryads-research.experiments.subgrouping_test.plot_aucs \
        Firehose SKCM Ridge
    python -m dryads-research.experiments.subgrouping_test.plot_aucs \
        Firehose BRCA_LumA Ridge --legends

"""

from ..utilities.mutations import (
    pnt_mtype, copy_mtype, deep_mtype, dup_mtype, loss_mtype, RandomType)
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.misc import get_label, get_subtype, choose_label_colour
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')


def add_scatterpie_legend(ax, plot_dict, plt_min, base_mtype, args):
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

    if base_mtype == pnt_mtype:
        base_lbl = 'any point mutation'
        all_lbl = 'point-mutated'

    elif base_mtype == (pnt_mtype | deep_mtype):
        base_lbl = 'any point mutation\nor deep loss/gain'
        all_lbl = 'mutated'

    else:
        raise ValueError(
            "Unrecognized `base_mtype` argument: {}".format(base_mtype))

    coh_lbl = "% of {} samples\nwith {} in gene".format(
        get_cohort_label(args.cohort), base_lbl)
    ax.text(lgnd_x - lgnd_sz / 103, lgnd_y + lgnd_sz * 0.71, coh_lbl,
            size=15, style='italic', ha='center', va='bottom')

    ax.text(lgnd_x - lgnd_sz * 0.7, lgnd_y - lgnd_sz * 0.13,
            "% of gene's {}\nsamples with best subgrouping".format(all_lbl),
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


def plot_sub_comparisons(auc_df, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.47

    # filter out experiment results for mutations representing randomly
    # chosen sets of samples rather than actual mutations
    use_aucs = auc_df[[not isinstance(mtype, RandomType)
                       and (get_subtype(mtype) & copy_mtype).is_empty()
                       for mtype in auc_df.index]]

    # filter out results for genes which did not have any subgrouping tasks,
    # and group results according to the gene each subgrouping came from
    auc_gby = use_aucs['mean'].groupby(get_label).filter(
        lambda aucs: len(aucs) > 1).groupby(get_label)

    # for each gene, get the gene-wide subgrouping
    for gene, auc_vec in auc_gby:
        base_mtype = MuType({('Gene', gene): pnt_mtype})

        # find the results of the gene-wide subgrouping as well
        # as of the best-performing subgrouping
        base_indx = auc_vec.index.get_loc(base_mtype)
        best_subtype = auc_vec[:base_indx].append(
            auc_vec[(base_indx + 1):]).idxmax()

        # get the plotting properties to use for this comparison
        auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
        line_dict[auc_tupl] = dict(c=choose_label_colour(gene))
        base_size = np.mean(pheno_dict[base_mtype])
        best_prop = np.mean(pheno_dict[best_subtype]) / base_size

        # add an entry to the list of labels to be placed, adjust size of
        # plotting region, find if best subgrouping is cv-significantly better
        plt_size = 0.07 * base_size ** 0.5
        plot_dict[auc_tupl] = [plt_size, ('', '')]
        plt_min = min(plt_min, auc_tupl[0] - 0.03, auc_tupl[1] - 0.03)
        cv_sig = (np.array(use_aucs['CV'][best_subtype])
                  > np.array(use_aucs['CV'][base_mtype])).all()

        # ...and if we are sure that the optimal subgrouping AUC is
        # better than the point mutation AUC then add a label with the
        # gene name and a description of the best found subgrouping...
        if auc_vec.max() >= 0.7:
            if cv_sig:
                plot_dict[auc_tupl][1] = gene, get_fancy_label(
                    get_subtype(best_subtype),
                    pnt_link='\nor ', phrase_link=' '
                    )

            # ...if we are not sure but the respective AUCs are still
            # pretty great then add a label with just the gene name...
            else:
                plot_dict[auc_tupl][1] = gene, ''

        # draw the scatter-piechart for this gene's results
        pie_bbox = (auc_tupl[0] - plt_size / 2, auc_tupl[1] - plt_size / 2,
                    plt_size, plt_size)

        pie_ax = inset_axes(ax, width='100%', height='100%',
                            bbox_to_anchor=pie_bbox,
                            bbox_transform=ax.transData,
                            axes_kwargs=dict(aspect='equal'), borderpad=0)

        pie_ax.pie(x=[best_prop, 1 - best_prop],
                   colors=[line_dict[auc_tupl]['c'] + (0.77, ),
                           line_dict[auc_tupl]['c'] + (0.29, )],
                   explode=[0.29, 0], startangle=90)

    # makes sure plot labels don't overlap with equal-AUC diagonal line
    for k in np.linspace(plt_min, 1, 400):
        plot_dict[k, k] = [(1 - plt_min) / 387, ('', '')]

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

    if args.legends:
        ax, plot_dict = add_scatterpie_legend(ax, plot_dict, plt_min,
                                              pnt_mtype, args)

    else:
        ax.text(0.97, 0.02, get_cohort_label(args.cohort), size=21,
                style='italic', ha='right', va='bottom',
                transform=ax.transAxes)

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[plt_lims, plt_lims],
                                       plc_lims=[[plt_min + 0.01, 0.98]] * 2,
                                       seed=args.seed, line_dict=line_dict)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "sub-comparisons_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_copy_comparisons(auc_df, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(10.3, 11))

    plot_dict = dict()
    line_dict = dict()
    plt_min = 0.47

    # filter out subgroupings that are neither the gene-wide task nor one of
    # the subgroupings containing copy number alterations
    use_aucs = auc_df[[not isinstance(mtype, RandomType)
                       and (get_subtype(mtype) == pnt_mtype
                            or not (get_subtype(mtype)
                                    & copy_mtype).is_empty())
                       for mtype in auc_df.index]]

    # filter out results for genes which did not have any subgrouping tasks,
    # and group results according to the gene each subgrouping came from
    auc_gby = use_aucs['mean'].groupby(get_label).filter(
        lambda aucs: len(aucs) > 1).groupby(get_label)

    # for each gene, get the gene-wide subgrouping
    for gene, auc_vec in auc_gby:
        base_mtype = MuType({('Gene', gene): pnt_mtype})

        # find the results of the gene-wide subgrouping as well
        # as of the best-performing subgrouping
        base_indx = auc_vec.index.get_loc(base_mtype)
        best_subtype = auc_vec[:base_indx].append(
            auc_vec[(base_indx + 1):]).idxmax()

        auc_tupl = auc_vec[base_mtype], auc_vec[best_subtype]
        line_dict[auc_tupl] = dict(c=choose_label_colour(gene))
        base_gain = base_mtype | MuType({('Gene', gene): dup_mtype})
        base_loss = base_mtype | MuType({('Gene', gene): loss_mtype})

        if not (best_subtype & dup_mtype).is_empty():
            cnv_size = np.mean(pheno_dict[base_gain])
        else:
            cnv_size = np.mean(pheno_dict[base_loss])

        plt_size = 0.07 * cnv_size ** 0.5
        plot_dict[auc_tupl] = [plt_size, ('', '')]
        plt_min = min(plt_min, auc_vec[base_indx] - 0.03,
                      auc_vec[best_subtype] - 0.03)
        best_prop = np.mean(pheno_dict[best_subtype]) / cnv_size

        cv_sig = (np.array(use_aucs['CV'][best_subtype])
                  > np.array(use_aucs['CV'][base_mtype])).all()

        # ...and if we are sure that the optimal subgrouping AUC is
        # better than the point mutation AUC then add a label with the
        # gene name and a description of the best found subgrouping...
        if auc_vec.max() >= 0.7:
            if cv_sig:
                plot_dict[auc_tupl][1] = gene, get_fancy_label(
                    get_subtype(best_subtype),
                    pnt_link='\nor ', phrase_link=' '
                )

            # ...if we are not sure but the respective AUCs are still
            # pretty great then add a label with just the gene name...
            else:
                plot_dict[auc_tupl][1] = gene, ''

        pie_bbox = (auc_tupl[0] - plt_size / 2,
                    auc_tupl[1] - plt_size / 2, plt_size, plt_size)

        pie_ax = inset_axes(ax, width='100%', height='100%',
                            bbox_to_anchor=pie_bbox,
                            bbox_transform=ax.transData,
                            axes_kwargs=dict(aspect='equal'), borderpad=0)

        pie_ax.pie(x=[best_prop, 1 - best_prop],
                   colors=[line_dict[auc_tupl]['c'] + (0.77, ),
                           line_dict[auc_tupl]['c'] + (0.29, )],
                   explode=[0.29, 0], startangle=90)

    # makes sure plot labels don't overlap with equal-AUC diagonal line
    for k in np.linspace(plt_min, 1, 400):
        plot_dict[k, k] = [(1 - plt_min) / 387, ('', '')]

    plt_lims = plt_min, 1 + (1 - plt_min) / 181
    ax.grid(linewidth=0.83, alpha=0.41)

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
                                       plc_lims=[[plt_min + 0.01, 0.98]] * 2,
                                       seed=args.seed, line_dict=line_dict)

    if args.legends:
        ax, plot_dict = add_scatterpie_legend(ax, plot_dict, plt_min,
                                              pnt_mtype | deep_mtype, args)

    else:
        ax.text(0.97, 0.02, get_cohort_label(args.cohort), size=21,
                style='italic', ha='right', va='bottom',
                transform=ax.transAxes)

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
        description="Plots comparisons of classifier tasks' performance."
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

    # parse command line arguments, find experiments that finished running
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    # find each experiment's subgrouping enumeration criteria
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

    # create directory where plots will be stored
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    phn_dict = dict()
    auc_dict = dict()

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

    auc_df = pd.concat(auc_dict.values())
    assert auc_df.index.isin(phn_dict).all()

    # create the plots
    plot_size_comparison(auc_df['mean'], phn_dict, args)
    plot_sub_comparisons(auc_df, phn_dict, args)
    plot_copy_comparisons(auc_df, phn_dict, args)


if __name__ == '__main__':
    main()

