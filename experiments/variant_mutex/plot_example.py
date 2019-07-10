
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'variant_mutex')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'example')

from HetMan.experiments.variant_mutex import *
from HetMan.experiments.variant_baseline.merge_tests import merge_cohort_data
from HetMan.experiments.subvariant_infer import variant_clrs

import argparse
from pathlib import Path
import dill as pickle
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.patches as ptchs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

expr_cmap = sns.diverging_palette(189, 11, sep=53, s=91, l=49, as_cmap=True)
expr_mat = np.random.rand(17, 71)


def get_mtype_str(mtype):
    mtype_str = str(mtype).replace('_Mutation', '')

    return mtype_str


def plot_base_classification(good_exs, stat_dict, out_infer, auc_dict,
                             cdata, args):
    fig, ((expr_ax1, expr_ax2),
          (clf_ax1, clf_ax2), (clf_ax3, sum_ax)) = plt.subplots(
              figsize=(13, 8), nrows=3, ncols=2)

    for ax in expr_ax1, expr_ax2, clf_ax1, clf_ax2, clf_ax3, sum_ax:
        ax.axis('off')

    use_mtype1, use_mtype2 = good_exs['Conv'].index[0]
    use_gene1 = use_mtype1.subtype_list()[0][0]
    use_gene2 = use_mtype2.subtype_list()[0][0]
    mtype_str1 = get_mtype_str(use_mtype1)
    mtype_str2 = get_mtype_str(use_mtype2)

    mtype_clr1 = sns.light_palette(variant_clrs['Point'],
                                   n_colors=7, as_cmap=False)[2]
    mtype_clr2 = sns.dark_palette(variant_clrs['Point'],
                                  n_colors=7, as_cmap=False)[4]

    coh_genes = cdata.get_features()
    ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                if annot['Chr'] in {cdata.gene_annot[use_gene1]['Chr'],
                                    cdata.gene_annot[use_gene2]['Chr']}}

    ex_prop = len(ex_genes) / len(coh_genes)
    stat_tbl = pd.crosstab(stat_dict[use_mtype1], stat_dict[use_mtype2],
                           margins=True)
    prop_tbl = stat_tbl / len(cdata.get_samples())
    ovlp_test = fisher_exact(stat_tbl.iloc[:2, :2])

    if '_' in args.cohort:
        coh_lbl = "{}({})".format(*args.cohort.split('_'))
    else:
        coh_lbl = args.cohort

    if args.cohort != 'beatAML':
        coh_lbl = "TCGA-{}".format(coh_lbl)

    fig.text(0, 1,
             "Inferring mutation similarity in the {} patient "
             "cohort".format(coh_lbl),
             size=21, ha='left', va='bottom', weight='semibold')

    expr_ax1.text(0, 0.96, '1', size=16, ha='right', va='top',
                  bbox={'boxstyle': 'circle', 'facecolor': 'white',
                        'linewidth': 2.3})

    expr_ax1.text(0.03, 1,
                  "Remove expression\nfeatures on the\nsame chromosome\n"
                  "as {} or {}.".format(use_gene1, use_gene2),
                  size=13, ha='left', va='top')

    heat_ax11 = inset_axes(
        expr_ax1, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.37, 0.05, 0.58 * ex_prop, 0.93),
        bbox_transform=expr_ax1.transAxes
        )
    heat_ax11.axis('off')

    sns.heatmap(expr_mat[:, :9], ax=heat_ax11, cmap=expr_cmap, center=0.5,
                cbar=False, linewidths=6/7, linecolor='0.93', alpha=7/13)

    heat_ax12 = inset_axes(
        expr_ax1, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.58 * ex_prop + 0.39, 0.05,
                        0.58 * (1 - ex_prop), 0.93),
        bbox_transform=expr_ax1.transAxes
        )
    heat_ax12.axis('off')

    sns.heatmap(expr_mat[:, 9:], ax=heat_ax12, cmap=expr_cmap, center=0.5,
                cbar=False, linewidths=3/7, linecolor='black')

    expr_ax1.axvline(x=0.58 * ex_prop + 0.38, ymin=0.03, ymax=1,
                     linestyle=':', linewidth=2.3)

    if args.cohort == 'beatAML':
        expr_ax1.text(0.66, 0.99,
                      "{} Kallisto RNA-seq features".format(len(coh_genes)),
                      size=12, ha='center', va='bottom')
    else:
        expr_ax1.text(0.66, 0.99,
                      "{} Firehose RNA-seq features".format(len(coh_genes)),
                      size=12, ha='center', va='bottom')

    expr_ax1.text(0.98, 0.51,
                  "{} tumour samples".format(len(cdata.get_samples())),
                  size=12, ha='left', va='center', rotation=270)

    expr_ax1.text(0.58 * ex_prop + 0.37, 0.04,
                  "{} features\nfiltered out".format(len(ex_genes)),
                  size=11, ha='right', va='top')
    expr_ax1.text(0.58 * ex_prop + 0.39, 0.04,
                  "{} features\nleft in".format(len(set(coh_genes)
                                                    - ex_genes)),
                  size=11, ha='left', va='top')

    expr_ax2.text(0, 0.96, '2', size=16, ha='right', va='top',
                  bbox={'boxstyle': 'circle', 'facecolor': 'white',
                        'linewidth': 2.3})

    expr_ax2.text(0.03, 1,
                  "Stratify samples\naccording to the\npresence of a pair"
                  "\nof mutation types:\n\nM1)\n  {}"
                  "\n\nM2)\n  {}".format(mtype_str1, mtype_str2),
                  size=13, ha='left', va='top')

    heat_ax21 = inset_axes(
        expr_ax2, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.38, 0.06 + 0.88 * (1 - prop_tbl.loc[False, False]),
                        0.35, 0.88 * prop_tbl.loc[False, False]),
        bbox_transform=expr_ax2.transAxes
        )
    heat_ax21.axis('off')

    sns.heatmap(expr_mat[:12, 27:], ax=heat_ax21, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')

    heat_ax22 = inset_axes(
        expr_ax2, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.38, 0.05 + 0.88 * prop_tbl.loc[True, 'All'],
                        0.35, 0.88 * prop_tbl.loc[False, True]),
        bbox_transform=expr_ax2.transAxes
        )
    heat_ax22.axis('off')

    sns.heatmap(expr_mat[12:14, 27:], ax=heat_ax22, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')

    heat_ax23 = inset_axes(
        expr_ax2, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.38, 0.04 + 0.88 * prop_tbl.loc[True, True],
                        0.35, 0.88 * prop_tbl.loc[True, False]),
        bbox_transform=expr_ax2.transAxes
        )
    heat_ax23.axis('off')

    sns.heatmap(expr_mat[14:16, 27:], ax=heat_ax23, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')

    heat_ax24 = inset_axes(
        expr_ax2, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.38, 0.03, 0.35, 0.88 * prop_tbl.loc[True, True]),
        bbox_transform=expr_ax2.transAxes
        )
    heat_ax24.axis('off')

    sns.heatmap(expr_mat[16:, 27:], ax=heat_ax24, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')

    expr_ax2.add_patch(ptchs.Rectangle(
        (0.75, 0.03), 0.03, 0.91, facecolor=variant_clrs['WT'], alpha=0.31))
    expr_ax2.text(0.765, 0.93, 'M1', ha='center', va='top', size=8)

    expr_ax2.add_patch(ptchs.Rectangle(
        (0.79, 0.03), 0.03, 0.91, facecolor=variant_clrs['WT'], alpha=0.31))
    expr_ax2.text(0.805, 0.93, 'M2', ha='center', va='top', size=8)

    if args.cohort == 'beatAML':
        expr_ax2.text(0.84, 0.81, "variant calls",
                      size=12, ha='left', va='top')
    else:
        expr_ax2.text(0.84, 1.03,
                      "mc3\nvariant calls\n&\nFirehose\nGISTIC2\nCNA calls",
                      size=12, ha='left', va='top')

    expr_ax2.add_patch(ptchs.Rectangle(
        (0.75, 0.04), 0.03, 0.91 * prop_tbl.loc[True, 'All'],
        facecolor=mtype_clr1, alpha=0.43)
        )

    expr_ax2.add_patch(ptchs.Rectangle(
        (0.79, 0.04), 0.03, 0.91 * prop_tbl.loc[True, True],
        facecolor=mtype_clr2, alpha=0.43)
        )

    expr_ax2.add_patch(ptchs.Rectangle(
        (0.79, 0.0475 + 0.91 * prop_tbl.loc[True, 'All']),
        0.03, 0.91 * prop_tbl.loc[False, True],
        facecolor=mtype_clr2, alpha=0.43)
        )

    expr_ax2.text(0.77, 0.02, "{:.1%}".format(prop_tbl.loc[True, 'All']),
                  size=10, ha='right', va='top', rotation=45)
    expr_ax2.text(0.81, 0.02, "{:.1%}".format(prop_tbl.loc['All', True]),
                  size=10, ha='right', va='top', rotation=45)

    expr_ax2.axhline(y=0.055 + 0.88 * (1 - prop_tbl.loc[False, False]),
                     xmin=0.37, xmax=0.83, linewidth=1.1, linestyle=':',
                     color='black', alpha=0.67)

    expr_ax2.axhline(y=0.045 + 0.88 * prop_tbl.loc[True, 'All'],
                     xmin=0.37, xmax=0.83, linewidth=1.1, linestyle=':',
                     color='black', alpha=0.67)

    expr_ax2.axhline(y=0.035 + 0.88 * prop_tbl.loc[True, True],
                     xmin=0.37, xmax=0.83, linewidth=1.1, linestyle=':',
                     color='black', alpha=0.67)

    expr_ax2.text(0.84, 0.03 + 0.83 * (1 - prop_tbl.loc[False, False]),
                  "genomic\nco-occurence:\n\n  {} samples with"
                  "\n  both mutations\n\n  two-sided Fisher's"
                  "\n  exact test p-val:\n    {:.3g}".format(
                      stat_tbl.loc[True, True], ovlp_test[1]),
                  size=9, ha='left', va='center')

    clf_ax1.text(0, 0.96, '3', ha='right', va='top', size=16,
                 bbox={'boxstyle': 'circle', 'facecolor': 'white',
                       'linewidth': 2.3})

    clf_ax1.text(0.03, 1,
                 "Train classifiers\nto separate the\n{} samples with\n{}\n"
                 "and without\n{}\nfrom the {} samples\nwith neither "
                 "mutation.".format(stat_tbl.loc[True, False], mtype_str1,
                                    mtype_str2, stat_tbl.loc[False, False]),
                 size=13, ha='left', va='top')

    heat_ax31 = inset_axes(
        clf_ax1, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.41, 0.04 + 0.96 * (prop_tbl.loc[False, False]
                                             / prop_tbl.loc['All', False]),
                        0.34, 0.96 * (prop_tbl.loc[True, False]
                                      / prop_tbl.loc['All', False])),
        bbox_transform=clf_ax1.transAxes
        )
    heat_ax31.axis('off')

    sns.heatmap(expr_mat[2:4, 27:], ax=heat_ax31, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')
    heat_ax31.text(0.5, 1.01, 'M1 & ~M2', size=9, ha='center', va='bottom',
                   transform=heat_ax31.transAxes)

    heat_ax32 = inset_axes(
        clf_ax1, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0.41, 0,
                        0.34, 0.96 * (prop_tbl.loc[False, False]
                                      / prop_tbl.loc['All', False])),
        bbox_transform=clf_ax1.transAxes
        )
    heat_ax32.axis('off')

    sns.heatmap(expr_mat[4:, 27:], ax=heat_ax32, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')
    heat_ax32.text(0.5, -0.01, '~M1 & ~M2', size=9, ha='center', va='top',
                   transform=heat_ax32.transAxes)

    clf_ax1.axhline(y=0.02 + 0.96 * (prop_tbl.loc[False, False]
                                     / prop_tbl.loc['All', False]),
                    xmin=0.4, xmax=0.76, linestyle='--', linewidth=2.3)

    infer_vals1 = np.array(out_infer.loc[[(use_mtype1, use_mtype2)], 0][0])
    wt_vals1 = np.concatenate(infer_vals1[~stat_dict[use_mtype1]
                                          & ~stat_dict[use_mtype2]])

    mut_vals1 = np.concatenate(infer_vals1[stat_dict[use_mtype1]
                                           & ~stat_dict[use_mtype2]])
    oth_vals1 = np.concatenate(infer_vals1[~stat_dict[use_mtype1]
                                           & stat_dict[use_mtype2]])

    clf_ax1.add_patch(ptchs.FancyArrow(
        0.77, 0.52, dx=0.04, dy=0, width=0.02, length_includes_head=True,
        head_length=0.02, linewidth=1.1, facecolor='white', edgecolor='black'
        ))

    vio_ax11 = inset_axes(clf_ax1, width='100%', height='100%', loc=10,
                          borderpad=0, bbox_to_anchor=(0.83, 0, 0.17, 1),
                          bbox_transform=clf_ax1.transAxes)
    vio_ax11.axis('off')

    sns.kdeplot(wt_vals1, shade=True, color=variant_clrs['WT'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax11)
    sns.kdeplot(mut_vals1, shade=True, color=mtype_clr1,
                vertical=True, linewidth=0, cut=0, ax=vio_ax11)

    clf_ax1.text(1, 0.99,
                 "task AUC: {:.3f}".format(
                     auc_dict[use_mtype1, use_mtype2][0]),
                 size=11, ha='right', va='bottom')

    clf_ax2.text(0, 0.96, '4', ha='right', va='top', size=16,
                 bbox={'boxstyle': 'circle', 'facecolor': 'white',
                       'linewidth': 2.3})

    clf_ax2.text(0.03, 1,
                 "Use trained classifiers\nto predict mutation\nscores for "
                 "the {}\nheld-out samples\nwith {}\nand without\n{}."
                 "\nCompare these scores\nto the scores obtained\nin "
                 "(3).".format(stat_tbl.loc[False, True],
                               mtype_str2, mtype_str1),
                 size=13, ha='left', va='top')

    vio_ax12 = inset_axes(clf_ax2, width='100%', height='100%', loc=10,
                          borderpad=0, bbox_to_anchor=(0.41, 0, 0.17, 1),
                          bbox_transform=clf_ax2.transAxes)
    vio_ax12.axis('off')

    sns.kdeplot(wt_vals1, shade=True, color=variant_clrs['WT'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax12)
    sns.kdeplot(mut_vals1, shade=True, color=mtype_clr1,
                vertical=True, linewidth=0, cut=0, ax=vio_ax12)

    vio_ax12.axhline(y=np.mean(wt_vals1), linewidth=2.1,
                     color=variant_clrs['WT'], linestyle=':',
                     alpha=0.71, clip_on=False)
    vio_ax12.axhline(y=np.mean(mut_vals1), linewidth=2.1, color=mtype_clr1,
                     linestyle=':', alpha=0.71, clip_on=False)

    vio_ax12.text(vio_ax12.get_xlim()[1] / 1.7, np.mean(mut_vals1),
                  "{:.2f} \u2192 1".format(np.mean(mut_vals1)),
                  size=9, ha='center', va='bottom')
    vio_ax12.text(vio_ax12.get_xlim()[1] / 1.7, np.mean(wt_vals1),
                  "{:.2f} \u2192 0".format(np.mean(wt_vals1)),
                  size=9, ha='center', va='bottom')

    oth_ax1 = inset_axes(clf_ax2, width='100%', height='100%', loc=10,
                         borderpad=0, bbox_to_anchor=(0.57, 0, 0.17, 1),
                         bbox_transform=clf_ax2.transAxes)
    oth_ax1.axis('off')

    sns.kdeplot(oth_vals1, shade=False, color=mtype_clr2,
                vertical=True, linewidth=1.9, cut=0, ax=oth_ax1)
    oth_ax1.set_ylim(vio_ax12.get_ylim())

    oth_ax1.axhline(y=np.mean(oth_vals1), linewidth=3.3, color=mtype_clr2,
                    linestyle=':', alpha=0.71, clip_on=False)

    siml1 = np.mean(oth_vals1) - np.mean(wt_vals1)
    siml1 /= np.mean(mut_vals1) - np.mean(wt_vals1)
    oth_ax1.text(oth_ax1.get_xlim()[1] * 0.83, np.mean(oth_vals1),
                 "{:.2f}\u2192({:.2f})".format(np.mean(oth_vals1), siml1),
                 size=11, ha='left', va='bottom', weight='semibold')

    clf_ax3.text(0, 0.96, '5', ha='right', va='top', size=16,
                 bbox={'boxstyle': 'circle', 'facecolor': 'white',
                       'linewidth': 2.3})

    clf_ax3.text(0.03, 1,
                 "Repeat (3) and (4) with the\ntwo sets of mutations "
                 "reversed.", size=13, ha='left', va='top')

    heat_ax41 = inset_axes(
        clf_ax3, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0, 0.65, 0.29, 0.13), bbox_transform=clf_ax3.transAxes
        )
    heat_ax41.axis('off')

    sns.heatmap(expr_mat[2:6, 27:], ax=heat_ax41, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')

    heat_ax42 = inset_axes(
        clf_ax3, width='100%', height='100%', loc=10, borderpad=0,
        bbox_to_anchor=(0, 0, 0.29, 0.61),
        bbox_transform=clf_ax3.transAxes
        )
    heat_ax42.axis('off')

    sns.heatmap(expr_mat[6:, 27:], ax=heat_ax42, cmap=expr_cmap,
                center=0.5, cbar=False, linewidths=3/7, linecolor='black')

    clf_ax3.axhline(y=0.63, xmin=-0.01, xmax=0.3,
                    linestyle='--', linewidth=2.3, clip_on=False)
    clf_ax3.add_patch(ptchs.FancyArrow(
        0.31, 0.43, dx=0.04, dy=0, width=0.02, length_includes_head=True,
        head_length=0.02, linewidth=1.1, facecolor='white', edgecolor='black'
        ))

    infer_vals2 = np.array(out_infer.loc[[(use_mtype1, use_mtype2)], 1][0])
    wt_vals2 = np.concatenate(infer_vals2[~stat_dict[use_mtype1]
                                          & ~stat_dict[use_mtype2]])

    mut_vals2 = np.concatenate(infer_vals2[~stat_dict[use_mtype1]
                                           & stat_dict[use_mtype2]])
    oth_vals2 = np.concatenate(infer_vals2[stat_dict[use_mtype1]
                                           & ~stat_dict[use_mtype2]])

    vio_ax2 = inset_axes(clf_ax3, width='100%', height='100%', loc=10,
                         borderpad=0, bbox_to_anchor=(0.37, 0, 0.17, 0.76),
                         bbox_transform=clf_ax3.transAxes)
    vio_ax2.axis('off')

    sns.kdeplot(wt_vals2, shade=True, color=variant_clrs['WT'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax2)
    sns.kdeplot(mut_vals2, shade=True, color=mtype_clr2, alpha=0.77,
                vertical=True, linewidth=0, cut=0, ax=vio_ax2)

    clf_ax3.text(0.41, 0.73,
                 "task AUC: {:.3f}".format(
                     auc_dict[use_mtype1, use_mtype2][1]),
                 size=10, ha='center', va='bottom')

    vio_ax2.axhline(y=np.mean(wt_vals2), linewidth=2.1,
                    color=variant_clrs['WT'], linestyle=':',
                    alpha=0.71, clip_on=False)
    vio_ax2.axhline(y=np.mean(mut_vals2), linewidth=2.1, color=mtype_clr2,
                    linestyle=':', alpha=0.71, clip_on=False)

    vio_ax2.text(vio_ax2.get_xlim()[1] / 1.7, np.mean(mut_vals2),
                 "{:.2f} \u2192 1".format(np.mean(mut_vals2)),
                 size=9, ha='center', va='bottom')
    vio_ax2.text(vio_ax2.get_xlim()[1] / 1.7, np.mean(wt_vals2),
                 "{:.2f} \u2192 0".format(np.mean(wt_vals2)),
                 size=9, ha='center', va='bottom')

    oth_ax2 = inset_axes(clf_ax3, width='100%', height='100%', loc=10,
                         borderpad=0, bbox_to_anchor=(0.53, 0, 0.17, 0.76),
                         bbox_transform=clf_ax3.transAxes)
    oth_ax2.axis('off')

    sns.kdeplot(oth_vals2, shade=False, color=mtype_clr1,
                vertical=True, linewidth=1.9, cut=0, ax=oth_ax2)
    oth_ax2.set_ylim(vio_ax2.get_ylim())

    oth_ax2.axhline(y=np.mean(oth_vals2), linewidth=3.3, color=mtype_clr1,
                    linestyle=':', alpha=0.71, clip_on=False)

    siml2 = np.mean(oth_vals2) - np.mean(wt_vals2)
    siml2 /= np.mean(mut_vals2) - np.mean(wt_vals2)
    oth_ax2.text(oth_ax2.get_xlim()[1] * 0.83, np.mean(oth_vals2),
                 "{:.2f}\u2192({:.2f})".format(np.mean(oth_vals2), siml2),
                 size=11, ha='left', va='bottom', weight='semibold')

    sum_ax.text(0, 0.96, '6', ha='right', va='top', size=16,
                bbox={'boxstyle': 'circle', 'facecolor': 'white',
                      'linewidth': 2.3})

    sum_ax.text(0.03, 1,
                "Combine the M1\u21d2M2 score calculated\nin (3) and the "
                "M1\u21d0M2 score calculated\nin (4) to infer the "
                "transcriptomic\nsimilarity M1\u21d4M2: ({:.2f})".format(
                    (siml1 + siml2) / 2),
                size=13, ha='left', va='top')

    plt.tight_layout(pad=1.7, w_pad=0.4, h_pad=1.3)
    plt.savefig(os.path.join(
        plot_dir, args.cohort, "mutex-classification_{}__{}.svg".format(
            args.cohort, args.classif)
        ),
        bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot an example diagram showing how overlap with other types of "
        "mutations can affect a mutation classification task."
        )

    # parse command line arguments, create directory where plots will be saved
    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('classif', help='a mutation classifier')
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)

    # search for experiment output directories corresponding to this cohort
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__samps-*/out-data__{}.p".format(args.cohort, args.classif))
        ]

    use_dir = out_datas[np.argmin([int(out_data[0].split('__samps-')[1])
                                   for out_data in out_datas])][0]
    cdata = merge_cohort_data(os.path.join(base_dir, use_dir), use_seed=671)

    # load inferred mutation relationship metrics generated by the experiment
    with open(os.path.join(base_dir, use_dir,
                           "out-simil__{}.p".format(args.classif)),
              'rb') as f:
        stat_dict, auc_dict, mutex_dict, siml_dict = pickle.load(f)

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                'FOUNDATION ONE', 'MSK-IMPACT']]
            == 'Yes').sum(axis=1) >= 3
        ]

    # find mutation pairs for which the classifier was able to successfully
    # predict the presence of each mutation in isolation from the other
    auc_df = (pd.DataFrame(auc_dict) >= 0.8).all(axis=0)
    use_mtypes = [(mtype1, mtype2)
                  for (mtype1, mtype2) in auc_df.index[auc_df]
                  if (mtype1.subtype_list()[0][0] in use_genes
                      and mtype2.subtype_list()[0][0] in use_genes
                      and (mtype1.subtype_list()[0][0]
                           != mtype2.subtype_list()[0][0]))]

    siml_df = pd.DataFrame({
        'Occur': pd.Series(mutex_dict)[use_mtypes],
        'SimilMean': pd.Series({mtypes: siml_dict[mtypes].loc['Other'].mean()
                                for mtypes in use_mtypes}),
        'SimilDiff': pd.Series({
            mtypes: np.abs(siml_dict[mtypes].loc['Other'].diff()[1])
            for mtypes in use_mtypes
            }),

        'SynerMean': pd.Series({mtypes: siml_dict[mtypes].loc['Both'].mean()
                                for mtypes in use_mtypes}),
        'SynerDiff': pd.Series({
            mtypes: np.abs(siml_dict[mtypes].loc['Both'].diff()[1])
            for mtypes in use_mtypes
            }),
        })

    good_exs = {'Conv': (siml_df.Occur * siml_df.SimilMean
                         + siml_df.SimilDiff).sort_values(),
                'Divr': (siml_df.Occur + siml_df.SimilMean
                         - siml_df.SimilDiff).sort_values()}

    with open(os.path.join(base_dir, use_dir,
                           "out-data__{}.p".format(args.classif)),
              'rb') as f:
        out_infer = pickle.load(f)['Infer'].loc[use_mtypes]

    plot_base_classification(good_exs, stat_dict, out_infer, auc_dict,
                             cdata, args)


if __name__ == '__main__':
    main()

