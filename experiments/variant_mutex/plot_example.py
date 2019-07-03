
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


def get_mtype_str(mtype):
    mtype_str = str(mtype).replace('_Mutation', '')

    return mtype_str


def plot_base_classification(good_exs, stat_dict, out_infer, auc_dict,
                             cdata, args):
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.axis('off')

    use_mtype1, use_mtype2 = good_exs['Conv'].index[0]
    use_gene1 = use_mtype1.subtype_list()[0][0]
    use_gene2 = use_mtype2.subtype_list()[0][0]
    mtype_str1 = get_mtype_str(use_mtype1)
    mtype_str2 = get_mtype_str(use_mtype2)

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
        ax.text(0, 1, "{}({})".format(*args.cohort.split('_')),
                size=21, ha='left', va='top', weight='semibold')
    else:
        ax.text(0, 1, args.cohort,
                size=21, ha='left', va='top', weight='semibold')

    ax.add_patch(ptchs.Rectangle((0, 0.61), 0.71, 0.3,
                                 facecolor='#FFEF8F', alpha=0.21))
    ax.text(1.7 * ex_prop, 0.76, "Firehose RNA-seq expression",
            size=18, weight='semibold', ha='left', va='center')

    ax.text(-0.01, 0.76, "{} tumour samples".format(len(cdata.get_samples())),
            size=15, ha='right', va='center', rotation=90)
    ax.text(0.35, 0.92,
            "{} non-sex chromosome gene features".format(len(coh_genes)),
            size=15, ha='center', va='bottom')

    ax.add_patch(ptchs.Rectangle((0.75, 0.61), 0.03, 0.3,
                                 facecolor=variant_clrs['WT'], alpha=0.31))
    ax.add_patch(ptchs.Rectangle((0.79, 0.61), 0.03, 0.3,
                                 facecolor=variant_clrs['WT'], alpha=0.31))
    ax.text(0.83, 0.81,
            "mc3\nvariant calls\n&\nFirehose\nGISTIC2\nCNA calls",
            size=12, weight='semibold', ha='left', va='center')

    ax.text(0.56 * ex_prop, 0.58, '1', size=16, ha='right', va='top',
            bbox={'boxstyle': 'circle', 'facecolor': 'white', 'linewidth': 2})

    ax.text(0.74 * ex_prop, 0.59,
            "Filter for {} features\nnot on the same chromosome\nas {} or "
            "{}.".format(len(set(coh_genes) - ex_genes),
                         use_gene1, use_gene2),
            size=13, ha='left', va='top')
 
    ax.add_patch(ptchs.Rectangle((0.75, 0.61),
                                 0.03, 0.3 * prop_tbl.loc[True, 'All'],
                                 facecolor=variant_clrs['Point'], alpha=0.43))

    ax.add_patch(ptchs.Rectangle((0.79, 0.61),
                                 0.03, 0.3 * prop_tbl.loc[True, True],
                                 facecolor=variant_clrs['Point'], alpha=0.43))

    ax.add_patch(ptchs.Rectangle((0.79,
                                  0.61 + 0.3 * prop_tbl.loc[True, 'All']),
                                 0.03, 0.3 * prop_tbl.loc[False, True],
                                 facecolor=variant_clrs['Point'], alpha=0.43))

    ax.text(0.77, 0.91, mtype_str1,
            size=10, ha='right', va='bottom', rotation=313)
    ax.text(0.81, 0.91, mtype_str2,
            size=10, ha='right', va='bottom', rotation=313)

    ax.text(0.77, 0.61, "{:.1%}".format(prop_tbl.loc[True, 'All']),
            size=10, ha='right', va='top', rotation=45)
    ax.text(0.81, 0.61, "{:.1%}".format(prop_tbl.loc['All', True]),
            size=10, ha='right', va='top', rotation=45)

    ax.axvline(x=0.71 * ex_prop, ymin=0.6, ymax=0.92, color='#E9B925',
               linestyle='--', linewidth=2.7, alpha=0.87)

    ax.axhline(y=0.61 + 0.3 * prop_tbl.loc[True, True],
               xmin=0.71 * ex_prop, xmax=0.82, linewidth=1.4,
               color=variant_clrs['Point'], linestyle=':', alpha=0.67)

    ax.axhline(y=0.61 + 0.3 * prop_tbl.loc[True, 'All'],
               xmin=0.71 * ex_prop, xmax=0.82, linewidth=1.4,
               color=variant_clrs['Point'], linestyle=':', alpha=0.67)

    ax.axhline(y=0.61 + 0.3 * (1 - prop_tbl.loc[False, False]),
               xmin=0.71 * ex_prop, xmax=0.82, linewidth=1.4,
               color=variant_clrs['Point'], linestyle=':', alpha=0.67)

    ax.add_patch(ptchs.Rectangle(
        (0.71 * ex_prop, 0.61),
        0.71 * (1 - ex_prop), 0.3 * prop_tbl.loc[True, True],
        facecolor='#FFEF8F', alpha=0.21, hatch='//\\\\'
        ))

    ax.add_patch(ptchs.Rectangle(
        (0.71 * ex_prop, 0.61 + 0.3 * prop_tbl.loc[True, True]),
        0.71 * (1 - ex_prop), 0.3 * prop_tbl.loc[True, False],
        facecolor='#FFEF8F', alpha=0.21, hatch='\\\\'
        ))

    ax.add_patch(ptchs.Rectangle(
        (0.71 * ex_prop, 0.61 + 0.3 * prop_tbl.loc[True, 'All']),
        0.71 * (1 - ex_prop), 0.3 * prop_tbl.loc[False, True],
        facecolor='#FFEF8F', alpha=0.21, hatch='//'
        ))

    ax.add_patch(ptchs.Rectangle(
        (0.71 * ex_prop, 0.61 + 0.3 * (1 - prop_tbl.loc[False, False])),
        0.71 * (1 - ex_prop), 0.3 * prop_tbl.loc[False, False],
        facecolor='#FFEF8F', alpha=0.21,
        ))

    ax.text(0.85, 0.61 + 1.83 * prop_tbl.loc[True, True],
            "{} samples with\nboth mutations,\ntwo-sided Fisher's\nexact "
            "test p-val:\n  {:.3g}".format(stat_tbl.loc[True, True],
                                           ovlp_test[1]),
            size=11, ha='left', va='center')

    ax.add_patch(ptchs.FancyArrow(
        0.83, 0.61 + 0.17 * prop_tbl.loc[True, True], dx=0.015, dy=0,
        width=0.004, length_includes_head=True, head_length=0.01,
        linewidth=1.1, facecolor='white', edgecolor='black'
        ))

    ax.text(0.01, 0.5, '2', ha='right', va='top', size=16,
            bbox={'boxstyle': 'circle', 'facecolor': 'white', 'linewidth': 2})

    ax.text(0.03, 0.51,
            "Train classifier to\nseparate the {}\nsamples with\n{}\nand "
            "without\n{} from\n{} samples with\nneither mutation.".format(
                stat_tbl.loc[True, False], mtype_str1, mtype_str2,
                stat_tbl.loc[False, False]
                ),
            size=13, ha='left', va='top')

    ax.add_patch(ptchs.Rectangle((0.31, 0.42), 0.11, 0.05,
                                 facecolor='#FFEF8F', alpha=0.37,
                                 hatch='\\\\'))
    ax.add_patch(ptchs.Rectangle((0.31, 0.35), 0.11, 0.05,
                                 facecolor='#FFEF8F', alpha=0.37))

    infer_vals1 = np.array(out_infer.loc[[(use_mtype1, use_mtype2)], 0][0])
    wt_vals1 = np.concatenate(infer_vals1[~stat_dict[use_mtype1]
                                          & ~stat_dict[use_mtype2]])

    mut_vals1 = np.concatenate(infer_vals1[stat_dict[use_mtype1]
                                           & ~stat_dict[use_mtype2]])
    oth_vals1 = np.concatenate(infer_vals1[~stat_dict[use_mtype1]
                                           & stat_dict[use_mtype2]])

    ax.add_patch(ptchs.FancyArrow(0.43, 0.41, dx=0.05, dy=0, width=0.008,
                                  length_includes_head=True, head_length=0.02,
                                  linewidth=2.3, facecolor='white',
                                  edgecolor='black'))

    vio_ax1 = inset_axes(ax, width='100%', height='100%', loc=10, borderpad=0,
                         bbox_to_anchor=(0.49, 0.31, 0.08, 0.2),
                         bbox_transform=ax.transAxes)
    vio_ax1.axis('off')

    sns.kdeplot(wt_vals1, shade=True, color=variant_clrs['WT'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax1)
    sns.kdeplot(mut_vals1, shade=True, color=variant_clrs['Point'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax1)

    ax.text(0.49, 0.51,
            "task AUC: {:.3f}".format(auc_dict[use_mtype1, use_mtype2][0]),
            size=12, ha='center', va='bottom')

    oth_ax1 = inset_axes(ax, width='100%', height='100%', loc=10, borderpad=0,
                         bbox_to_anchor=(0.59, 0.31, 0.08, 0.2),
                         bbox_transform=ax.transAxes)
    oth_ax1.axis('off')

    ax.add_patch(ptchs.Rectangle((0.6, 0.47), 0.11, 0.05,
                                 facecolor='#FFEF8F', alpha=0.37,
                                 hatch='//'))

    sns.kdeplot(oth_vals1, shade=False, color=variant_clrs['Point'],
                vertical=True, linewidth=1.9, cut=0, ax=oth_ax1)
    oth_ax1.set_xlim(vio_ax1.get_xlim())
    oth_ax1.set_ylim(vio_ax1.get_ylim())

    vio_ax1.axhline(y=np.mean(wt_vals1), xmin=0, xmax=2.1, linewidth=2.1,
                    color=variant_clrs['WT'], linestyle=':', alpha=0.71,
                    clip_on=False)
    vio_ax1.axhline(y=np.mean(mut_vals1), xmin=0, xmax=2.1, linewidth=2.1,
                    color=variant_clrs['Point'], linestyle=':', alpha=0.71,
                    clip_on=False)

    oth_ax1.text(oth_ax1.get_xlim()[1] * 0.85, np.mean(mut_vals1),
                 "{:.2f} \u2192 1".format(np.mean(mut_vals1)),
                 size=9, ha='left', va='center')
    oth_ax1.text(oth_ax1.get_xlim()[1] * 0.85, np.mean(wt_vals1),
                 "{:.2f} \u2192 0".format(np.mean(wt_vals1)),
                 size=9, ha='left', va='center')

    oth_ax1.axhline(y=np.mean(oth_vals1), xmin=0, xmax=0.99, linewidth=3.3,
                    color=variant_clrs['Point'], linestyle=':', alpha=0.71,
                    clip_on=False)

    oth_ax1.text(oth_ax1.get_xlim()[1] * 1.09, np.mean(oth_vals1),
                 "{:.2f} \u2192 {:.2f}".format(
                     np.mean(oth_vals1), (
                         (np.mean(oth_vals1) - np.mean(wt_vals1))
                         / (np.mean(mut_vals1) - np.mean(wt_vals1))
                        )
                    ),
                 size=11, ha='left', va='center')

    ax.text(0.77, 0.48, '3', ha='right', va='top', size=16,
            bbox={'boxstyle': 'circle', 'facecolor': 'white', 'linewidth': 2})

    ax.text(0.79, 0.49,
            "Use trained classifiers\nto predict mutation\nscores for {} "
            "held-out\nsamples with\n{}\nand without\n{}.".format(
                stat_tbl.loc[False, True], mtype_str2, mtype_str1),
            size=13, ha='left', va='top')

    ax.text(0.01, 0.29, '4', ha='right', va='top', size=16,
            bbox={'boxstyle': 'circle', 'facecolor': 'white', 'linewidth': 2})

    ax.text(0.03, 0.3,
            "Repeat (2) and (3) with the\ntwo sets of mutations reversed.",
            size=13, ha='left', va='top')

    ax.add_patch(ptchs.Rectangle((0.04, 0.18), 0.11, 0.05,
                                 facecolor='#FFEF8F', alpha=0.37,
                                 hatch='//'))
    ax.add_patch(ptchs.Rectangle((0.04, 0.11), 0.11, 0.05,
                                 facecolor='#FFEF8F', alpha=0.37))

    infer_vals2 = np.array(out_infer.loc[[(use_mtype1, use_mtype2)], 1][0])
    wt_vals2 = np.concatenate(infer_vals2[~stat_dict[use_mtype1]
                                          & ~stat_dict[use_mtype2]])

    mut_vals2 = np.concatenate(infer_vals2[~stat_dict[use_mtype1]
                                           & stat_dict[use_mtype2]])
    oth_vals2 = np.concatenate(infer_vals2[stat_dict[use_mtype1]
                                           & ~stat_dict[use_mtype2]])

    ax.add_patch(ptchs.FancyArrow(0.16, 0.17, dx=0.05, dy=0, width=0.008,
                                  length_includes_head=True, head_length=0.02,
                                  linewidth=2.3, facecolor='white',
                                  edgecolor='black'))

    vio_ax2 = inset_axes(ax, width='100%', height='100%', loc=10, borderpad=0,
                         bbox_to_anchor=(0.22, 0.09, 0.07, 0.16),
                         bbox_transform=ax.transAxes)
    vio_ax2.axis('off')

    sns.kdeplot(wt_vals2, shade=True, color=variant_clrs['WT'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax2)
    sns.kdeplot(mut_vals2, shade=True, color=variant_clrs['Point'],
                vertical=True, linewidth=0, cut=0, ax=vio_ax2)

    ax.text(0.22, 0.08,
            "task AUC: {:.3f}".format(auc_dict[use_mtype1, use_mtype2][1]),
            size=12, ha='center', va='top')

    oth_ax2 = inset_axes(ax, width='100%', height='100%', loc=10, borderpad=0,
                         bbox_to_anchor=(0.3, 0.09, 0.07, 0.16),
                         bbox_transform=ax.transAxes)
    oth_ax2.axis('off')

    ax.add_patch(ptchs.Rectangle((0.33, 0.2), 0.11, 0.05,
                                 facecolor='#FFEF8F', alpha=0.37,
                                 hatch='\\\\'))

    sns.kdeplot(oth_vals2, shade=False, color=variant_clrs['Point'],
                vertical=True, linewidth=1.9, cut=0, ax=oth_ax2)
    oth_ax2.set_xlim(vio_ax2.get_xlim())
    oth_ax2.set_ylim(vio_ax2.get_ylim())

    vio_ax2.axhline(y=np.mean(wt_vals2), xmin=0, xmax=1.9, linewidth=2.1,
                    color=variant_clrs['WT'], linestyle=':', alpha=0.71,
                    clip_on=False)
    vio_ax2.axhline(y=np.mean(mut_vals2), xmin=0, xmax=1.9, linewidth=2.1,
                    color=variant_clrs['Point'], linestyle=':', alpha=0.71,
                    clip_on=False)

    oth_ax2.text(oth_ax2.get_xlim()[1] * 0.85, np.mean(mut_vals2),
                 "{:.2f} \u2192 1".format(np.mean(mut_vals2)),
                 size=9, ha='left', va='center')
    oth_ax2.text(oth_ax2.get_xlim()[1] * 0.85, np.mean(wt_vals2),
                 "{:.2f} \u2192 0".format(np.mean(wt_vals2)),
                 size=9, ha='left', va='center')

    oth_ax2.axhline(y=np.mean(oth_vals2), xmin=0, xmax=1.05, linewidth=3.3,
                    color=variant_clrs['Point'], linestyle=':', alpha=0.71,
                    clip_on=False)

    oth_ax2.text(oth_ax2.get_xlim()[1] * 1.09, np.mean(oth_vals2),
                 "{:.2f} \u2192 {:.2f}".format(
                     np.mean(oth_vals2), (
                         (np.mean(oth_vals2) - np.mean(wt_vals2))
                         / (np.mean(mut_vals2) - np.mean(wt_vals2))
                        )
                    ),
                 size=11, ha='left', va='center')

    plt.tight_layout(pad=-0.4, w_pad=0.5, h_pad=0)
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

