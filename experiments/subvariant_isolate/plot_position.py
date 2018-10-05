
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'position')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
import argparse

from HetMan.experiments.subvariant_isolate.setup_isolate import load_cohort
from dryadic.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output, simil_cmap

import numpy as np
import pandas as pd
from functools import reduce
from operator import or_, and_
from scipy.stats import ks_2samp
import re

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import matplotlib.colors as colors
import matplotlib.cm as cmx
color_norm = colors.Normalize(vmin=-1., vmax=2.)
use_cmap = cmx.ScalarMappable(norm=color_norm, cmap=simil_cmap).to_rgba


def plot_mtype_projection(prob_series, plot_mtypes, args, cdata):
    fig, ax = plt.subplots(figsize=(6, 11))

    all_mtype = MuType(cdata.train_mut.allkey())
    all_pheno = np.array(cdata.train_pheno(all_mtype))
    none_vals = prob_series[~all_pheno].tolist()

    use_mtypes = set(plot_mtypes) | set([prob_series.name])
    phenos_list = {mtypes: None for mtypes in use_mtypes}

    for mtypes in use_mtypes:
        cur_phenos = [np.array(cdata.train_pheno(mtype)) for mtype in mtypes]
        and_pheno = reduce(and_, cur_phenos)

        phenos_list[mtypes] = and_pheno & ~(
            np.array(cdata.train_pheno(all_mtype - reduce(or_, mtypes)))
            | (reduce(or_, cur_phenos) & ~and_pheno)
            )

    plot_list = sorted([(mtypes, prob_series[pheno].tolist())
                        for mtypes, pheno in phenos_list.items()],
                       key=lambda x: np.median(x[1]))
    cur_vals = plot_list.pop(
        [mtypes for mtypes, _ in plot_list].index(prob_series.name))

    plot_list = [cur_vals] + plot_list[::-1]
    plot_list += [["{} Wild-Type".format(args.gene), none_vals]]
    phenos_list = [phenos_list[mtypes] for mtypes, _ in plot_list[:-1]]
    phenos_list += [~all_pheno]

    cur_mean = np.mean(plot_list[0][1])
    none_mean = np.mean(plot_list[-1][1])

    val_clrs = {i: use_cmap((np.mean(vals) - none_mean)
                            / (cur_mean - none_mean))
                for i, (_, vals) in tuple(enumerate(plot_list))[1:-1]}
    val_clrs[0] = '0.39'
    val_clrs[len(plot_list) - 1] = '0.91'

    plot_df = pd.concat([pd.Series({mtype: vals})
                         for mtype, vals in plot_list])
    sns.boxplot(data=plot_df, palette=val_clrs, width=7./13,
                linewidth=0.5, showfliers=False, saturation=1.)

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .73))

    for i, phenos in enumerate(phenos_list):
        k = -1

        for j, pheno in enumerate(phenos):
            if pheno:
                k += 1

                if phenos_list[0][j]:
                    mrk_shape = '*'
                    mrk_size = 49

                else:
                    mrk_shape = 'o'
                    mrk_size = 16

                ax.scatter(i + np.random.randn() / 7, plot_list[i][1][k],
                           c=use_cmap((plot_list[i][1][k] - none_mean)
                                      / (cur_mean - none_mean)),
                           marker=mrk_shape, s=mrk_size,
                           alpha=0.37, edgecolors='black')

    plt.axhline(color='0.23', y=cur_mean, linestyle='--',
                linewidth=1.8, alpha=0.57)
    plt.axhline(color='0.59', y=none_mean, linestyle='--',
                linewidth=1.8, alpha=0.57)

    xlabs = [mtypes if isinstance(mtypes, str)
             else 'ONLY {}'.format(str(mtypes[0])) if len(mtypes) == 1
             else ' AND '.join(str(mtype) for mtype in sorted(mtypes))
             for mtypes in plot_df.index]

    xlabs = [xlab.replace('Point:', '') for xlab in xlabs]
    xlabs = [xlab.replace('Copy:', '') for xlab in xlabs]
    plt.xticks(tuple(range(len(plot_list))), xlabs,
               rotation=29, ha='right', size=11)
    plt.yticks(size=10)

    ax.tick_params(axis='y', length=7, width=2)
    plt.ylim((prob_series.min() * 1.03, prob_series.max() * 1.03))
    plt.ylabel('{} Decision Function'.format(args.classif),
               fontsize=19, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, '{}_{}__{}'.format(args.cohort, args.gene, args.mut_levels),
        "singleton-projection_{}__{}_samps_{}.png".format(
            "".join([c for c in xlabs[0] if re.match(r'\w', c)]),
            args.classif, args.samp_cutoff)
            ),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def plot_mtype_positions(prob_series, args, cdata):
    kern_bw = (np.max(prob_series) - np.min(prob_series)) / 29

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 18),
                                   sharex=True, sharey=False,
                                   gridspec_kw={'height_ratios': [1, 3.41]})

    base_mtype = MuType({('Gene', args.gene): None})
    cur_mtype = MuType({('Gene', args.gene): prob_series.name})
    base_pheno = np.array(cdata.train_pheno(base_mtype))
    cur_pheno = np.array(cdata.train_pheno(cur_mtype))

    without_phenos = {
        mtype: np.array(cdata.train_pheno(mtype))
        for mtype in cdata.train_mut.branchtypes(min_size=args.samp_cutoff)
        if (mtype & base_mtype).is_empty()
        }

    within_mtypes = {MuType({('Gene', args.gene): mtype})
                     for mtype in cdata.train_mut[args.gene].combtypes(
                         comb_sizes=(1, 2), min_type_size=args.samp_cutoff)
                     if (mtype & prob_series.name).is_empty()}

    within_phenos = {mtype: np.array(cdata.train_pheno(mtype))
                     for mtype in within_mtypes}

    cur_diff = (np.mean(prob_series[cur_pheno])
                - np.mean(prob_series[~base_pheno]))

    sns.kdeplot(prob_series[~base_pheno], ax=ax1, cut=0,
                color='0.4', alpha=0.45, linewidth=2.8,
                bw=kern_bw, gridsize=250, shade=True,
                label='{} Wild-Type'.format(args.gene))
    sns.kdeplot(prob_series[cur_pheno], ax=ax1, cut=0,
                color=(0.267, 0.137, 0.482), alpha=0.45, linewidth=2.8,
                bw=kern_bw, gridsize=250, shade=True,
                label='{} Mutant'.format(prob_series.name))
    sns.kdeplot(prob_series[base_pheno & ~cur_pheno], ax=ax1, cut=0,
                color=(0.698, 0.329, 0.616), alpha=0.3, linewidth=1.0,
                bw=kern_bw, gridsize=250, shade=True,
                label='Other {} Mutants'.format(args.gene))

    ax1.set_ylabel('Density', size=23, weight='semibold')
    ax1.yaxis.set_tick_params(labelsize=14)

    without_tests = {
        mtype: {
            'pval': ks_2samp(prob_series[~base_pheno & ~pheno],
                             prob_series[~base_pheno & pheno]).pvalue,
            'diff': (np.mean(prob_series[~base_pheno & pheno])
                     - np.mean(prob_series[~base_pheno & ~pheno]))
            }
        for mtype, pheno in without_phenos.items()
        }

    without_tests = sorted(
        [(mtype, tests) for mtype, tests in without_tests.items()
         if tests['pval'] < 0.05 and tests['diff'] > 0],
        key=lambda x: x[1]['pval']
        )[:8]

    within_tests = {
        mtype: {
            'pval': ks_2samp(
                prob_series[base_pheno & ~cur_pheno & ~pheno],
                prob_series[base_pheno & ~cur_pheno & pheno]).pvalue,
            'diff': (np.mean(prob_series[base_pheno & ~cur_pheno & pheno])
                     - np.mean(prob_series[base_pheno & ~cur_pheno & ~pheno]))
            }
        for mtype, pheno in within_phenos.items()
        }

    within_tests = sorted(
        [(mtype, tests) for mtype, tests in within_tests.items()
         if tests['pval'] < 0.1],
        key=lambda x: x[1]['pval']
        )[:8]

    subtype_df = pd.concat(
        [pd.DataFrame({'Mtype': repr(mtype).replace(' WITH ', '\n'),
                       'Type': '{} Wild-Type'.format(args.gene),
                       'Scores': prob_series[~base_pheno
                                             & without_phenos[mtype]]})
         for mtype, tests in without_tests]
        + [pd.DataFrame(
            {'Mtype': repr(mtype).replace(
                'Gene IS {}'.format(args.gene), '').replace(' WITH ', '\n'),
                'Type': '{} Mutants'.format(args.gene),
                'Scores': prob_series[base_pheno & within_phenos[mtype]]}
            )
            for mtype, tests in within_tests]
        )

    plt_order = subtype_df.groupby(
        ['Mtype'])['Scores'].mean().sort_values().index
    subtype_df['Mtype'] = subtype_df['Mtype'].astype(
        'category').cat.reorder_categories(plt_order)

    sns.violinplot(
        data=subtype_df, x='Scores', y='Mtype', hue='Type',
        palette={'{} Wild-Type'.format(args.gene): '0.5',
                 '{} Mutants'.format(args.gene): (0.812, 0.518, 0.745)},
        alpha=0.3, linewidth=1.3, bw=kern_bw, dodge=False,
        cut=0, gridsize=500, legend=False
        )

    ax2.set_ylabel('Mutation Type', size=23, weight='semibold')
    ax2.yaxis.set_tick_params(labelsize=12)

    ax2.xaxis.set_tick_params(labelsize=18)
    ax2.set_xlabel('Inferred {} Score'.format(prob_series.name),
                   size=23, weight='semibold')

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir,
                     args.cohort, args.gene,
                     "{}_positions__{}_{}__{}__levels__{}.png".format(
                         re.sub('/|\.|:', '_', str(prob_series.name)),
                         args.cohort, args.gene, args.classif,
                         args.mut_levels
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot experiment results for given mutation classifier.')

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Form_base__Exon',
                        help='a set of mutation annotation levels')
    parser.add_argument('--samp_cutoff', default=20)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '{}_{}__{}'.format(args.cohort, args.gene,
                                                args.mut_levels)),
                exist_ok=True)

    cdata = load_cohort(args.cohort, [args.gene], args.mut_levels.split('__'))
    prob_df = load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.gene, args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        ).applymap(np.mean)

    singl_mtypes = [mtypes for mtypes in prob_df.index
                    if all(len(mtype.subkeys()) == 1 for mtype in mtypes)]

    for singl_mtype in singl_mtypes:
        plot_mtype_projection(prob_df.loc[[singl_mtype]].iloc[0, :],
                              singl_mtypes, args, cdata)
        #plot_mtype_positions(prob_df.loc[[singl_mtype]].iloc[0, :],
        #                     args, cdata)


if __name__ == '__main__':
    main()

