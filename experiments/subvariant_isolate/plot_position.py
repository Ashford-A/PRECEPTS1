
import os
import sys

if 'DATADIR' in os.environ:
    base_dir = os.path.join(os.environ['DATADIR'],
                            'HetMan', 'subvariant_isolate')
else:
    base_dir = os.path.dirname(__file__)

plot_dir = os.path.join(base_dir, 'plots', 'position')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])

from dryadic.features.mutations import MuType
from HetMan.experiments.subvariant_isolate.setup_isolate import load_cohort
from HetMan.experiments.subvariant_isolate.utils import compare_scores
from HetMan.experiments.utilities import load_infer_output, simil_cmap

import numpy as np
import pandas as pd

import argparse
from scipy.stats import mannwhitneyu
import re

from functools import reduce
from operator import or_, and_
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'

import matplotlib.colors as colors
import matplotlib.cm as cmx
color_norm = colors.Normalize(vmin=-1., vmax=2.)
use_cmap = cmx.ScalarMappable(norm=color_norm, cmap=simil_cmap).to_rgba


def plot_mtype_projection(prob_series, plot_mtypes, pheno_dict,
                          args, cdata, proj_tag='default'):
    fig, ax = plt.subplots(figsize=(8, 6))

    none_vals = prob_series[pheno_dict['Wild-Type']].tolist()
    plot_list = [(prob_series.name,
                  prob_series[pheno_dict[prob_series.name]].tolist())]

    plot_list += [('', [])] + sorted(
        [(mtypes, prob_series[pheno_dict[mtypes]].tolist())
         for mtypes in set(plot_mtypes) - {prob_series.name}],
        key=lambda x: np.median(x[1]), reverse=True
        ) + [('', [])] + [('Wild-Type', none_vals)]

    cur_mean = np.mean(plot_list[0][1])
    none_mean = np.mean(plot_list[-1][1])

    med_vals = [0] + [(np.mean(vals) - none_mean) / (cur_mean - none_mean)
                      for _, vals in plot_list[2:-2]] + [0]
    med_clrs = [use_cmap(val) for val in
                [med_vals[0]] + [0] + med_vals[2:-2] + [0] + [med_vals[-1]]]

    plot_df = pd.concat([pd.Series({lbl: vals}) for lbl, vals in plot_list])
    sns.boxplot(data=plot_df, palette=med_clrs, width=6./13,
                linewidth=0.61, showfliers=False, saturation=0.93)

    for i, (lbl, vals) in enumerate(x for x in plot_list if x[0]):
        if i == 0 or i == (len(ax.artists) - 1):
            ax.artists[i].set_linewidth(1.3)
            ax.artists[i].set_linestyle('--')

            for j in range(i * 5, (i + 1) * 5):
                ax.lines[j].set_linewidth(1.3)
                ax.lines[j].set_linestyle('--')

        else:
            r, g, b, a = ax.artists[i].get_facecolor()
            ax.artists[i].set_facecolor((r, g, b, .73))

        k = -1
        for j, pheno in enumerate(pheno_dict[lbl]):
            if pheno:
                k += 1

                if pheno_dict[prob_series.name][j]:
                    mrk_shape = 'P'
                    mrk_size = 21

                elif pheno_dict['Wild-Type'][j]:
                    mrk_shape = 'X'
                    mrk_size = 11

                else:
                    mrk_shape = 'o'
                    mrk_size = 8

                plt_x = i + np.random.randn() / 7.3
                if i > 0:
                    plt_x += 1
                if i == (len(ax.artists) - 1):
                    plt_x += 1
 
                trs_xy = ax.transData.transform([plt_x, vals[k]])
                edge_clr = '0.0'

                if ax.artists[i].contains_point(trs_xy):
                    edge_clr = np.clip(med_vals[i], 0, 2)
                    if edge_clr > 1:
                        edge_clr = 2 - edge_clr
                    edge_clr = '{:.2f}'.format(edge_clr ** 0.5)

                ax.scatter(plt_x, vals[k],
                           c=use_cmap((vals[k] - none_mean)
                                      / (cur_mean - none_mean)),
                           marker=mrk_shape, s=mrk_size, alpha=0.31,
                           edgecolors=edge_clr)

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
    plt.xticks(tuple(range(plot_df.shape[0])), xlabs,
               rotation=29, ha='right', size=11)

    plt.locator_params(axis='y', nbins=4)
    plt.yticks(size=10)

    ax.tick_params(axis='y', length=8, width=1.7)
    plt.ylim((prob_series.min() * 1.03, prob_series.max() * 1.03))
    plt.ylabel('{} Decision Function'.format(args.classif),
               fontsize=19, weight='semibold')
    fig.patch.set_facecolor('white')

    ax.legend([Line2D([], [], marker='P', linestyle='None', markersize=17,
                      markerfacecolor=use_cmap(1.9), markeredgecolor='0.1'),
               Line2D([], [], marker='X', linestyle='None', markersize=14,
                      markerfacecolor=use_cmap(-0.7), markeredgecolor='0.1'),
               Line2D([], [], marker='o', linestyle='None', markersize=14,
                      markerfacecolor=use_cmap(0.4), markeredgecolor='0.1')],
              ["{} mutated samples with {}\n(positive class)".format(
                  args.gene, xlabs[0]),
               "Wild-Type samples\n(negative class)",
               "remaining {} samples\n(held-out)".format(args.gene)],
              fontsize=9, loc=9, ncol=3,
              bbox_to_anchor=(0.5, 1.1)).get_frame().set_linewidth(0.0)

    plt.savefig(os.path.join(
        plot_dir, '{}_{}__{}'.format(args.cohort, args.gene, args.mut_levels),
        "mtype-projection_{}__{}_{}_samps_{}.png".format(
            "".join([c for c in xlabs[0] if re.match(r'\w', c)]),
            proj_tag, args.classif, args.samp_cutoff)
            ),
        dpi=300, bbox_inches='tight')

    plt.close()


def plot_pair_projection(prob_pair, plot_mtypes, pheno_dict,
                         args, cdata, proj_tag='default', proj_clrs=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    plot_dict = {'Wild-Type': prob_pair.iloc[:, pheno_dict['Wild-Type']]}
    all_mtype = MuType(cdata.train_mut.allkey())
    all_pheno = np.array(cdata.train_pheno(all_mtype))

    use_mtypes = [plot_mtype[1] if isinstance(plot_mtype[0], str)
                  else plot_mtype for plot_mtype in plot_mtypes]

    for mtypes in set(use_mtypes) | set(prob_pair.index):
        if mtypes in pheno_dict:
            plot_dict[mtypes] = prob_pair.iloc[:, pheno_dict[mtypes]]

        else:
            rest_pheno = np.array(cdata.train_pheno(
                all_mtype - reduce(or_, mtypes)))

            cur_phenos = [np.array(cdata.train_pheno(mtype))
                          for mtype in mtypes]
            and_pheno = reduce(and_, cur_phenos)

            plot_dict[mtypes] = prob_pair.iloc[
                :, and_pheno & ~(rest_pheno
                                 | (reduce(or_, cur_phenos) & ~and_pheno))
                ]

    plot_lbls = [
        plot_mtype if isinstance(plot_mtype, str)
        else plot_mtype[0] if isinstance(plot_mtype[0], str)
        else 'ONLY {}'.format(str(plot_mtype[0])) if len(plot_mtype) == 1
        else ' AND '.join(str(mtype) for mtype in sorted(plot_mtype))
        for plot_mtype in list(prob_pair.index) + ['Wild-Type'] + plot_mtypes
        ]

    lgnd_list = []
    use_clrs = {'Wild-Type': '0.55'}

    if proj_clrs is None:
        proj_pal = sns.color_palette('muted', n_colors=len(plot_mtypes) + 2)
        use_clrs = {**use_clrs,
                    **dict(zip(prob_pair.index + use_mtypes, proj_pal))}

    else:
        use_clrs = {**use_clrs, **proj_clrs}

    for base_mtypes in prob_pair.index:
        lgnd_list += [Line2D([], [], color=use_clrs[base_mtypes],
                             marker='o', alpha=0.49, linestyle='None',
                             markersize=15, markeredgecolor='white')]
 
        ax.scatter(plot_dict[base_mtypes].iloc[0, :],
                   plot_dict[base_mtypes].iloc[1, :],
                   s=23, c=use_clrs[base_mtypes], marker='o',
                   alpha=0.52, edgecolors='white')

    for proj_mtype in ['Wild-Type'] + use_mtypes:
        lgnd_list += [Patch(color=use_clrs[proj_mtype], alpha=0.41)]
 
        sns.kdeplot(plot_dict[proj_mtype].iloc[0, :],
                    plot_dict[proj_mtype].iloc[1, :],
                    cmap=sns.light_palette(use_clrs[proj_mtype],
                                           as_cmap=True),
                    shade=True, alpha=0.39, shade_lowest=False,
                    bw=0.12, gridsize=250, n_levels=3)

    ax.legend(lgnd_list, plot_lbls, fontsize=15, loc=8, ncol=2)
    plt.xlabel("Inferred {} Score".format(plot_lbls[0]),
               size=21, weight='semibold')
    plt.ylabel("Inferred {} Score".format(plot_lbls[1]),
               size=21, weight='semibold')

    plt.savefig(os.path.join(
        plot_dir, '{}_{}__{}'.format(args.cohort, args.gene, args.mut_levels),
        "singleton-projection_{}__{}__{}_samps_{}.png".format(
            "".join([c for c in str(prob_pair.index[0])
                     if re.match(r'\w', c)]),
            "".join([c for c in str(prob_pair.index[1])
                     if re.match(r'\w', c)]),
            args.classif, args.samp_cutoff)
            ),
        dpi=300, bbox_inches='tight')

    plt.close()


def plot_mtype_enrichment(prob_series, pheno_dict, args, cdata):
    fig, ax = plt.subplots(figsize=(8, 9))
    kern_bw = (np.max(prob_series) - np.min(prob_series)) / 29

    pnt_mtypes = {MuType({('Scale', 'Point'): mtype})
                  for mtype in cdata.train_mut['Point'].combtypes() | {None}}

    cna_mtypes = cdata.train_mut['Copy'].branchtypes()
    cna_mtypes |= {MuType({('Copy', ('HetGain', 'HomGain')): None})}
    cna_mtypes |= {MuType({('Copy', ('HetDel', 'HomDel')): None})}
    cna_mtypes = {MuType({('Scale', 'Copy'): mtype}) for mtype in cna_mtypes}

    all_mtype = MuType(cdata.train_mut.allkey())
    only_mtypes = {(mtype, ) for mtype in pnt_mtypes | cna_mtypes
                   if (len(mtype.get_samples(cdata.train_mut)
                           - (all_mtype - mtype).get_samples(cdata.train_mut))
                       > 1)}

    comb_mtypes = {
        (mtype1, mtype2)
        for mtype1, mtype2 in combn(pnt_mtypes | cna_mtypes, 2)
        if ((mtype1 & mtype2).is_empty() and (
            len((mtype1.get_samples(cdata.train_mut)
                 & mtype2.get_samples(cdata.train_mut))
                - (mtype1.get_samples(cdata.train_mut)
                   ^ mtype2.get_samples(cdata.train_mut))
                - (all_mtype - mtype1 - mtype2).get_samples(cdata.train_mut))
            > 1
            ))
        }

    test_mtypes = [
        mtypes for mtypes in only_mtypes | comb_mtypes | {prob_series.name}
        if ((len(prob_series.name) == 1 and len(mtypes) == 1
             and (prob_series.name[0] & mtypes[0]).is_empty())
            or (len(mtypes) == 2))
        ]

    none_vals = prob_series[pheno_dict['Wild-Type']].tolist()
    none_mean = np.mean(none_vals)
    none_list = [('Wild-Type', none_vals)]

    cur_vals = prob_series[pheno_dict[prob_series.name]].tolist()
    cur_mean = np.mean(cur_vals)
    cur_list = [(prob_series.name, cur_vals)]

    proj_dict = dict()
    pval_dict = dict()
    for mtypes in test_mtypes:
        if mtypes in pheno_dict:
            pheno = pheno_dict[mtypes]

        else:
            rest_pheno = np.array(cdata.train_pheno(
                all_mtype - reduce(or_, mtypes)))

            cur_phenos = [np.array(cdata.train_pheno(mtype))
                          for mtype in mtypes]
            and_pheno = reduce(and_, cur_phenos)

            pheno = and_pheno & ~(rest_pheno
                                  | (reduce(or_, cur_phenos) & ~and_pheno))

        proj_dict[mtypes] = prob_series[pheno]
        pval_dict[mtypes] = np.log10(min(
            mannwhitneyu(prob_series[pheno], none_vals).pvalue,
            mannwhitneyu(prob_series[pheno], cur_vals).pvalue
            ))

    proj_list = [(mtypes, proj_dict[mtypes])
                 for mtypes, _ in sorted(pval_dict.items(),
                                         key=lambda x: x[1])[:10]]

    med_clrs = ['1.0'] + ['0.0']
    med_clrs += [use_cmap((np.mean(vals) - none_mean)
                          / (cur_mean - none_mean))
                 for _, vals in proj_list] + ['0.0'] + ['1.0']

    plot_df = pd.concat([pd.Series({lbl: vals}) for lbl, vals in
                         cur_list + [('', [])] + proj_list
                         + [('', [])] + none_list])

    sns.violinplot(data=plot_df, palette=med_clrs,
                   width=11./14, alpha=0.49, linewidth=1.9, saturation=0.93,
                   bw=kern_bw, gridsize=500, cut=0)

    xlabs = [mtypes if isinstance(mtypes, str)
             else 'ONLY {}'.format(str(mtypes[0])) if len(mtypes) == 1
             else ' AND '.join(str(mtype) for mtype in sorted(mtypes))
             for mtypes in plot_df.index]

    xlabs = [xlab.replace('Point:', '') for xlab in xlabs]
    xlabs = [xlab.replace('Copy:', '') for xlab in xlabs]
    plt.xticks(tuple(range(plot_df.shape[0])), xlabs,
               rotation=29, ha='right', size=11)

    plt.locator_params(axis='y', nbins=4)
    plt.yticks(size=10)

    plt.savefig(os.path.join(
        plot_dir, '{}_{}__{}'.format(args.cohort, args.gene, args.mut_levels),
        "mtype-enrichment_{}__{}_samps_{}.png".format(
            "".join([c for c in xlabs[0] if re.match(r'\w', c)]),
            args.classif, args.samp_cutoff)
            ),
        dpi=300, bbox_inches='tight')

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
    infer_df = load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.gene, args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        )

    prob_df = infer_df.applymap(np.mean)
    singl_mtypes = {mtypes for mtypes in prob_df.index
                    if all(len(mtype.subkeys()) == 1 for mtype in mtypes)}
    pheno_dict, auc_list, _ = compare_scores(infer_df, cdata,
                                             get_similarities=False)

    for singl_mtype in singl_mtypes:
        plot_mtype_projection(prob_df.loc[[singl_mtype]].iloc[0, :],
                              singl_mtypes, pheno_dict,
                              args, cdata, proj_tag='singleton')

        plot_mtype_enrichment(prob_df.loc[[singl_mtype]].iloc[0, :],
                              pheno_dict, args, cdata)

    gain_mtype = (MuType({('Scale', 'Copy'): {('Copy', 'HomGain'): None}}), )
    if gain_mtype not in prob_df.index:
        gain_mtype = (MuType({
            ('Scale', 'Copy'): {('Copy', ('HomGain', 'HetGain')): None}}), )

    allpnt_mtype = MuType({('Scale', 'Point'): None})
    pnt_mtypes = {
        mtypes for mtypes in singl_mtypes - {(allpnt_mtype, )}
        if len(mtypes) == 1 and not (mtypes[0] & allpnt_mtype).is_empty()
        }

    pnt_scores = sorted(
        [(mtypes, np.sum(pheno_dict[mtypes]) * (1 - auc_list[mtypes]))
         for mtypes in pnt_mtypes], key=lambda x: x[1]
        )

    pnt_mtype = pnt_scores.pop(0)[0]
    pnt_str = str(pnt_mtype[0]).split(':')[-1]

    use_mtypes = [
        (MuType({('Scale', 'Copy'): {('Copy', 'HetGain'): None}}), ),
        ('Only Not {}'.format(pnt_str), (MuType({
            ('Scale', 'Point'): cdata.train_mut['Point'].allkey()})
            - pnt_mtype[0], )),
        ('Mutation and Gain', (MuType({
            ('Scale', 'Copy'): {('Copy', ('HomGain', 'HetGain')): None}}),
            allpnt_mtype))
        ]
 
    use_clrs = {gain_mtype: '#9B5500', pnt_mtype: '#03314C',
                use_mtypes[0]: '#774200', use_mtypes[1][1]: '#044063',
                use_mtypes[2][1]: '#4B004E'}

    plot_pair_projection(prob_df.loc[[gain_mtype, pnt_mtype]],
                         use_mtypes, pheno_dict, args, cdata,
                         proj_clrs=use_clrs)


if __name__ == '__main__':
    main()

