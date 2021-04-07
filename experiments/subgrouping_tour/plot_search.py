
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_tour import base_dir
from ..utilities.metrics import calc_conf
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.labels import get_cohort_label

import os
import argparse
import bz2
from pathlib import Path
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
plot_dir = os.path.join(base_dir, 'plots', 'search')


def plot_original_concordance(auc_df, orig_aucs, conf_vals, orig_confs,
                              pheno_dict, args):
    fig, (auc_ax, conf_ax) = plt.subplots(figsize=(17, 8), nrows=1, ncols=2)

    use_aucs = orig_aucs['mean'][[not isinstance(mtype, RandomType)
                                  and (get_subtype(mtype)
                                       & copy_mtype).is_empty()
                                  for mtype in orig_aucs.index]]

    auc_min = 0.47
    for mtype, auc_val in auc_df['mean'].iteritems():
        cur_gene, cur_subt = tuple(mtype.subtype_iter())[0]

        orig_mtype = MuType({
            ('Gene', cur_gene): {('Scale', 'Point'): cur_subt}})

        if orig_mtype in use_aucs.index:
            plt_clr = choose_label_colour(cur_gene)
            plt_sz = 503 * np.mean(pheno_dict[mtype])

            auc_min = min(auc_min,
                          use_aucs[orig_mtype] - 0.023, auc_val - 0.023)

            auc_ax.scatter(use_aucs[orig_mtype], auc_val, s=plt_sz,
                           c=[plt_clr], edgecolor='none', alpha=0.19)

            auc_mean = (use_aucs[orig_mtype] + auc_val) / 2
            conf_sc = calc_conf(orig_confs[orig_mtype], conf_vals[mtype])

            conf_ax.scatter(auc_mean, conf_sc, s=plt_sz, c=[plt_clr],
                            edgecolor='none', alpha=0.19)

    auc_lims = auc_min, 1 + (1 - auc_min) / 131
    conf_lims = -0.007, 1.007
    auc_ax.grid(alpha=0.41, linewidth=0.9)
    conf_ax.grid(alpha=0.41, linewidth=0.9)

    auc_ax.plot(auc_lims, [0.5, 0.5],
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    auc_ax.plot([0.5, 0.5], auc_lims,
                color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    conf_ax.plot([0.5, 0.5], conf_lims,
                 color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    conf_ax.plot(conf_lims, [0.5, 0.5],
                 color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)
    auc_ax.plot(auc_lims, auc_lims,
                color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    auc_ax.plot(auc_lims, [1, 1], color='black', linewidth=1.3, alpha=0.89)
    auc_ax.plot([1, 1], auc_lims, color='black', linewidth=1.3, alpha=0.89)
    conf_ax.plot([1, 1], conf_lims, color='black', linewidth=1.3, alpha=0.89)
    conf_ax.plot(auc_lims, [1, 1], color='black', linewidth=1.3, alpha=0.89)
    conf_ax.plot(auc_lims, [0, 0], color='black', linewidth=1.3, alpha=0.89)

    auc_ax.set_xlabel("AUC in Original Experiment",
                      size=21, weight='semibold')
    auc_ax.set_ylabel("AUC in Enlarged\nSubgrouping Search Experiment",
                      size=21, weight='semibold')

    conf_ax.set_xlabel("AUC in Original Experiment",
                       size=21, weight='semibold')
    conf_ax.set_ylabel("Sub-Sampled\nAUC Comparison Confidence",
                       size=21, weight='semibold')

    auc_ax.set_xlim(auc_lims)
    auc_ax.set_ylim(auc_lims)
    conf_ax.set_xlim(auc_lims)
    conf_ax.set_ylim(conf_lims)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "original-concordance_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_original_comparison(auc_df, orig_aucs, pheno_dict, args):
    use_aucs = orig_aucs['mean'][[
        not isinstance(mtype, RandomType)
        and (get_subtype(mtype) & copy_mtype).is_empty()
        for mtype in orig_aucs.index
        ]]

    new_dict = dict()
    plt_min = use_aucs.min()
    orig_gby = use_aucs.groupby(get_label)

    for gene, auc_vec in auc_df['mean'].groupby(get_label):
        orig_indx = {
            mtype: MuType({('Gene', gene): {
                ('Scale', 'Point'): tuple(mtype.subtype_iter())[0][1]}})
            for mtype in auc_vec.index
            }

        new_mtypes = {mtype for mtype, orig_mtype in orig_indx.items()
                      if orig_mtype not in orig_gby.get_group(gene).index}

        if len(new_mtypes) >= 20:
            new_dict[gene] = new_mtypes

    fig, axarr = plt.subplots(figsize=(0.5 + 1.5 * len(new_dict), 7),
                              nrows=1, ncols=len(new_dict))

    for i, (gene, new_mtypes) in enumerate(tuple(sorted(
            new_dict.items(),
            key=lambda x: auc_df['mean'][list(x[1])].max(), reverse=True
            ))[:15]):
        axarr[i].set_title(gene, size=19, weight='bold')
        plt_clr = choose_label_colour(gene)

        base_mtypes = {'Orig': MuType({('Gene', gene): pnt_mtype}),
                       'Tour': MuType({('Gene', gene): None})}
        orig_df = pd.DataFrame({'AUC': orig_gby.get_group(gene),
                                'Type': 'Orig'})

        orig_df['SigCV'] = [
            (np.array(orig_aucs['CV'][mtype])
             > np.array(orig_aucs['CV'][base_mtypes['Orig']])).all()
            for mtype in orig_df.index
            ]

        tour_df = pd.DataFrame({'AUC': auc_df['mean'][list(new_mtypes)],
                                'Type': 'Tour'})
        tour_df['SigCV'] = [
            (np.array(auc_df['CV'][mtype])
             > np.array(auc_df['CV'][base_mtypes['Tour']])).all()
            for mtype in tour_df.index
            ]

        plt_df = pd.concat([orig_df, tour_df])
        if (plt_df.Type == 'Orig').sum() > 10:
            sns.violinplot(x=plt_df.Type, y=plt_df.AUC, ax=axarr[i],
                           palette=[plt_clr], cut=0, order=['Orig', 'Tour'],
                           linewidth=0, width=0.93)

        else:
            for j, sctr_val in enumerate(np.random.randn(plt_df.shape[0])):
                if plt_df.Type.iloc[j] == 'Orig':
                    plt_x = sctr_val / 7.1
                else:
                    plt_x = 1 + sctr_val / 7

                axarr[i].scatter(plt_x, plt_df.AUC.iloc[j], s=37, alpha=0.29,
                                 facecolor=plt_clr, edgecolor='none')

        axarr[i].plot([-0.6, 1.6], [1, 1],
                      color='black', linewidth=1.7, alpha=0.79)
        axarr[i].plot([-0.6, 1.6], [0.5, 0.5],
                      color='black', linewidth=1.3, linestyle=':', alpha=0.61)

        axarr[i].get_children()[0].set_alpha(0.53)
        axarr[i].get_children()[2].set_alpha(0.53)

        axarr[i].set_xlabel('')
        axarr[i].set_xticklabels([])
        axarr[i].grid(axis='x', linewidth=0)
        axarr[i].grid(axis='y', linewidth=0.5)

        axarr[i].text(0.37, 0, "n={}".format((plt_df.Type == 'Orig').sum()),
                      size=12, rotation=45, ha='right', va='center',
                      transform=axarr[i].transAxes)
        axarr[i].text(5 / 6, 0, "n={}".format((plt_df.Type == 'Tour').sum()),
                      size=12, rotation=45, ha='right', va='center',
                      transform=axarr[i].transAxes)

        best_aucs = plt_df.groupby('Type')['AUC'].max()
        best_types = plt_df.groupby('Type')['AUC'].idxmax()
        sig_stats = plt_df.groupby('Type')['SigCV'].any()

        for j, (type_lbl, sig_stat) in enumerate(sig_stats.iteritems()):
            if sig_stat:
                lbl_pos = min(0.963, best_aucs[type_lbl] - (1 - plt_min) / 83)
                axarr[i].text(j, lbl_pos, '*', size=19,
                              ha='center', va='bottom', weight='semibold')

        tour_stat = np.array([
            (np.array(orig_aucs['CV'][best_types['Orig']])
             < np.array(auc_df['CV'][tour_mtype])).all()
            for tour_mtype in tour_df.index
            ]).any()

        if tour_stat:
            axarr[i].text(0.5, 0.977, '*',
                          size=27, ha='center', va='top', weight='semibold',
                          transform=axarr[i].transAxes)

        if i == 0:
            axarr[i].set_ylabel('AUC', size=21, weight='semibold')
        else:
            axarr[i].set_yticklabels([])
            axarr[i].set_ylabel('')

        axarr[i].set_xlim([-0.6, 1.6])
        axarr[i].set_ylim([plt_min - 0.07, 1.007])

    axarr[-1].text(0.9, 0.07, get_cohort_label(args.cohort),
                   size=20, style='italic', ha='right', va='bottom',
                   transform=axarr[-1].transAxes)

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "original-comparison_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_search',
        description="Plots outcome of a larger subgrouping search."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort")
    parser.add_argument('search_params', type=str)
    parser.add_argument('mut_lvls', type=str)
    parser.add_argument('classif', help="a mutation classifier")

    args = parser.parse_args()
    out_dir = os.path.join(base_dir,
                           '__'.join([args.expr_source, args.cohort]))

    out_files = {
        out_lbl: os.path.join(
            out_dir, "out-{}__{}__{}__{}.p.gz".format(
                out_lbl, args.search_params, args.mut_lvls, args.classif)
            )
        for out_lbl in ['pred', 'pheno', 'aucs', 'conf']
        }

    if not os.path.isfile(out_files['conf']):
        raise ValueError("No experiment output found for these parameters!")

    orig_dir = os.path.join(Path(base_dir).parent, "subgrouping_test")
    orig_datas = [
        out_file.parts[-2:] for out_file in Path(orig_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    orig_list = pd.DataFrame([
        {'Samps': int(orig_data[0].split('__samps-')[1]),
         'Levels': '__'.join(orig_data[1].split(
             'out-trnsf__')[1].split('__')[:-1])}
        for orig_data in orig_datas
        ])

    if orig_list.shape[0] == 0:
        raise ValueError("No subgrouping testing experiment output found "
                         "for these parameters!")

    orig_use = orig_list.groupby('Levels')['Samps'].min()
    orig_phns = dict()
    orig_aucs = pd.DataFrame(dtype=float)
    orig_confs = pd.Series(dtype=object)

    for lvls, ctf in orig_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(orig_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            orig_phns.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(orig_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            orig_aucs = orig_aucs.append(pickle.load(f))

        with bz2.BZ2File(os.path.join(orig_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            orig_confs = orig_confs.append(pickle.load(f))

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_list = []
    for out_lbl in ['pheno', 'aucs', 'conf']:
        with bz2.BZ2File(out_files[out_lbl], 'r') as f:
            out_list += [pickle.load(f)]

    phn_dict, auc_dfs, conf_df = out_list
    for auc_df in auc_dfs.values():
        assert auc_df.index.isin(phn_dict).all()

    plot_original_concordance(auc_dfs['Chrm'], orig_aucs,
                              conf_df.Chrm, orig_confs, phn_dict, args)
    plot_original_comparison(auc_dfs['Chrm'], orig_aucs, phn_dict, args)


if __name__ == '__main__':
    main()

