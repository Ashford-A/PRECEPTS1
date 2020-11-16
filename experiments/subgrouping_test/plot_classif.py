
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir, train_cohorts
from .utils import choose_cohort_colour
from ..utilities.misc import choose_label_colour, get_distr_transform
from ..utilities.labels import get_cohort_label, get_fancy_label
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'classif')


def plot_gene_accuracy(auc_vals, pheno_dict, args):
    fig, axarr = plt.subplots(figsize=(13, 8), nrows=2, ncols=8)

    use_aucs = auc_vals[[not isinstance(mtype, RandomType)
                         and tuple(mtype.subtype_iter())[0][1] == pnt_mtype
                         for _, _, mtype in auc_vals.index]]

    plt_cohs = sorted(set((src, coh) for src, coh, _ in use_aucs.index))
    top_genes = use_aucs.groupby(['Source', 'Cohort']).apply(
        lambda aucs: aucs.sort_values().index.get_level_values('Mtype')[-8:])

    plot_dict = {(src, coh): dict() for src, coh in plt_cohs}
    line_dict = {(src, coh): dict() for src, coh in plt_cohs}
    plt_ylims = use_aucs.min() - 0.01, 1.003

    for (src, coh, mtype), auc_val in use_aucs.iteritems():
        ax_indx = plt_cohs.index((src, coh))
        cur_ax = axarr[ax_indx // 8, ax_indx % 8]
        cur_gene = tuple(mtype.label_iter())[0]

        # jitter the plotted point on the horizontal plane, scale the point
        # according to how frequently the gene was mutated in the cohort
        plt_x = 0.5 + np.random.randn() / 7.9
        base_size = np.mean(pheno_dict[src, coh][mtype])
        pnt_size = 0.91 * base_size ** 0.5
        use_clr = choose_label_colour(cur_gene)
        line_dict[src, coh][plt_x, auc_val] = dict(c=use_clr)

        # if classification performance was good enough, add a gene name label
        if auc_val > 0.7 and mtype in top_genes.loc[src, coh]:
            plot_dict[src, coh][plt_x, auc_val] = pnt_size, (cur_gene, '')
        else:
            plot_dict[src, coh][plt_x, auc_val] = pnt_size, ('', '')

        cur_ax.scatter(plt_x, auc_val, s=311 * base_size,
                       c=[use_clr], alpha=0.37, edgecolors='none')

    for i, (ax, (src, coh)) in enumerate(zip(axarr.flatten(), plt_cohs)):
        if i not in {0, 8}:
            ax.set_yticklabels([])

        ax.text(0.5, -0.01, get_cohort_label(coh).replace("(", "\n("),
                size=18, weight='semibold', ha='center', va='top',
                transform=ax.transAxes)

        ax.plot([0, 1], [1, 1], color='black', linewidth=2.7, alpha=0.89)
        ax.plot([0, 1], [0.5, 0.5],
                color='black', linewidth=1.7, linestyle=':', alpha=0.71)

        ax.set_xticklabels([])
        ax.grid(axis='x', linewidth=0)
        ax.grid(axis='y', linewidth=0.73, alpha=0.41)

    for ax, (src, coh) in zip(axarr.flatten(), plt_cohs):
        lbl_pos = place_scatter_labels(
            plot_dict[src, coh], ax,
            plt_lims=[plt_ylims] * 2,
            plc_lims=[[use_aucs.min() + 0.01, 0.99]] * 2,
            plt_type='scatter', font_size=9, seed=args.seed,
            line_dict=line_dict[src, coh], linewidth=0.7, alpha=0.37
            )

        ax.set_xlim([0, 1])
        ax.set_ylim(plt_ylims)

    fig.text(-0.023, 0.5, 'Gene-Wide Classifier AUC', size=23,
             ha='center', va='center', rotation=90, weight='semibold')

    fig.tight_layout(w_pad=0.3, h_pad=2.9)
    fig.savefig(
        os.path.join(plot_dir, args.classif, "gene-accuracy.svg"),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_gene_results(auc_vals, conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
        for _, _, mtype in auc_vals.index
        ]]

    plot_dict = dict()
    for (src, coh, gene), auc_vec in use_aucs.groupby(
            lambda x: (x[0], x[1], tuple(x[2].label_iter())[0])):
        if len(auc_vec) > 1 and auc_vec.max() > 0.68:
            base_mtype = MuType({('Gene', gene): pnt_mtype})

            base_indx = auc_vec.index.get_loc((src, coh, base_mtype))
            _, _, best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()

            if auc_vec[src, coh, best_subtype] > 0.68:
                conf_sc = np.greater.outer(
                    conf_vals[src, coh, best_subtype],
                    conf_vals[src, coh, base_mtype]
                    ).mean()

                if conf_sc > 0.77:
                    auc_tupl = auc_vec[src, coh, best_subtype], conf_sc
                    coh_lbl = get_cohort_label(coh)
                    use_clr = choose_cohort_colour(coh)

                    base_size = np.mean(pheno_dict[src, coh][base_mtype])
                    best_prop = np.mean(pheno_dict[src, coh][best_subtype])
                    best_prop /= base_size
                    plt_size = 0.023 * base_size ** 0.5

                    plot_dict[auc_tupl] = [plt_size, (gene, coh_lbl)]
                    auc_bbox = (auc_tupl[0] - plt_size / 2,
                                auc_tupl[1] - plt_size / 2,
                                plt_size, plt_size)

                    pie_ax = inset_axes(ax, width='100%', height='100%',
                                        bbox_to_anchor=auc_bbox,
                                        bbox_transform=ax.transData,
                                        axes_kwargs=dict(aspect='equal'),
                                        borderpad=0)

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.19, 0],
                               colors=[use_clr + (0.83,), use_clr + (0.23,)],
                               wedgeprops=dict(edgecolor='black',
                                               linewidth=5 / 13))

    ax.grid(linewidth=0.83, alpha=0.41)
    ax.tick_params(pad=7.3)
    ax.plot([0.65, 1.0005], [1, 1], color='black', linewidth=1.7, alpha=0.89)
    ax.plot([1, 1], [0.59, 1.0005], color='black', linewidth=1.7, alpha=0.89)

    ax.set_xlabel("AUC of Best Found Subgrouping",
                  size=23, weight='semibold')
    ax.set_ylabel("Down-Sampled AUC Confidence\nAgainst Gene-Wide Task",
                  size=23, weight='semibold')

    lbl_pos = place_scatter_labels(plot_dict, ax,
                                   plt_lims=[[0.68, 1], [0.77, 1]],
                                   plc_lims=[[0.69, 0.991], [0.78, 0.991]],
                                   font_size=11, seed=args.seed,
                                   c='black', linewidth=0.71, alpha=0.61)

    ax.set_xlim([0.68, 1.001])
    ax.set_ylim([0.77, 1.001])

    fig.savefig(
        os.path.join(plot_dir, args.classif, "gene-results.svg"),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def plot_tuning_profile(acc_df, use_clf, auc_vals, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

    use_aucs = auc_vals.round(4)
    auc_bins = pd.qcut(
        use_aucs.values.flatten(), q=[0., 0.5, 0.75, 0.8, 0.85, 0.9,
                                      0.92, 0.94, 0.96, 0.98, 0.99, 1.],
        precision=5
        ).categories

    par_fxs = {par_name: get_distr_transform(tune_distr)
               for par_name, tune_distr in use_clf.tune_priors}

    use_cohs = sorted(set(acc_df.index.get_level_values('Cohort')))
    coh_clrs = dict(zip(use_cohs,
                        sns.color_palette("muted", n_colors=len(use_cohs))))
    plt_min = 0.47

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        plot_df = pd.DataFrame([])
        for (src, coh, mtype), acc_vals in acc_df.iterrows():
            par_df = pd.DataFrame.from_records([
                pd.Series({par_fxs[par_name](pars[par_name]): avg_val
                           for pars, avg_val in zip(par_ols, avg_ols)})
                for par_ols, avg_ols in zip(
                    acc_vals['par'], acc_vals['avg'])
                ]).quantile(q=0.25).reset_index()

            # note that we take the maximum of AUCs for finding the bin due
            # to (rare) cases where RandomTypes get duplicated
            par_df.columns = ['par', 'auc']
            par_df['auc_bin'] = auc_bins.get_loc(
                use_aucs[src, coh, mtype].max())
            plot_df = plot_df.append(par_df)

        for auc_bin, bin_vals in plot_df.groupby('auc_bin'):
            plot_vals = bin_vals.groupby('par').mean()
            plt_min = min(plt_min, plot_vals.auc.min() - 0.03)
            ax.plot(plot_vals.index, plot_vals.auc)

        ax.set_xlim((2 * par_fxs[par_name](tune_distr[0])
                     - par_fxs[par_name](tune_distr[1])),
                    (2 * par_fxs[par_name](tune_distr[-1])
                     - par_fxs[par_name](tune_distr[-2])))
        ax.set_ylim(plt_min, 1 + (1 - plt_min) / 91)

        ax.tick_params(labelsize=19)
        ax.set_xlabel("Tested {} Value".format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, "Aggregate AUC", ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, args.classif, "tuning-profile.svg"),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_classif',
        description="Summarizes results for a given subgrouping classifier."
        )

    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument(
        '--seed', type=int,
        help="random seed for fixing plot elements like label placement"
        )

    # parse command line arguments, find experiments matching the given
    # criteria that have run to completion
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "*__*__samps-*", "out-trnsf__*__{}.p.gz".format(args.classif)))
        ]

    out_list = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "out-trnsf__")[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby('Cohort').filter(
            lambda outs: 'Consequence__Exon' in set(outs.Levels))

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby(['Source', 'Cohort', 'Levels'])['Samps'].min()
    out_use = out_use[out_use.index.get_level_values(
        'Cohort').isin(train_cohorts)]

    out_use = out_use.loc[
        ~((out_use.index.get_level_values('Cohort') == 'BRCA_LumA')
          & (out_use.index.get_level_values('Source') == 'toil__gns'))
        ]
    os.makedirs(os.path.join(plot_dir, args.classif), exist_ok=True)

    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()
    acc_dict = dict()
    out_clf = None

    for (src, coh, lvls), ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(src, coh, ctf)

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-pheno__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as f:
            phns = pickle.load(f)

        if (src, coh) in phn_dict:
            phn_dict[src, coh].update(phns)
        else:
            phn_dict[src, coh] = phns

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_vals = pickle.load(f)['mean']

        auc_vals.index = pd.MultiIndex.from_product(
            [[src], [coh], auc_vals.index],
            names=('Source', 'Cohort', 'Mtype')
            )
        auc_dict[src, coh, lvls] = auc_vals

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-conf__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as f:
            conf_vals = pickle.load(f)

        conf_vals = conf_vals[[not isinstance(mtype, RandomType)
                               for mtype in conf_vals.index]]

        conf_vals.index = pd.MultiIndex.from_product(
            [[src], [coh], conf_vals.index],
            names=('Source', 'Cohort', 'Mtype')
            )
        conf_dict[src, coh, lvls] = conf_vals

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-tune__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as f:
            (_, _, acc_vals, cur_clf) = pickle.load(f)

        if out_clf is not None:
            if cur_clf != out_clf:
                raise ValueError("Mismatching classifiers in subvariant "
                                 "testing experment output!")

        else:
            out_clf = cur_clf

        acc_vals.index = pd.MultiIndex.from_product(
            [[src], [coh], acc_vals.index],
            names=('Source', 'Cohort', 'Mtype')
            )
        acc_dict[coh, lvls] = acc_vals

    # consolidate filtered experiment output in data frames
    auc_vals = pd.concat(auc_dict.values()).sort_index()
    conf_vals = pd.concat(conf_dict.values()).sort_index()
    acc_df = pd.concat(acc_dict.values()).sort_index()

    # create the plots
    plot_gene_accuracy(auc_vals, phn_dict, args)
    plot_gene_results(auc_vals, conf_vals, phn_dict, args)
    plot_tuning_profile(acc_df, out_clf, auc_vals, args)


if __name__ == "__main__":
    main()

