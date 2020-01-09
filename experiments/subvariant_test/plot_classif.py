
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'classif')

from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, train_cohorts)
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.plot_aucs import place_labels
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_test.plot_gene import (
    get_cohort_label, choose_cohort_colour)
from HetMan.experiments.variant_baseline.plot_tuning import detect_log_distr

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_gene_results(auc_vals, conf_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    use_aucs = auc_vals[[
        not isinstance(mtype, RandomType)
        and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()
        for _, mtype in auc_vals.index
        ]]

    pnt_dict = dict()
    clr_dict = dict()

    for (coh, gene), auc_vec in use_aucs.groupby(
            lambda x: (x[0], x[1].get_labels()[0])):

        if len(auc_vec) > 1 and auc_vec.max() > 0.68:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc((coh, base_mtype))

            _, best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc((coh, best_subtype))

            if auc_vec[best_indx] > 0.68:
                conf_sc = np.greater.outer(conf_vals[coh, best_subtype],
                                           conf_vals[coh, base_mtype]).mean()

                if conf_sc > 0.77:
                    clr_dict[gene, coh] = choose_cohort_colour(coh)

                    base_size = 0.23 * np.mean(pheno_dict[coh][base_mtype])
                    best_prop = 0.23 * np.mean(
                        pheno_dict[coh][best_subtype]) / base_size

                    pnt_dict[auc_vec[best_indx], conf_sc] = (
                        base_size ** 0.53, (gene, coh))

                    pie_ax = inset_axes(
                        ax, width=base_size ** 0.5, height=base_size ** 0.5,
                        bbox_to_anchor=(auc_vec[best_indx], conf_sc),
                        bbox_transform=ax.transData, loc=10,
                        axes_kwargs=dict(aspect='equal'), borderpad=0
                        )

                    pie_ax.pie(x=[best_prop, 1 - best_prop],
                               explode=[0.19, 0],
                               colors=[clr_dict[gene, coh] + (0.83,),
                                       clr_dict[gene, coh] + (0.23,)],
                               wedgeprops=dict(edgecolor='black',
                                               linewidth=4 / 13))

    lbl_pos = place_labels(pnt_dict, lims=(0.76, 1), lbl_dens=0.89)
    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                pnt_dict[pnt_x, pnt_y][1][0],
                size=13, ha=pos[1], va='bottom')
        ax.text(pos[0][0], pos[0][1] - 700 ** -1,
                get_cohort_label(pnt_dict[pnt_x, pnt_y][1][1]),
                size=9, ha=pos[1], va='top')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]
        ln_lngth = np.sqrt((x_delta ** 2) + (y_delta ** 2))

        # if the label is sufficiently far away from its point...
        if ln_lngth > (0.021 + pnt_dict[pnt_x, pnt_y][0] / 31):
            use_clr = clr_dict[pnt_dict[pnt_x, pnt_y][1]]
            pnt_gap = pnt_dict[pnt_x, pnt_y][0] / (29 * ln_lngth)
            lbl_gap = 0.006 / ln_lngth

            ax.plot([pnt_x - pnt_gap * x_delta,
                     pos[0][0] + lbl_gap * x_delta],
                    [pnt_y - pnt_gap * y_delta,
                     pos[0][1] + lbl_gap * y_delta
                     + 0.008 + 0.004 * np.sign(y_delta)],
                    c=use_clr, linewidth=2.3, alpha=0.27)

    ax.plot([0.65, 1.0005], [1, 1], color='black', linewidth=1.7, alpha=0.89)
    ax.plot([1, 1], [0.59, 1.0005], color='black', linewidth=1.7, alpha=0.89)

    ax.set_xlabel('AUC of Best Found Subgrouping',
                  size=23, weight='semibold')
    ax.set_ylabel('Down-Sampled AUC\nSuperiority Confidence',
                  size=23, weight='semibold')

    ax.tick_params(pad=5.3)
    ax.set_xlim([0.68, 1.002])
    ax.set_ylim([0.77, 1.005])

    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__gene-results.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_tuning_profile(acc_df, out_clf, auc_vals, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(out_clf.tune_priors)),
                              nrows=len(out_clf.tune_priors), ncols=1,
                              squeeze=False)

    use_aucs = auc_vals.round(4)
    auc_bins = pd.qcut(
        use_aucs.values.flatten(), q=[0., 0.5, 0.75, 0.8, 0.85, 0.9,
                                      0.92, 0.94, 0.96, 0.98, 0.99, 1.],
        precision=5
        ).categories

    use_cohs = sorted(set(acc_df.index.get_level_values('Cohort')))
    coh_clrs = dict(zip(use_cohs,
                        sns.color_palette("muted", n_colors=len(use_cohs))))

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          out_clf.tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        plot_df = pd.DataFrame([])
        for (coh, mtype), acc_vals in acc_df.iterrows():
            par_df = pd.DataFrame.from_records([
                pd.Series({par_fnc(pars[par_name]): avg_val
                           for pars, avg_val in zip(par_ols, avg_ols)})
                for par_ols, avg_ols in zip(
                    acc_vals['par'], acc_vals['avg'])
                ]).quantile(q=0.25).reset_index()

            # note that we take the maximum of AUCs for finding the bin due
            # to (rare) cases where RandomTypes get duplicated
            par_df.columns = ['par', 'auc']
            par_df['auc_bin'] = auc_bins.get_loc(use_aucs[coh, mtype].max())
            plot_df = pd.concat([plot_df, par_df])

        for auc_bin, bin_vals in plot_df.groupby('auc_bin'):
            plot_vals = bin_vals.groupby('par').mean()
            ax.plot(plot_vals.index, plot_vals.auc)

        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0.45, 1.01)
        ax.tick_params(labelsize=19)
        ax.set_xlabel("Tested {} Value".format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, "Aggregate AUC", ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(os.path.join(plot_dir, args.expr_source,
                             "{}__tuning-profile.svg".format(args.classif)),
        bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Summarizes the enumeration and prediction results for a given "
        "classifier over a particular source of expression data."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__*__samps-*".format(args.expr_source),
            "out-conf__*__{}.p.gz".format(args.classif)
            ))
        ]

    if args.expr_source == 'Firehose':
        out_datas += [
            out_file.parts[-2:] for out_file in Path(base_dir).glob(
                os.path.join("toil__gns__beatAML__samps-*",
                             "out-conf__*__{}.p.gz".format(args.classif))
                )
            ]

    out_list = pd.DataFrame([
        {'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split('__')[1:-1])}
        for out_data in out_datas
        ]).groupby('Cohort').filter(
            lambda outs: ('Exon__Location__Protein' in set(outs.Levels)
                          and outs.Levels.str.match('Domain_').any())
            )

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby(['Cohort', 'Levels'])['Samps'].min()
    out_use = out_use[out_use.index.get_level_values(
        'Cohort').isin(train_cohorts)]
    os.makedirs(os.path.join(plot_dir, args.expr_source), exist_ok=True)

    phn_dict = {coh: dict()
                for coh in set(out_use.index.get_level_values('Cohort'))}
    auc_dict = dict()
    conf_dict = dict()
    acc_dict = dict()
    out_clf = dict()

    for (coh, lvls), ctf in out_use.iteritems():
        if coh == 'beatAML':
            out_tag = "toil__gns__beatAML__samps-{}".format(ctf)
        else:
            out_tag = "{}__{}__samps-{}".format(args.expr_source, coh, ctf)

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-pheno__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            phn_dict[coh].update(pickle.load(fl))

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-aucs__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            auc_vals = pickle.load(fl)['mean']

        auc_vals.index = pd.MultiIndex.from_product([[coh], auc_vals.index],
                                                    names=('Cohort', 'Mtype'))
        auc_dict[coh, lvls] = auc_vals

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-conf__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            conf_vals = pickle.load(fl)['mean']

        conf_vals = conf_vals[[not isinstance(mtype, RandomType)
                               for mtype in conf_vals.index]]
        conf_vals.index = pd.MultiIndex.from_product(
            [[coh], conf_vals.index], names=('Cohort', 'Mtype'))
        conf_dict[coh, lvls] = conf_vals

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-tune__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as fl:
            (_, _, acc_vals, out_clf[coh, lvls]) = pickle.load(fl)

        acc_vals.index = pd.MultiIndex.from_product([[coh], acc_vals.index],
                                                    names=('Cohort', 'Mtype'))
        acc_dict[coh, lvls] = acc_vals

    auc_vals = pd.concat(auc_dict.values()).sort_index()
    conf_vals = pd.concat(conf_dict.values()).sort_index()
    acc_df = pd.concat(acc_dict.values()).sort_index()

    out_clf = set(out_clf.values())
    assert len(out_clf) == 1
    out_clf = tuple(out_clf)[0]

    plot_gene_results(auc_vals, conf_vals, phn_dict, args)
    plot_tuning_profile(acc_df, out_clf, auc_vals, args)


if __name__ == "__main__":
    main()

