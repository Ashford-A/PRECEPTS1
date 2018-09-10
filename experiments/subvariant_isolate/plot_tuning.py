
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_isolate import *
from HetMan.experiments.subvariant_isolate.utils import compare_scores
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType

from HetMan.experiments.utilities.process_output import (
    load_infer_tuning, load_infer_output)
from HetMan.experiments.utilities import auc_cmap
from HetMan.experiments.mut_baseline.plot_model import detect_log_distr

import argparse
from pathlib import Path
import synapseclient

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from itertools import combinations as combn

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_tuning_auc(tune_df, auc_vals, size_vals, use_clf, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

    # gets the gene and number of samples associated with each mutation tested
    gene_vec = [[gn for gn, _ in mtypes[0].subtype_list()][0]
                for mtypes in auc_vals.index]
    size_vec = (417 * size_vals.values) / np.max(size_vals)

    # assigns a plotting colour to each gene whose mutations were tested
    use_genes = sorted(set(gene_vec))
    gene_clrs = sns.color_palette("muted", n_colors=len(use_genes))
    clr_vec = [gene_clrs[use_genes.index(gn)] for gn in gene_vec]

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        par_vals = np.array(tune_df.loc[:, par_name].values, dtype=np.float)

        if detect_log_distr(tune_distr):
            par_vals = np.log10(par_vals)
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        # jitters the paramater values and plots them against mutation AUC
        par_vals += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tune_distr) * 7), len(auc_vals))
        ax.scatter(par_vals, auc_vals, s=size_vec, c=clr_vec, alpha=0.21)

        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0.48, 1.02)
        ax.tick_params(labelsize=19)
        ax.set_xlabel('Tuned {} Value'.format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, 'Aggregate AUC', ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, "{}_{}__tuning-auc.png".format(
            args.cohort, args.classif)),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_tuning_grid(tune_df, auc_vals, size_vals, use_clf, args):
    par_count = len(use_clf.tune_priors)
    fig, axarr = plt.subplots(figsize=(0.5 + 7 * par_count, 7 * par_count),
                              nrows=par_count, ncols=par_count)

    size_vec = (417 * size_vals.values) / np.max(size_vals)
    auc_clrs = auc_vals.apply(auc_cmap)

    for i, (par_name, tune_distr) in enumerate(use_clf.tune_priors):
        axarr[i, i].grid(False)

        if detect_log_distr(tune_distr):
            use_distr = [np.log10(par_val) for par_val in tune_distr]
            par_lbl = par_name + '\n(log-scale)'

        else:
            use_distr = tune_distr
            par_lbl = par_name

        distr_diff = np.array(use_distr[-1]) - np.array(use_distr[0])
        plt_min = use_distr[0] - distr_diff / 9
        plt_max = use_distr[-1] + distr_diff / 9

        axarr[i, i].set_xlim(plt_min, plt_max)
        axarr[i, i].set_ylim(plt_min, plt_max)
        axarr[i, i].text(
            (plt_min + plt_max) / 2, (plt_min + plt_max) / 2, par_lbl,
            ha='center', fontsize=31, weight='semibold'
            )

        for par_val in use_distr:
            axarr[i, i].axhline(y=par_val, color='#116611',
                                ls='--', linewidth=1.9, alpha=0.23)
            axarr[i, i].axvline(x=par_val, color='#116611',
                                ls='--', linewidth=1.9, alpha=0.23)

    for (i, (par_name1, tn_distr1)), (j, (par_name2, tn_distr2)) in combn(
            enumerate(use_clf.tune_priors), 2):
        par_vals1 = np.array(tune_df.loc[:, par_name1].values, dtype=np.float)
        par_vals2 = np.array(tune_df.loc[:, par_name2].values, dtype=np.float)

        if detect_log_distr(tn_distr1):
            use_vals1 = np.log10(par_vals1)
            distr_diff = np.log10(np.array(tn_distr1[-1]))
            distr_diff -= np.log10(np.array(tn_distr1[0]))

            plt_ymin = np.log10(tn_distr1[0]) - distr_diff / 9
            plt_ymax = np.log10(tn_distr1[-1]) + distr_diff / 9

        else:
            use_vals1 = par_vals1
            distr_diff = tn_distr1[-1] - tn_distr1[0]

            plt_ymin = tn_distr1[0] - distr_diff / 9
            plt_ymax = tn_distr1[-1] + distr_diff / 9

        if detect_log_distr(tn_distr2):
            use_vals2 = np.log10(par_vals2)
            distr_diff = np.log10(np.array(tn_distr2[-1]))
            distr_diff -= np.log10(np.array(tn_distr2[0]))

            plt_xmin = np.log10(tn_distr2[0]) - distr_diff / 9
            plt_xmax = np.log10(tn_distr2[-1]) + distr_diff / 9

        else:
            use_vals2 = par_vals2
            distr_diff = np.log10(np.array(tn_distr2[-1]))
            distr_diff -= np.log10(np.array(tn_distr2[0]))

            plt_xmin = np.log10(tn_distr2[0]) - distr_diff / 9
            plt_xmax = np.log10(tn_distr2[-1]) + distr_diff / 9

        use_vals1 += np.random.normal(
            0, (plt_ymax - plt_ymin) / (len(tn_distr1) * 11), auc_vals.shape[0])
        use_vals2 += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tn_distr2) * 11), auc_vals.shape[0])

        axarr[i, j].scatter(use_vals2, use_vals1, s=size_vec, c=auc_clrs,
                            alpha=0.35, edgecolor='black')
        axarr[j, i].scatter(use_vals1, use_vals2, s=size_vec, c=auc_clrs,
                            alpha=0.35, edgecolor='black')

        axarr[i, j].set_xlim(plt_xmin, plt_xmax)
        axarr[i, j].set_ylim(plt_ymin, plt_ymax)
        axarr[j, i].set_ylim(plt_xmin, plt_xmax)
        axarr[j, i].set_xlim(plt_ymin, plt_ymax)

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, "{}_{}__tuning-grid.png".format(
            args.cohort, args.classif)),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the tuning characteristics of a model in "
        "classifying the mutation status of the genes in a given cohort."
        )

    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir), exist_ok=True)
    out_path = Path(os.path.join(base_dir, 'output', args.cohort))

    out_dirs = [
        out_dir.parent for out_dir in out_path.glob(
            "*/{}/**/out__task-0.p".format(args.classif))
        if (len(tuple(out_dir.parent.glob("out__*.p"))) > 0
            and (len(tuple(out_dir.parent.glob("out__*.p")))
                 == len(tuple(out_dir.parent.glob("slurm/fit-*.txt")))))
        ]

    out_paths = [
        str(out_dir).split("/output/{}/".format(args.cohort))[1].split('/')
        for out_dir in out_dirs
        ]

    for gene, mut_levels in set((out_path[0], out_path[3])
                                for out_path in out_paths):
        use_data = [(i, out_path) for i, out_path in enumerate(out_paths)
                    if out_path[0] == gene and out_path[3] == mut_levels]

        if len(use_data) > 1:
            use_samps = np.argmin(int(x[1][2].split('samps_')[-1])
                                  for x in use_data)

            for x in use_data[:use_samps] + use_data[(use_samps + 1):]:
                del(out_dirs[x[0]])

    tune_list = [load_infer_tuning(str(out_dir)) for out_dir in out_dirs]
    mut_clf = set(clf for _, clf in tune_list)
    if len(mut_clf) != 1:
        raise ValueError("Each subvariant isolation experiment must be run "
                         "with exactly one classifier!")

    mut_clf = tuple(mut_clf)[0]
    out_genes = [
        str(out_dir).split("/output/{}/".format(args.cohort))[1].split('/')[0]
        for out_dir in out_dirs
        ]

    use_lvls = ['Gene']
    mut_lvls = [
        tuple(str(out_dir).split(
            "/output/{}/".format(args.cohort))[1].split('/')[3].split('__'))
        for out_dir in out_dirs
        ]
    lvl_set = list(set(mut_lvls))

    seq_match = SequenceMatcher(a=lvl_set[0], b=lvl_set[1])
    for (op, start1, end1, start2, end2) in seq_match.get_opcodes():

        if op == 'equal' or op=='delete':
            use_lvls += lvl_set[0][start1:end1]

        elif op == 'insert':
            use_lvls += lvl_set[1][start2:end2]

        elif op == 'replace':
            use_lvls += lvl_set[0][start1:end1]
            use_lvls += lvl_set[1][start2:end2]

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=list(set(out_genes)),
                           mut_levels=use_lvls, expr_source='Firehose',
                           expr_dir=expr_dir, var_source='mc3',
                           copy_source='Firehose', annot_file=annot_file,
                           syn=syn, cv_prop=1.0)

    iso_list = [load_infer_output(str(out_dir)) for out_dir in out_dirs]
    info_lists = [
        compare_scores(
            iso_df, cdata, get_similarities=False,
            muts=cdata.train_mut[out_gene],
            all_mtype=MuType(cdata.train_mut[out_gene].allkey(
                ['Scale', 'Copy'] + list(out_lvl)))
            )
        for iso_df, out_gene, out_lvl in zip(iso_list, out_genes, mut_lvls)
        ]

    tune_list = [lists[0] for lists in tune_list]
    auc_list = [lists[1] for lists in info_lists]
    size_list = [lists[2] for lists in info_lists]
    out_lists = [tune_list, auc_list, size_list, mut_clf, args]

    for i in range(3):
        for j, out_gene in enumerate(out_genes):
            out_lists[i][j].index = [tuple(MuType({('Gene', out_gene): mtype})
                                           for mtype in mtypes)
                                     for mtypes in out_lists[i][j].index]

        out_lists[i] = pd.concat(out_lists[i]).sort_index()

    plot_tuning_auc(*out_lists)
    if len(mut_clf.tune_priors) > 1:
        plot_tuning_grid(*out_lists)


if __name__ == "__main__":
    main()

