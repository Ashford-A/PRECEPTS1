
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.module_isolate import *
from HetMan.experiments.subvariant_isolate.utils import compare_scores
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType

from HetMan.experiments.utilities.process_output import (
    load_infer_tuning, load_infer_output)
from HetMan.experiments.utilities.scatter_plotting import place_annot
from HetMan.experiments.mut_baseline.plot_model import detect_log_distr

import argparse
from pathlib import Path
import synapseclient

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from functools import reduce
from operator import or_

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_tuning_auc(tune_df, auc_vals, size_vals, use_clf, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

    size_vec = (417 * size_vals.values) / np.max(size_vals)
    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        par_vals = tune_df.loc[:, par_name].values

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
        ax.scatter(par_vals, auc_vals, s=size_vec, alpha=0.21)

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

    for genes, mut_levels in set((out_path[0], out_path[3])
                                 for out_path in out_paths):
        use_data = [(i, out_path) for i, out_path in enumerate(out_paths)
                    if out_path[0] == genes and out_path[3] == mut_levels]

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
    out_modules = [
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

 
    out_genes = reduce(or_, [set(out_module.split('_'))
                             for out_module in out_modules])

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=list(set(out_genes)),
                           mut_levels=use_lvls, expr_source='Firehose',
                           expr_dir=firehose_dir, var_source='mc3',
                           copy_source='Firehose', annot_file=annot_file,
                           syn=syn, cv_prop=1.0)

    iso_list = [load_infer_output(str(out_dir)) for out_dir in out_dirs]
    info_lists = [
        compare_scores(
            iso_df, cdata, get_similarities=False,
            all_mtype=reduce(
                or_, [
                    MuType({('Gene', out_gene):
                            cdata.train_mut[out_gene].allkey(
                                ['Scale', 'Copy'] + list(out_lvl))})
                    for out_gene in out_modl.split('_')
                    ]
                )
            )
        for iso_df, out_modl, out_lvl in zip(iso_list, out_modules, mut_lvls)
        ]

    tune_list = [lists[0] for lists in tune_list]
    auc_list = [lists[1] for lists in info_lists]
    size_list = [lists[2] for lists in info_lists]

    out_lists = [tune_list, auc_list, size_list, mut_clf, args]
    for i in range(3):
        out_lists[i] = pd.concat(out_lists[i]).sort_index()

    plot_tuning_auc(*out_lists)


if __name__ == "__main__":
    main()

