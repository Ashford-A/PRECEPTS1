
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.module_isolate import *

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities.process_output import (
    load_infer_tuning, load_infer_output)
from HetMan.experiments.utilities.scatter_plotting import place_annot
from HetMan.experiments.gene_baseline.plot_model import detect_log_distr

import argparse
from pathlib import Path
from functools import reduce
from operator import or_
import synapseclient

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def get_aucs(iso_df, cdata):
    auc_vals = pd.Series(index=iso_df.index)

    for (base_genes, mtype), iso_vals in iso_df.iterrows():
        base_mtype = MuType({('Gene', base_genes): None})

        none_vals = np.concatenate(iso_vals[
            ~np.array(cdata.train_pheno(base_mtype))].values)
        cur_vals = np.concatenate(iso_vals[
            np.array(cdata.train_pheno(mtype))].values)

        auc_vals[(base_genes, mtype)] = np.less.outer(
            none_vals, cur_vals).mean()

    return auc_vals


def plot_tuning_auc(tune_df, auc_vals, use_clf, args, cdata):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

    module_vec = [module for module, _ in auc_vals.index]
    size_vec = [(463 * len(mtype.get_samples(cdata.train_mut))
                 / len(cdata.samples))
                for _, mtype in auc_vals.index]

    use_modules = sorted(set(module_vec))
    module_clrs = sns.color_palette("muted", n_colors=len(use_modules))
    clr_vec = [module_clrs[use_modules.index(gns)] for gns in module_vec]

    for ax, (par_name, tune_distr) in zip(axarr.flatten(),
                                          use_clf.tune_priors):
        par_vals = tune_df.loc[auc_vals.index, par_name]

        if detect_log_distr(tune_distr):
            par_vals = np.log10(par_vals)
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

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


def main():
    parser = argparse.ArgumentParser(
        "Plots the tuning characteristics of a model in classifying the "
        "isolated mutation status of module subvariants in a given cohort."
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

    tune_dict = {out_dir: load_infer_tuning(str(out_dir))
                 for out_dir in out_dirs}
    mut_clf = set(clf for _, clf in tune_dict.values())

    if len(mut_clf) != 1:
        raise ValueError("Each module subvariant isolation experiment must "
                         "be run with exactly one classifier!")

    mut_clf = tuple(mut_clf)[0]
    out_modules = [
        str(out_dir).split("/output/{}/".format(args.cohort))[1].split('/')[0]
        for out_dir in out_dirs
        ]

    tune_dict = {out_dir: tune_df
                 for out_dir, (tune_df, _) in tune_dict.items()}
    iso_dict = {out_dir: load_infer_output(str(out_dir))
                for out_dir in out_dirs}

    for out_dir, out_module in zip(out_dirs, out_modules):
        tune_dict[out_dir].index = [(tuple(out_module.split('_')), mtype)
                                    for mtype in tune_dict[out_dir].index]
        iso_dict[out_dir].index = [(tuple(out_module.split('_')), mtype)
                                   for mtype in iso_dict[out_dir].index]

    tune_df = pd.concat(tune_dict.values())
    use_lvls = ['Gene']
    mut_lvls = list(set(
        tuple(str(out_dir).split(
            "/output/{}/".format(args.cohort))[1].split('/')[3].split('__'))
        for out_dir in out_dirs
        ))

    seq_match = SequenceMatcher(a=mut_lvls[0], b=mut_lvls[1])
    for (op, start1, end1, start2, end2) in seq_match.get_opcodes():

        if op == 'equal' or op=='delete':
            use_lvls += mut_lvls[0][start1:end1]

        elif op == 'insert':
            use_lvls += mut_lvls[1][start2:end2]

        elif op == 'replace':
            use_lvls += mut_lvls[0][start1:end1]
            use_lvls += mut_lvls[1][start2:end2]
    
    out_genes = reduce(or_, [set(out_module.split('_'))
                             for out_module in out_modules])
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=list(out_genes), mut_levels=use_lvls,
        expr_source='Firehose', expr_dir=firehose_dir, syn=syn, cv_prop=1.0
        )
    auc_vals = get_aucs(pd.concat(iso_dict.values()), cdata)

    plot_tuning_auc(tune_df, auc_vals, mut_clf, args, cdata)


if __name__ == "__main__":
    main()

