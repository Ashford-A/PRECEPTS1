
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.dyad_isolate import firehose_dir, syn_root

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities.process_output import (
    load_infer_tuning, load_infer_output)
from HetMan.experiments.utilities.scatter_plotting import place_annot
from HetMan.experiments.gene_baseline.plot_model import detect_log_distr

import argparse
from pathlib import Path
import synapseclient

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def get_aucs(iso_df, cdata):
    auc_vals = pd.Series(index=iso_df.index, dtype=np.float)

    for (cur_mtype, other_mtype), iso_vals in iso_df.iterrows():
        cur_pheno = np.array(cdata.train_pheno(cur_mtype))
        other_pheno = np.array(cdata.train_pheno(other_mtype))

        none_vals = np.concatenate(iso_vals[~cur_pheno & ~other_pheno].values)
        cur_vals = np.concatenate(iso_vals[cur_pheno & ~other_pheno].values)

        auc_vals[(cur_mtype, other_mtype)] = (
            np.less.outer(none_vals, cur_vals).mean()
            + np.equal.outer(none_vals, cur_vals).mean() / 2
            )

    return auc_vals


def plot_tuning_auc(tune_df, auc_vals, use_clf, args, cdata):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(use_clf.tune_priors)),
                              nrows=len(use_clf.tune_priors), ncols=1,
                              squeeze=False)

    size_vals = pd.Series(index=auc_vals.index, dtype=np.float)
    for cur_mtype, other_mtype in auc_vals.index:
        cur_pheno = np.array(cdata.train_pheno(cur_mtype))
        other_pheno = np.array(cdata.train_pheno(other_mtype))

        size_vals[(cur_mtype, other_mtype)] = (
            np.sum(cur_pheno & ~other_pheno)
            / np.sum(~cur_pheno & ~other_pheno)
            )

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
        ax.scatter(par_vals, auc_vals,
                   s=(size_vals * 103).tolist(), alpha=0.21)

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
        "classifying the mutation status of paired genes in a given cohort."
        )

    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir), exist_ok=True)
    out_path = Path(os.path.join(base_dir, 'output',
                                 args.cohort, args.classif))

    out_dirs = [out_dir.parent for out_dir in out_path.glob("*/out__task-0.p")
                if len(tuple(out_dir.parent.glob("out__*.p"))) > 0]
    out_samps = [int(out_dir.name.split('samps_')[1]) for out_dir in out_dirs
                 if (len(tuple(out_dir.parent.glob("out__*.p")))
                     == len(tuple(out_dir.parent.glob("slurm/fit-*.txt"))))]

    use_dir = str(out_dirs[np.argmin(out_samps)])
    tune_df, mut_clf = load_infer_tuning(use_dir)
    iso_df = load_infer_output(use_dir)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=None,
                           mut_levels=['Gene'], samp_cutoff=min(out_samps),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)
    auc_vals = get_aucs(iso_df, cdata)

    plot_tuning_auc(tune_df, auc_vals, mut_clf, args, cdata)


if __name__ == "__main__":
    main()

