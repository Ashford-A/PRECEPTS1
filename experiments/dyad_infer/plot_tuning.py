
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'variant_mutex')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

from HetMan.experiments.variant_mutex import *
from HetMan.experiments.variant_baseline.plot_model import detect_log_distr
from HetMan.experiments.utilities.pcawg_colours import cohort_clrs

import argparse
from pathlib import Path
import dill as pickle

import numpy as np
import pandas as pd
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def choose_coh_colour(cohort):
    if cohort == 'beatAML':
        use_clr = cohort_clrs['LAML']
    elif cohort.split('_')[0] in cohort_clrs:
        use_clr = cohort_clrs[cohort.split('_')[0]]

    else:
        raise ValueError("No colour available for cohort {} !".format(cohort))

    return use_clr


def plot_tuning_auc(tune_dict, simil_dict, prior_vals, args):
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(prior_vals)),
                              nrows=len(prior_vals), ncols=1,
                              squeeze=False)

    tune_df = pd.concat(tune_dict.values(),
                        keys=tune_dict.keys(), names=['Cohort'])
    auc_df = pd.concat([pd.DataFrame.from_records(auc_dict).transpose()
                        for _, auc_dict, _, _ in simil_dict.values()],
                       keys=simil_dict.keys(), names=['Cohort'])

    size_df = pd.concat([
        pd.DataFrame({(mtype1, mtype2): [
            np.mean(pheno_dict[mtype1] & ~pheno_dict[mtype2]),
            np.mean(~pheno_dict[mtype1] & pheno_dict[mtype2])
            ] for mtype1, mtype2 in simil_df}).transpose()
        for pheno_dict, _, _, simil_df in simil_dict.values()
        ], keys=simil_dict.keys(), names=['Cohort'])

    for ax, (par_name, tune_distr) in zip(axarr.flatten(), prior_vals):
        par_vals = tune_df.applymap(itemgetter(par_name))

        if detect_log_distr(tune_distr):
            par_vals = np.log10(par_vals)
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        par_vals += np.random.normal(
            0, (plt_xmax - plt_xmin) / (len(tune_distr) * 7), par_vals.shape)

        for coh in tune_dict:
            ax.scatter(
                par_vals.loc[coh].sort_index().values.tolist(),
                auc_df.loc[coh].sort_index().values.tolist(),
                s=(size_df.loc[coh].sort_index().values * 81).tolist(),
                c=choose_coh_colour(coh), alpha=0.41, edgecolor='none'
                )

        ax.set_xlim(plt_xmin, plt_xmax)
        ax.set_ylim(0.48, 1.02)
        ax.tick_params(labelsize=19)
        ax.set_xlabel('Tuned {} Value'.format(par_name),
                      fontsize=27, weight='semibold')

        ax.axhline(y=1.0, color='black', linewidth=2.1, alpha=0.37)
        ax.axhline(y=0.5, color='#550000',
                   linewidth=2.7, linestyle='--', alpha=0.29)

    fig.text(-0.01, 0.5, 'Task AUC', ha='center', va='center',
             fontsize=27, weight='semibold', rotation='vertical')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(os.path.join(plot_dir,
                             "{}__tuning-auc.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the tuning characteristics of a model in classifying "
        "the mutation status of paired mutations across all tested cohorts."
        )

    # parse command line arguments, create directory where plots will be saved
    parser.add_argument('classif', help='a mutation classifier')
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir), exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "*__samps-*/out-data__{}.p".format(args.classif))
        ]

    out_use = pd.DataFrame(
        [{'Cohort': out_data[0].split('__')[0],
          'Samps': int(out_data[0].split('__')[1].split('-')[1])}
         for out_data in out_datas]
        ).groupby('Cohort').min()

    tune_dict = dict()
    simil_dict = dict()
    prior_dict = dict()

    for coh, ctf in out_use.iterrows():
        out_tag = "{}__samps-{}".format(coh, ctf.values[0])

        with open(os.path.join(base_dir, out_tag,
                               "out-data__{}.p".format(args.classif)),
                  'rb') as f:

            out_dict = pickle.load(f)
            tune_dict[coh] = out_dict['Tune']
            prior_dict[coh] = out_dict['TunePriors']

        with open(os.path.join(base_dir, out_tag,
                               "out-simil__{}.p".format(args.classif)),
                  'rb') as f:
            simil_dict[coh] = pickle.load(f)

    assert len(set(prior_dict.values())) == 1, (
        "Experiments have been run with different tuning priors!")
    prior_vals = tuple(prior_dict.values())[0]

    plot_tuning_auc(tune_dict, simil_dict, prior_vals, args)


if __name__ == "__main__":
    main()

