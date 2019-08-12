
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

from HetMan.experiments.subvariant_tour import *
from HetMan.experiments.subvariant_tour.utils import calculate_aucs
from HetMan.experiments.subvariant_infer.merge_infer import merge_cohort_data
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
plt.style.use('fivethirtyeight')

use_marks = [(0, 3, 0)]
use_marks += [(i, 0, k) for k in (0, 70, 140, 210) for i in range(3, 8)]


def plot_tuning_auc(out_dict, auc_dict, args):
    tune_priors = tuple(out_dict.values())[0]['Clf'].tune_priors
    fig, axarr = plt.subplots(figsize=(17, 1 + 7 * len(tune_priors)),
                              nrows=len(tune_priors), ncols=1, squeeze=False)

    use_srcs = sorted(set(src for src, _, _ in out_dict.keys()))
    src_mrks = dict(zip(use_srcs, use_marks[:len(use_srcs)]))

    use_cohs = sorted(set(coh for _, coh, _ in out_dict.keys()))
    coh_clrs = dict(zip(use_cohs,
                        sns.color_palette("muted", n_colors=len(use_cohs))))

    for ax, (par_name, tune_distr) in zip(axarr.flatten(), tune_priors):
        if detect_log_distr(tune_distr):
            par_fnc = np.log10
            plt_xmin = 2 * np.log10(tune_distr[0]) - np.log10(tune_distr[1])
            plt_xmax = 2 * np.log10(tune_distr[-1]) - np.log10(tune_distr[-2])

        else:
            par_fnc = lambda x: x
            plt_xmin = 2 * tune_distr[0] - tune_distr[1]
            plt_xmax = 2 * tune_distr[-1] - tune_distr[-2]

        for (src, coh, lvls), ols in out_dict.items():
            tune_df = ols['Tune'].loc[:, (slice(None), par_name)]
            auc_df, phn_dict = auc_dict[src, coh, lvls]

            for mtype, vals in tune_df.iterrows():
                for (cis_lbl, _), val in vals.iteritems():
                    ax.scatter(par_fnc(val), auc_df.loc[mtype, cis_lbl],
                               marker=src_mrks[src],
                               s=491 * np.mean(phn_dict[mtype]),
                               c=coh_clrs[coh], alpha=0.17, edgecolor='none')

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
        os.path.join(plot_dir, "{}__tuning-auc.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser("Plots the tuning characteristics of a "
                                     "classifier used to infer the presence "
                                     "of genes' subvariants across cohorts.")

    parser.add_argument('classif', help='a mutation classifier')
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir), exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "*__samps-*", "out-data__*__{}.p.gz".format(args.classif)))
        ]

    out_use = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split('__')[1:-1])}
        for out_data in out_datas
        ]).groupby(['Source', 'Cohort', 'Levels'])['Samps'].min()

    out_dict = dict()
    cdata_dict = dict()
    auc_dict = dict()

    for (src, coh, lvls), ctf in tuple(out_use.iteritems())[2:]:
        with bz2.BZ2File(os.path.join(base_dir,
                                      "{}__{}__samps-{}".format(
                                          src, coh, ctf),
                                      "out-data__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as fl:
            out_dict[src, coh, lvls] = pickle.load(fl)

        cdata_dict[src, coh, lvls] = merge_cohort_data(
            os.path.join(base_dir, "{}__{}__samps-{}".format(src, coh, ctf)),
            lvls, use_seed=8713
            )

        auc_dict[src, coh, lvls] = calculate_aucs(
            out_dict[src, coh, lvls]['Infer'], cdata_dict[src, coh, lvls])

    plot_tuning_auc(out_dict, auc_dict, args)


if __name__ == "__main__":
    main()

