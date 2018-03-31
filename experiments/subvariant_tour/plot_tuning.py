
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities import test_output, test_parameters
from HetMan.predict.basic.classifiers import *

import argparse
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_hyperparameters(acc_df, par_df, args):
    tune_priors = eval(args.classif).tune_priors
    fig = plt.figure(figsize=(6 * len(tune_priors), 7))
    plt_grid = plt.GridSpec(4, len(tune_priors), wspace=0.3, hspace=0.1)
    
    for i, (par_name, tune_distr) in enumerate(tune_priors):
        use_par = par_df.loc[:, (slice(None), par_name)]

        if isinstance(tune_distr, tuple):
            main_ax = fig.add_subplot(plt_grid[:, i])

            use_df = pd.DataFrame({'Acc': acc_df.values.flatten(),
                                   'Par': use_par.values.flatten()})
            sns.violinplot(data=use_df, x='Par', y='Acc', ax=main_ax)

        elif tune_distr.dist.name == 'lognorm':
            main_ax = fig.add_subplot(plt_grid[:-1, i])
            plt.ylabel('AUC', size=16)

            sns.kdeplot(
                np.log10(use_par.values.flatten()), acc_df.values.flatten(),
                linewidths=0.9, alpha=0.5, gridsize=250, shade_lowest=True,
                n_levels=25, ax=main_ax
                )

            dist_ax = fig.add_subplot(plt_grid[-1, i])
            dist_ax.hist(np.log10(tune_distr.rvs(100000)), bins=100,
                         normed=True, histtype='stepfilled', alpha=0.6)
            dist_ax.set_xlim(*np.log10(tune_distr.interval(0.9999)))
            plt.ylabel('Density', size=16)

            plt.xlabel('Hyper-Parameter Value', size=17)
            dist_ax.set_xticklabels(["$10^{" + str(int(x)) + "}$"
                                     for x in dist_ax.get_xticks()])

            main_ax.set_xlim(*np.log10(tune_distr.interval(0.9999)))
            main_ax.xaxis.set_ticklabels([])

        main_ax.set_title(par_name, size=23)

    fig.savefig(os.path.join(
        plot_dir, 'par_distr__{}_{}__freq_{}__{}.png'.format(
            args.cohort, args.classif, args.freq_cutoff, args.mut_levels)
        ),
        dpi=250, bbox_inches='tight')
    plt.close()


def main():

    # parses command line arguments
    parser = argparse.ArgumentParser(
        description='Plot experiment results for given mutation classifier.')

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('classif', help='a mutation classifier')

    parser.add_argument('--freq_cutoff', default=0.02,
                        help='a mutation classifier')
    parser.add_argument('--mut_levels', default='Gene',
                        help='a mutation classifier')

    # load ICGC expression and mutation data, create directory to save plots
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    acc_df = test_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif,
        'freq_{}'.format(args.freq_cutoff), args.mut_levels
        ))

    par_df = test_parameters(os.path.join(
        base_dir, 'output', args.cohort, args.classif,
        'freq_{}'.format(args.freq_cutoff), args.mut_levels
        ))

    plot_hyperparameters(acc_df, par_df, args)


if __name__ == '__main__':
    main()

