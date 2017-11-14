
"""Plots the results of describing the classification signatures of sub-types.

Examples:
    plot_portray.py LUAD ElasticNet
    plot_portray.py BRCA Ridge
    plot_portray.py THCA GradBoost

"""

import os
import sys

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir, 'plots', 'cross')
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities import cross_output
from HetMan.features.variants import MuType

import numpy as np
import pandas as pd
import argparse

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from functools import reduce
from operator import add


def plot_cross_performance(out_list, args):
   
    fig, axarr = plt.subplots(nrows=2, ncols=3)
    plot_fl = 'mtype-performance_{}-{}-{}.png'.format(
        args.cohort, args.classif, args.gene)

    out_data = reduce(add, out_list) / len(out_list)
    plot_min = out_data.min().min() - 0.01

    axes_list = [[['Base'], ['Mut', 'Null']],
                 [['Base'], ['MutX', 'NullX']],
                 [['Base'], ['Mut']],
                 [['Mut', 'Null'], ['MutX', 'NullX']],
                 [['Mut'], ['Null']],
                 [['MutX'], ['NullX']]]
    titles_list = [['AUC using all samples', 'AUC within {} status'],
                   ['AUC using all samples', 'AUC across {} status'],
                   ['AUC using all samples', 'AUC within {} mutated'],
                   ['AUC within {} status', 'AUC across {} status'],
                   ['AUC within {} status', 'AUC within {} status'],
                   ['AUC from {} mutated to wild-type',
                    'AUC from {} wild-type to mutated']]

    for ax, (x_vals, y_vals), (x_title, y_title) in zip(
            axarr.reshape(-1), axes_list, titles_list):

        x_perf = out_data[x_vals].mean(axis=1).dropna()
        y_perf = out_data[y_vals].mean(axis=1).dropna()

        use_mtypes = x_perf.index & y_perf.index
        ax.scatter(x_perf[use_mtypes], y_perf[use_mtypes],
                   s=16, color='black', alpha=0.5)
        ax.plot([-1, 2], [-1, 2],
                color='r', linewidth=0.9, linestyle='--', alpha=0.8)

        ax.set_xlim(plot_min, 1.02)
        ax.set_ylim(plot_min, 1.02)

        ax.axhline(y=0.5, xmin=-1, xmax=2,
                   color='black', linewidth=0.8, linestyle='-', alpha=0.7)
        ax.axvline(x=0.5, ymin=-1, ymax=2,
                   color='black', linewidth=0.8, linestyle='-', alpha=0.7)

        ax.tick_params(labelsize=10)
        ax.set_xlabel(x_title.format(args.gene), fontsize=16)
        ax.set_ylabel(y_title.format(args.gene), fontsize=16)

    fig.set_size_inches(22, 14)
    plt.savefig(os.path.join(plot_dir, plot_fl),
                dpi=600, bbox_inches='tight')
    plt.close()


def main():
    """Creates plots for the given combination of cohort and classifier."""

    # parses command line arguments
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('cohort', help='a TCGA cohort.')
    parser.add_argument('classif',
                        help='a classifier in HetMan.predict.classifiers')
    parser.add_argument('gene', help='a gene to cross over')
    args = parser.parse_args()

    cross_list = cross_output(os.path.join(
        base_dir, 'output', args.cohort, args.classif, 'cross', args.gene))

    plot_cross_performance(cross_list, args)


if __name__ == '__main__':
    main()

