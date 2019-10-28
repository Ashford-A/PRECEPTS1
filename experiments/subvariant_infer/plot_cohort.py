
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_infer')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'cohort')

from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_infer import copy_mtype
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_infer.utils import (
    Mcomb, ExMcomb, get_mtype_gene)
from dryadic.features.mutations import MuType

from HetMan.experiments.subvariant_tour.plot_aucs import choose_gene_colour
from HetMan.experiments.subvariant_infer.setup_infer import choose_source
from HetMan.experiments.subvariant_infer.plot_aucs import get_retest_mtype

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_retest_comparison(orig_aucs, auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(8, 8))

    plt_aucs = auc_vals[[mtype for mtype in auc_vals.index
                         if not isinstance(mtype, (Mcomb, ExMcomb))]]
    mtype_dict = {mtype: get_retest_mtype(mtype) for mtype in orig_aucs.index}

    for old_mtype, new_mtype in mtype_dict.items():
        if new_mtype in plt_aucs.index:
            plt_clr = choose_gene_colour(old_mtype.get_labels()[0])

            if old_mtype.subtype_list()[0][1] == pnt_mtype:
                use_mrk = 'X'
                use_size = 1387 * np.mean(pheno_dict[new_mtype])
            else:
                use_mrk = 'o'
                use_size = 791 * np.mean(pheno_dict[new_mtype])

            ax.scatter(orig_aucs[old_mtype], plt_aucs[new_mtype],
                       marker=use_mrk, s=use_size, c=[plt_clr], alpha=0.23,
                       edgecolor='none')

    plt_lims = (min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot(plt_lims, plt_lims,
            color='#550000', linewidth=1.7, linestyle='--', alpha=0.41)

    ax.plot(plt_lims, [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.53)
    ax.plot([0.5, 0.5], plt_lims,
            color='black', linewidth=1.3, linestyle=':', alpha=0.53)
    ax.plot(plt_lims, [1, 1], color='black', linewidth=1.7, alpha=0.89)
    ax.plot([1, 1], plt_lims, color='black', linewidth=1.7, alpha=0.89)

    ax.set_xlim(plt_lims)
    ax.set_ylim(plt_lims)
    ax.set_xlabel("Original AUC", size=23, weight='semibold')
    ax.set_ylabel("Retested AUC", size=23, weight='semibold')

    plt.savefig(os.path.join(plot_dir, args.cohort,
                             "retest-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_random_comparison(auc_vals, pheno_dict, args):
    use_aucs = auc_vals[
        [mtype for mtype in auc_vals.index
         if (not isinstance(mtype, (Mcomb, ExMcomb))
             and (isinstance(mtype, RandomType)
                  or (mtype.subtype_list()[0][1] != pnt_mtype
                      and (mtype.subtype_list()[0][1]
                           & copy_mtype).is_empty())))]
        ]

    plt_grps = use_aucs.groupby(get_mtype_gene)
    fig, axarr = plt.subplots(figsize=(0.5 + 1.5 * len(plt_grps), 7),
                              nrows=1, ncols=len(plt_grps), sharey=True)
 
    for i, (gene, plt_aucs) in enumerate(sorted(plt_grps,
                                                key=lambda x: x[1].max(),
                                                reverse=True)):
        axarr[i].set_title(gene, size=21, weight='semibold')

        plt_df = pd.DataFrame({
            'AUC': plt_aucs, 'Type': ['Rand' if isinstance(mtype, RandomType)
                                      else 'Orig' for mtype in plt_aucs.index]
            })

        sns.violinplot(x=plt_df.Type, y=plt_df.AUC, ax=axarr[i],
                       order=['Orig', 'Rand'],
                       palette=[choose_gene_colour(gene), '0.61'],
                       cut=0, linewidth=0, width=0.93)

        axarr[i].plot([-1, 2], [0.5, 0.5],
                      color='black', linewidth=2.3, linestyle=':', alpha=0.83)

        axarr[i].get_children()[0].set_alpha(0.53)
        axarr[i].get_children()[2].set_alpha(0.53)
        axarr[i].set_xlabel('')
        axarr[i].set_xticklabels([])

        if i == 0:
            axarr[i].set_ylabel('AUC', size=21, weight='semibold')
        else:
            axarr[i].set_ylabel('')

    plt.savefig(os.path.join(plot_dir, args.cohort,
                             "random-comparison_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the results of retesting subvariants in a particular cohort.")

    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)
    phn_dict = dict()
    auc_dict = dict()

    new_datas = [
        orig_file.parts[-2:]
        for orig_file in Path(base_dir).glob(
            os.path.join("*", "out-trnsf__{}__{}.p.gz".format(
                args.cohort, args.classif))
            )
        ]

    for gene, out_file in new_datas:
        with bz2.BZ2File(os.path.join(
                base_dir, gene, out_file.replace('out-trnsf', 'out-simil')
                )) as fl:
            out_vals = pickle.load(fl)[:2]

            phn_dict.update(out_vals[0])
            auc_dict[gene] = out_vals[1]

    auc_df = pd.concat(auc_dict.values())
    orig_dir = os.path.join(Path(base_dir).parent, 'subvariant_tour')
    orig_src = choose_source(args.cohort)

    orig_datas = [
        orig_file.parts[-2:]
        for orig_file in Path(orig_dir).glob(os.path.join(
            "{}__{}__samps-*".format(orig_src, args.cohort),
            "out-conf__*__{}.p.gz".format(args.classif)
            ))
        ]

    orig_use = pd.DataFrame([
        {'Samps': int(orig_data[0].split('__samps-')[1]),
         'Levels': '__'.join(orig_data[1].split(
             'out-conf__')[1].split('__')[:-1])}
        for orig_data in orig_datas
        ]).groupby(['Levels'])['Samps'].min()

    if 'Exon__Location__Protein' not in orig_use.index:
        raise ValueError("Cannot evaluate retested AUC performance until the "
                         "original enumeration experiment is run with "
                         "mutation levels `Exon__Location__Protein` which "
                         "tests genes' base mutations!")

    old_aucs = dict()
    for lvls, ctf in orig_use.iteritems():
        with bz2.BZ2File(os.path.join(
                    orig_dir, "{}__{}__samps-{}".format(
                        orig_src, args.cohort, ctf),
                    "out-aucs__{}__{}.p.gz".format(lvls, args.classif)
                    ),
                'r') as f:
            auc_vals = pickle.load(f).Chrm

            old_aucs[lvls] = auc_vals[[not isinstance(mtype, RandomType)
                                       for mtype in auc_vals.index]]

    orig_aucs = pd.concat(old_aucs.values())
    plot_retest_comparison(orig_aucs.copy(), auc_df.All, phn_dict, args)
    plot_random_comparison(auc_df.All, phn_dict, args)


if __name__ == '__main__':
    main()

