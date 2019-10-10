
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_tour')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'scores')

from HetMan.experiments.subvariant_tour import pnt_mtype
from HetMan.experiments.subvariant_tour.utils import (
    get_fancy_label, RandomType)
from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.mutations import MuType

import argparse
from glob import glob
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


def plot_score_comparison(plt_mtype, infer_df, auc_vals, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(11, 10))

    mtype_lbl = get_fancy_label(plt_mtype)
    plt_gene = plt_mtype.get_labels()[0]
    base_mtype = MuType({('Gene', plt_gene): pnt_mtype})

    plot_df = pd.DataFrame({'Base': infer_df.loc[base_mtype].apply(np.mean),
                            'Subg': infer_df.loc[plt_mtype].apply(np.mean)})

    ax.plot(plot_df.Base[~pheno_dict[base_mtype]],
            plot_df.Subg[~pheno_dict[base_mtype]],
            marker='o', markersize=6, linewidth=0, alpha=0.19,
            mfc=variant_clrs['WT'], mec='none')

    ax.plot(plot_df.Base[pheno_dict[base_mtype] & ~pheno_dict[plt_mtype]],
            plot_df.Subg[pheno_dict[base_mtype] & ~pheno_dict[plt_mtype]],
            marker='o', markersize=9, linewidth=0, alpha=0.23,
            mfc='#B200FF', mec='none')

    ax.plot(plot_df.Base[pheno_dict[plt_mtype]],
            plot_df.Subg[pheno_dict[plt_mtype]],
            marker='o', markersize=11, linewidth=0, alpha=0.23,
            mfc=variant_clrs['Point'], mec='none')

    ax.text(0.98, 0.03, "{} mutants w/o\n{}".format(plt_gene, mtype_lbl),
            size=14, c='#B200FF', ha='right', va='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.98, "{}\nmutants".format(mtype_lbl),
            size=14, c=variant_clrs['Point'], ha='left', va='top',
            transform=ax.transAxes)

    ax.set_xlabel("{} Point Mutation Inferred Score".format(plt_gene),
                  size=21, weight='semibold')
    ax.set_ylabel("Best Subgrouping Inferred Score".format(str(plt_mtype)),
                  size=21, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_score-comparison_{}.svg".format(
                         plt_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the scores inferred by a particular classifier for the "
        "mutations enumerated for a given cohort."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    parser.add_argument(
        '--seed', default=3401, type=int,
        help="the random seed to use for setting plotting colours"
        )

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(
            "{}__{}__samps-*/out-conf__*__{}.p.gz".format(
                args.expr_source, args.cohort, args.classif)
            )
        ]

    out_use = pd.DataFrame([
        {'Samps': int(out_data[0].split('__samps-')[1]),
         'Levels': '__'.join(out_data[1].split(
             'out-conf__')[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby(['Levels'])['Samps'].min()

    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError(
            "Cannot plot inferred scores until this experiment is run with "
            "mutation levels `Exon__Location__Protein` which tests genes' "
            "base mutations!"
            )

    infer_dict = dict()
    phn_dict = dict()
    auc_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-data__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            infer_df = pickle.load(f)['Infer']['Chrm']

            infer_dict[lvls] = infer_df.loc[[
                mtype for mtype in infer_df.index
                if not isinstance(mtype, RandomType)
                ]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_vals = pickle.load(f).Chrm

            auc_dict[lvls] = auc_vals[[mtype for mtype in auc_vals.index
                                       if not isinstance(mtype, RandomType)]]

    infer_df = pd.concat(infer_dict.values())
    auc_vals = pd.concat(auc_dict.values())

    auc_vals = auc_vals[[
        mtype for mtype in auc_vals.index
        if not (mtype.subtype_list()[0][1] != pnt_mtype
                and phn_dict[mtype].sum() == phn_dict[MuType(
                    {('Gene', mtype.get_labels()[0]): pnt_mtype})].sum())
        ]]

    mtype_list = []
    for gene, auc_vec in auc_vals.groupby(
            lambda mtype: mtype.get_labels()[0]):

        if len(auc_vec) > 1:
            base_mtype = MuType({('Gene', gene): pnt_mtype})
            base_indx = auc_vec.index.get_loc(base_mtype)

            best_subtype = auc_vec[:base_indx].append(
                auc_vec[(base_indx + 1):]).idxmax()
            best_indx = auc_vec.index.get_loc(best_subtype)

            if auc_vec[best_indx] > 0.7:
                plot_score_comparison(best_subtype, infer_df, auc_vals,
                                      phn_dict, args)


if __name__ == '__main__':
    main()

