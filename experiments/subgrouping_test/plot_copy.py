
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan', 'subvariant_test')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'copy')

from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.utils import get_fancy_label
from HetMan.experiments.subvariant_infer import variant_clrs
from dryadic.features.mutations import MuType

import argparse
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


def select_mtype(mtype, gene):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is None:
            slct_stat = False
        else:
            slct_stat = mtype.base_mtype.get_labels()[0] == gene

    else:
        slct_stat = mtype.get_labels()[0] == gene

    return slct_stat


def plot_sub_comparison(auc_vals, pheno_dict, conf_vals, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    mtype_dict = {mtype: (mtype | MuType({('Gene', args.gene): gain_mtype}),
                          mtype | MuType({('Gene', args.gene): loss_mtype}))
                  for mtype in auc_vals.index
                  if (not isinstance(mtype, RandomType)
                      and (mtype.subtype_list()[0][1]
                           & copy_mtype).is_empty())}

    plt_min = 0.83
    for base_mtype, (gain_dyad, loss_dyad) in mtype_dict.items():
        plt_min = min(plt_min, auc_vals[base_mtype] - 0.02)

        if base_mtype.subtype_list()[0][1] == pnt_mtype:
            use_mrk = 'X'
            use_sz = 2917 * np.mean(pheno_dict[base_mtype])
            use_alpha = 0.47

        else:
            use_mrk = 'o'
            use_sz = 1751 * np.mean(pheno_dict[base_mtype])
            use_alpha = 0.29

        if gain_dyad in auc_vals.index:
            plt_min = min(plt_min, auc_vals[gain_dyad] - 0.02)
            ax.scatter(auc_vals[base_mtype], auc_vals[gain_dyad],
                       marker=use_mrk, s=use_sz, alpha=use_alpha,
                       facecolor=variant_clrs['Gain'], edgecolor='none')

        if loss_dyad in auc_vals.index:
            plt_min = min(plt_min, auc_vals[loss_dyad] - 0.02)
            ax.scatter(auc_vals[base_mtype], auc_vals[loss_dyad],
                       marker=use_mrk, s=use_sz, alpha=use_alpha,
                       facecolor=variant_clrs['Gain'], edgecolor='none')

    ax.plot([plt_min, 1], [0.5, 0.5],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0.5, 0.5], [plt_min, 1],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min, 1.0005], [1, 1],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([1, 1], [plt_min, 1.0005],
            color='black', linewidth=1.9, alpha=0.89)
    ax.plot([plt_min + 0.005, 0.997], [plt_min + 0.005, 0.997],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    ax.set_xlim([plt_min, 1 + (1 - plt_min) / 71])
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 71])

    ax.set_xlabel("AUC of subgrouping", size=27, weight='semibold')
    ax.set_ylabel("AUC of subgrouping when\nCNAs were added",
                  size=27, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_sub-comparisons_{}.svg".format(
                         args.gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plots the AUCs for a particular classifier on the mutations "
        "enumerated for a given mutated gene in a cohort relative to "
        "mutations in which copy number alterations are included."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help="a TCGA cohort", type=str)
    parser.add_argument('gene', help="a mutation classifier", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "{}__{}__samps-*".format(args.expr_source, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    out_list = pd.DataFrame([{'Samps': int(out_data[0].split('__samps-')[1]),
                              'Levels': '__'.join(out_data[1].split(
                                  'out-trnsf__')[1].split('__')[:-1])}
                             for out_data in out_datas])

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    out_use = out_list.groupby('Levels')['Samps'].min()
    if 'Exon__Location__Protein' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Exon__Location__Protein` "
                         "which tests genes' base mutations!")

    pred_dict = dict()
    phn_dict = dict()
    auc_dict = dict()
    conf_dict = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_data = pickle.load(f)

            pred_dict[lvls] = pred_data.loc[[select_mtype(mtype, args.gene)
                                             for mtype in pred_data.index]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_data = pickle.load(f)

            phn_dict.update({mtype: phn for mtype, phn in phn_data.items()
                             if select_mtype(mtype, args.gene)})

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_data = pd.DataFrame.from_dict(pickle.load(f))

            auc_dict[lvls] = auc_data.loc[[select_mtype(mtype, args.gene)
                                           for mtype in auc_data.index]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_data = pd.DataFrame.from_dict(pickle.load(f))

            conf_dict[lvls] = conf_data.loc[[select_mtype(mtype, args.gene)
                                             for mtype in conf_data.index]]

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    pred_df = pd.concat(pred_dict.values())
    auc_df = pd.concat(auc_dict.values())
    conf_df = pd.concat(conf_dict.values())
    assert auc_df.index.isin(phn_dict).all()

    plot_sub_comparison(auc_df['mean'], phn_dict, conf_df['mean'], args)


if __name__ == '__main__':
    main()

