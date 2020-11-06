
from ..utilities.mutations import (pnt_mtype, copy_mtype,
                                   dup_mtype, loss_mtype, RandomType)
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.colour_maps import variant_clrs
from ..utilities.labels import get_cohort_label, get_fancy_label

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'copy')


def plot_sub_comparison(auc_vals, pheno_dict, use_gene, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    mtype_dict = {mtype: (mtype | MuType({('Gene', use_gene): dup_mtype}),
                          mtype | MuType({('Gene', use_gene): loss_mtype}))
                  for mtype in auc_vals.index
                  if (tuple(mtype.subtype_iter())[0][1]
                      & copy_mtype).is_empty()}

    plt_min = 0.83
    for base_mtype, (gain_dyad, loss_dyad) in mtype_dict.items():
        plt_min = min(plt_min, auc_vals[base_mtype] - 0.02)

        if tuple(base_mtype.subtype_iter())[0][1] == pnt_mtype:
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

    ax.set_xlim([plt_min, 1 + (1 - plt_min) / 181])
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 181])

    ax.set_xlabel("AUC of subgrouping", size=27, weight='semibold')
    ax.set_ylabel("AUC of subgrouping when\nCNAs were added",
                  size=27, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "{}_sub-comparisons_{}.svg".format(
                         use_gene, args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_copy',
        description="Plots data on classification tasks involving CNAs."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)

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

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    out_use = out_list.groupby('Levels')['Samps'].min()
    cdata = None
    pred_dict = dict()
    phn_dict = dict()
    auc_vals = pd.Series()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "cohort-data__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            new_cdata = pickle.load(f)

        if cdata is None:
            cdata = new_cdata
        else:
            cdata.merge(new_cdata)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_vals = auc_vals.append(pickle.load(f)['mean'])

    assert auc_vals.index.isin(phn_dict).all()
    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    use_aucs = auc_vals[[not isinstance(mtype, RandomType)
                         for mtype in auc_vals.index]]

    for gene, auc_vec in use_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        copy_mtypes = {
            mtype for mtype in auc_vec.index
            if not (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()
            }

        if copy_mtypes:
            plot_sub_comparison(auc_vec, phn_dict, gene, args)


if __name__ == '__main__':
    main()

