
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from ..utilities.colour_maps import form_clrs
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.label_placement import place_scatter_labels

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
plot_dir = os.path.join(base_dir, 'plots', 'phantm')


base_mtype = MuType({('Gene', 'TP53'): pnt_mtype})
ls_mtype = MuType({('Gene', 'TP53'): {('Scale', 'Copy'): {(
    'Copy', 'ShalDel'): None}}})
mtree_k = ('Gene', 'Scale', 'Copy', 'Consequence', 'HGVSp')


def get_phantm_scores(phantm_file):
    phantm_data = pd.read_excel(phantm_file, header=1, index_col=0)

    return {
        var.replace('B', '%3D').replace('Z', '*'): (dt[3] + dt[4] - dt[5]) / 3
        for var, dt in phantm_data.iterrows()
        }


def plot_phantm_scores(phantm_scrs, pred_df, cdata, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    plt_mtypes = {
        mtype
        for mtype in cdata.mtrees[mtree_k]['TP53'].branchtypes(
            mtype=MuType({('Scale', 'Point'): {
                ('Consequence',
                 ('missense_variant', 'stop_gained',
                  'synonymous_variant')): None
                }}),
            )
        if 'HGVSp' in mtype.get_levels()
        }

    pred_scrs = pred_df.loc[base_mtype].apply(np.mean)
    ls_phn = np.array(cdata.train_pheno(ls_mtype))
    plt_dict = dict()

    for mtype in plt_mtypes:
        mtype_lbl = get_fancy_label(mtype)

        if mtype_lbl in phantm_scrs:
            mtype_phn = np.array(cdata.train_pheno(mtype)) & ls_phn

            if mtype_phn.any():
                plt_dict[mtype] = mtype_lbl, mtype_phn

        else:
            print("Could not find `{}` in PHANTM table!".format(mtype_lbl))

    for mtype, (lbl, phn) in plt_dict.items():
        mtype_scrs = pred_scrs[phn].mean()
        plt_sz = 71003 * np.mean(phn)
        mtype_cnsq = tuple(tuple(mtype.subtype_iter())[0][1].label_iter())[0]

        if mtype_cnsq == 'missense_variant':
            plt_clr = form_clrs['Missense_Mutation']
        elif mtype_cnsq == 'stop_gained':
            plt_clr = form_clrs['Nonsense_Mutation']
        elif mtype_cnsq == 'synonymous_variant':
            plt_clr = form_clrs['Silent']

        else:
            raise ValueError(
                "Unknown mutation consequence `{}`!".format(mtype_cnsq))

        ax.scatter(pred_scrs[phn].mean(), phantm_scrs[lbl],
                   c=[plt_clr], s=plt_sz, alpha=0.31, edgecolor='none')

    ax.grid(linewidth=0.71, alpha=0.37)
    ax.tick_params(labelsize=15)
    coh_lbl = get_cohort_label(args.cohort)

    ax.set_xlabel("Predicted TP53 Scores\nin {}".format(coh_lbl),
                  fontsize=27, weight='semibold')
    ax.set_ylabel("PHANTM Combined\nPhenotype Score",
                  fontsize=27, weight='semibold')

    plt.tight_layout(h_pad=1.7)
    fig.savefig(
        os.path.join(plot_dir, '__'.join([args.expr_source, args.cohort]),
                     "pred-scores_{}.svg".format(args.classif)),
        bbox_inches='tight', format='svg'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_phantm',
        description="Plots predictions' concordance with TP53 PHANTM scores."
        )

    parser.add_argument('expr_source', help="a source of expression datasets")
    parser.add_argument('cohort', help="a tumour cohort", type=str)
    parser.add_argument('classif', help="a mutation classifier", type=str)
    parser.add_argument('phantm_file',
                        help="a file containing TP53 PHANTM scores", type=str)

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
    auc_dict = dict()

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
                                      "out-pred__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            pred_dict[lvls] = pickle.load(f)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_dict.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_dict[lvls] = pd.DataFrame.from_dict(pickle.load(f))

    cdata.add_mut_lvls(mtree_k)
    pred_df = pd.concat(pred_dict.values())
    auc_df = pd.concat(auc_dict.values())
    phantm_scrs = get_phantm_scores(args.phantm_file)

    plot_phantm_scores(phantm_scrs, pred_df, cdata, args)


if __name__ == "__main__":
    main()

