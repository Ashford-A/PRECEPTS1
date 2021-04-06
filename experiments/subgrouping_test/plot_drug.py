
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_test import base_dir
from .plot_ccle import load_response_data
from ..utilities.misc import get_label, get_subtype, choose_label_colour
from ..utilities.labels import get_cohort_label, get_fancy_label
from ..utilities.label_placement import place_scatter_labels

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'drug')


def plot_sub_comparisons(infr_vals, resp_df, auc_df, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(11, 11))

    use_samps = set(infr_vals.columns) & set(resp_df.index)
    use_infr = infr_vals[use_samps]
    use_resp = resp_df.loc[use_samps]

    line_dict = dict()
    plot_dict = dict()
    plt_min = -0.17
    plt_max = 0.17

    for drug, resp_vals in use_resp.iteritems():
        resp_samps = resp_vals.index[~resp_vals.isna()]

        if len(resp_samps) >= 100:
            plt_size = 97 * len(resp_samps) / len(use_samps)

            base_corr = spearmanr(use_infr.iloc[0][resp_samps],
                                   resp_vals[resp_samps])
            subt_corr = spearmanr(use_infr.iloc[1][resp_samps],
                                   resp_vals[resp_samps])

            corr_tupl = -base_corr.correlation, -subt_corr.correlation
            plt_clr = choose_label_colour(str(drug))
            line_dict[corr_tupl] = dict(c=plt_clr)
            plt_min = min(plt_min, min(corr_tupl) - 0.07)
            plt_max = max(plt_max, max(corr_tupl) + 0.07)

            if base_corr.pvalue > 0.001 and subt_corr.pvalue < 0.001:
                plot_dict[corr_tupl] = [plt_size, (str(drug), '')]
            else:
                plot_dict[corr_tupl] = [plt_size, ('', '')]

    for corr_tupl in tuple(plot_dict):
        ax.scatter(*corr_tupl,
                   s=plot_dict[corr_tupl][0], c=[line_dict[corr_tupl]['c']],
                   alpha=0.31, edgecolor='none')
        plot_dict[corr_tupl][0] *= (plt_max - plt_min) / 13703

    ax.grid(linewidth=0.83, alpha=0.41)
    ax.plot([plt_min, plt_max], [0, 0],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)
    ax.plot([0, 0], [plt_min, plt_max],
            color='black', linewidth=1.3, linestyle=':', alpha=0.71)

    ax.plot([plt_min + 0.01, plt_max - 0.01],
            [plt_min + 0.01, plt_max - 0.01],
            color='#550000', linewidth=2.1, linestyle='--', alpha=0.41)

    mtype_lbl = get_fancy_label(get_subtype(infr_vals.index[1]))
    lbl_len = sum(len(lbl) for lbl in mtype_lbl)

    if lbl_len < 50:
        mtype_lbl = mtype_lbl.replace('\n', ' ')
    else:
        mtype_lbl = mtype_lbl.replace('\n', ' ').replace(
            ' or ', '\nor ').replace(' are ', ' are\n')

    ax.text(0.99, 0.03, get_cohort_label(args.cohort), size=23,
            style='italic', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel("correlation with drug response\nusing all point mutations",
                  size=21, weight='semibold')
    ax.set_ylabel("correlation with response using\n{}".format(mtype_lbl),
                  size=21, weight='semibold')

    if plot_dict:
        lbl_pos = place_scatter_labels(plot_dict, ax,
                                       plt_lims=[(plt_min, plt_max),
                                                 (plt_min, plt_max)],
                                       font_size=12, line_dict=line_dict)

    ax.set_xlim([plt_min, plt_max])
    ax.set_ylim([plt_min, plt_max])

    plt.savefig(os.path.join(
        plot_dir, '__'.join([args.expr_source, args.cohort]),
        "{}__sub-comparisons__{}__{}.svg".format(
            args.gene, infr_vals.index[1].get_filelabel(), args.classif)
        ), bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_drug'
        "Plots comparisons of classifier predictions and drug response."
        )

    parser.add_argument('expr_source',
                        help="a source of expression data", type=str)
    parser.add_argument('cohort', help='a training cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    resp_data = load_response_data()

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
    if 'Consequence__Exon' not in out_use.index:
        raise ValueError("Cannot compare AUCs until this experiment is run "
                         "with mutation levels `Consequence__Exon` "
                         "which tests genes' base mutations!")

    phn_dict = dict()
    auc_dict = dict()
    trnsf_dict = {lvls: dict() for lvls in out_use.index}
    trnsf_vals = dict()

    for lvls, ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(
            args.expr_source, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            phn_data = pickle.load(f)

            phn_dict.update({mtype: phn for mtype, phn in phn_data.items()
                             if not isinstance(mtype, RandomType)})

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_data = pickle.load(f)

            auc_dict[lvls] = auc_data.loc[[
                mtype for mtype in auc_data.index
                if (not isinstance(mtype, RandomType)
                    and get_label(mtype) == args.gene)
                ]]

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-trnsf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            trnsf_out = pickle.load(f)['CCLE']
            trnsf_dict[lvls]['Samps'] = trnsf_out['Samps']

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "trnsf-preds__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            trnsf_mat = pickle.load(f)['CCLE']

            trnsf_vals[lvls] = pd.DataFrame(np.vstack(trnsf_mat.values),
                                            index=trnsf_mat.index,
                                            columns=trnsf_dict[lvls]['Samps'])

    if not any(auc_data.shape[0] > 0 for auc_data in auc_dict.values()):
        raise ValueError("No experiment output found for "
                         "gene `{}` !".format(args.gene))

    os.makedirs(os.path.join(plot_dir,
                             '__'.join([args.expr_source, args.cohort])),
                exist_ok=True)

    auc_df = pd.concat(auc_dict.values())
    trnsf_vals = pd.concat(trnsf_vals.values())

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    base_aucs = np.array(auc_df.CV[base_mtype])

    for mtype, aucs in auc_df.CV.iteritems():
        if (np.array(aucs) > base_aucs).all():
            plot_sub_comparisons(trnsf_vals.loc[[base_mtype, mtype]],
                                 resp_data, auc_df, phn_dict, args)


if __name__ == '__main__':
    main()

