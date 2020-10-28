
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_threshold import base_dir
from ..utilities.data_dirs import choose_source
from ..utilities.misc import choose_label_colour
from ..utilities.labels import get_cohort_label
from ..utilities.colour_maps import variant_clrs

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'conf')


def plot_sub_comparison(orig_aucs, auc_vals, orig_conf, conf_vals,
                        orig_phenos, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    cur_gene = {tuple(mtype.label_iter())[0] for mtype in orig_aucs.index}
    assert len(cur_gene) == 1
    cur_gene = tuple(cur_gene)[0]

    base_mtype = MuType({('Gene', cur_gene): pnt_mtype})
    base_size = orig_phenos[base_mtype].sum()
    md_size = int((base_size - 20) * 0.53) + 21

    ax.scatter(base_size, orig_aucs[base_mtype], c='none',
               edgecolors=variant_clrs['Point'], s=401, alpha=0.53,
               linewidths=3.7)
    plt_min = orig_aucs[base_mtype] - 0.13

    orig_df = pd.DataFrame({mtype: {'Size': orig_phenos[mtype].sum()}
                            for mtype in orig_aucs.index}).transpose()
    orig_df['AUC'] = orig_aucs[orig_df.index]

    orig_df = orig_df.loc[orig_df.Size < base_size]
    orig_plt = orig_df.groupby('Size')['AUC'].max()
    best_orig = orig_df.sort_values('AUC').index[-1]

    ax.plot(orig_plt.index, orig_plt.values, marker='o', markersize=9,
            linewidth=0, c=variant_clrs['Point'], alpha=0.23)

    ax.scatter(orig_df.loc[best_orig, 'Size'], orig_plt.max(),
               marker='*', facecolor=variant_clrs['Point'], edgecolor='black',
               s=611, alpha=0.43)

    plt_min = min(
        plt_min, orig_plt.min() - 0.03,
        orig_plt.iloc[abs((orig_plt.index - md_size)).argmin():].min() - 0.13
        )

    lgnd_ptchs = [Patch(color=variant_clrs['Point'], alpha=0.43,
                        label="{} subgroupings".format(cur_gene))]

    for annt_lbl in ['PolyPhen', 'SIFT']:
        annt_mtypes = {mtype for mtype in auc_vals.index
                       if mtype.annot == annt_lbl}

        if annt_mtypes:
            annt_clr = choose_label_colour(annt_lbl, clr_seed=9930)

            annt_df = pd.DataFrame({mtype: {'Size': pheno_dict[mtype].sum()}
                                            for mtype in annt_mtypes}).T
            annt_df['AUC'] = auc_vals[annt_df.index]
            annt_df = annt_df.sort_values('Size')

            best_annt = annt_df.sort_values('AUC').index[-1]
            annt_sc = np.greater.outer(conf_vals[best_annt],
                                       orig_conf[best_orig]).mean()

            ax.plot(annt_df.Size, annt_df.AUC, marker='o', markersize=6,
                    linewidth=3.7, c=annt_clr, alpha=0.47)
            ax.scatter(annt_df.loc[best_annt, 'Size'], annt_df.AUC.max(),
                       marker='*', facecolor=annt_clr, edgecolor='black',
                       s=503, alpha=0.41)

            plt_min = min(
                plt_min, annt_df.AUC.min() - 0.03, annt_df.AUC.iloc[
                    abs(annt_df.Size - md_size).values.argmin():].min() - 0.13
                )

            lgnd_ptchs += [Patch(color=annt_clr, alpha=0.43,
                                 label="{} thresholds  ({:.2f})".format(
                                     annt_lbl, annt_sc))]

    if plt_min < 0.5:
        plt_lgnd = ax.legend(handles=lgnd_ptchs, frameon=False, fontsize=19,
                             ncol=1, loc=2, handletextpad=0.7,
                             bbox_to_anchor=(0.09, 0.97))

    else:
        plt_lgnd = ax.legend(handles=lgnd_ptchs, frameon=False, fontsize=19,
                             ncol=1, loc=4, handletextpad=0.7,
                             bbox_to_anchor=(0.94, 0.011))

    ax.add_artist(plt_lgnd)
    ax.text(base_size, orig_aucs[base_mtype] + (1 - plt_min) / 31,
            "any {}\npoint mutation".format(cur_gene),
            size=13, ha='center', va='bottom')

    ax.set_xlabel("# of Mutated Samples", size=26, weight='semibold')
    ax.set_ylabel("AUC in {}".format(get_cohort_label(args.cohort)),
                  size=26, weight='semibold')

    xlims = ax.get_xlim()
    ax.plot(xlims, [1, 1], color='black', linewidth=1.5, alpha=0.89)
    ax.plot(xlims, [0.5, 0.5],
            color='black', linewidth=1.1, linestyle=':', alpha=0.89)

    ax.set_xlim(xlims)
    ax.set_ylim([plt_min, 1 + (1 - plt_min) / 71])

    plt.savefig(os.path.join(plot_dir, args.cohort,
                             "{}__sub-comparison_{}.svg".format(
                                 cur_gene, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_conf',
        description="Compares threshold-based results to subgrouping testing."
        )

    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    orig_dir = os.path.join(Path(base_dir).parent, "subgrouping_test")
    use_src = choose_source(args.cohort)

    orig_datas = [
        out_file.parts[-2:] for out_file in Path(orig_dir).glob(os.path.join(
            "{}__{}__samps-*".format(use_src, args.cohort),
            "out-trnsf__*__{}.p.gz".format(args.classif)
            ))
        ]

    orig_list = pd.DataFrame([
        {'Samps': int(orig_data[0].split('__samps-')[1]),
         'Levels': '__'.join(orig_data[1].split(
             'out-trnsf__')[1].split('__')[:-1])}
        for orig_data in orig_datas
        ])

    if orig_list.shape[0] == 0:
        raise ValueError("No subgrouping testing experiment output found "
                         "for these parameters!")

    orig_use = orig_list.groupby('Levels')['Samps'].min()
    if 'Consequence__Exon' not in orig_use.index:
        raise ValueError("Cannot compare AUCs until the subgrouping testing "
                         "experiment is run with mutation levels "
                         "`Consequence__Exon` which tests genes' "
                         "base mutations!")


    with bz2.BZ2File(os.path.join(
            base_dir, "out-pheno__{}__{}.p.gz".format(
                args.cohort, args.classif)
            )) as fl:
        phn_dict = pickle.load(fl)

    with bz2.BZ2File(os.path.join(
            base_dir, "out-aucs__{}__{}.p.gz".format(
                args.cohort, args.classif)
            )) as fl:
        auc_vals = pickle.load(fl)['mean']

    with bz2.BZ2File(os.path.join(
            base_dir, "out-conf__{}__{}.p.gz".format(
                args.cohort, args.classif)
            )) as fl:
        conf_vals = pickle.load(fl)

    assert sorted(auc_vals.index) == sorted(conf_vals.index), (
        "Threshold mutation types for which AUCs were calculated do not "
        "match those for which bootstrapped AUCs were calculated!"
        )

    orig_phns = dict()
    auc_df = dict()
    conf_df = dict()

    for lvls, ctf in orig_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(use_src, args.cohort, ctf)

        with bz2.BZ2File(os.path.join(orig_dir, out_tag,
                                      "out-pheno__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            orig_phns.update(pickle.load(f))

        with bz2.BZ2File(os.path.join(orig_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_df[lvls] = pickle.load(f)['mean']

        with bz2.BZ2File(os.path.join(orig_dir, out_tag,
                                      "out-conf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            conf_df[lvls] = pickle.load(f)

    orig_aucs = pd.concat(auc_df.values())
    orig_conf = pd.concat(conf_df.values())

    assert sorted(orig_aucs.index) == sorted(orig_conf.index), (
        "Mutations for which subvariant testing AUCs were calculated do not "
        "match those for which bootstrapped AUCs were calculated!"
        )

    orig_mtypes = [mtype for mtype in orig_aucs.index
                   if ((isinstance(mtype, RandomType)
                        and isinstance(mtype.size_dist, int)
                        and mtype.base_mtype is not None)
                       or (not isinstance(mtype, RandomType)
                           and (tuple(mtype.subtype_iter())[0][1]
                                & copy_mtype).is_empty()))]

    orig_aucs = orig_aucs[orig_mtypes]
    orig_conf = orig_conf[orig_mtypes]
    os.makedirs(os.path.join(plot_dir, args.cohort), exist_ok=True)

    for gene, auc_vec in orig_aucs.groupby(
            lambda mtype: tuple(mtype.label_iter())[0]):
        use_aucs = auc_vec[[mtype for mtype in auc_vec.index
                            if not isinstance(mtype, RandomType)]]

        if len(use_aucs) > 1:
            thresh_mtypes = [
                mtype for mtype in auc_vals.index
                if (mtype.annot in {'PolyPhen', 'SIFT'}
                    and tuple(mtype.base_mtype.label_iter())[0] == gene
                    and (tuple(mtype.base_mtype.subtype_iter())[0][1]
                         & copy_mtype).is_empty())
                ]

            if thresh_mtypes:
                plot_sub_comparison(
                    use_aucs, auc_vals[thresh_mtypes],
                    orig_conf[use_aucs.index], conf_vals[thresh_mtypes],
                    orig_phns, phn_dict, args
                    )


if __name__ == '__main__':
    main()

