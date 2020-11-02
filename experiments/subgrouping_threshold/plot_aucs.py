
from ..utilities.mutations import pnt_mtype, copy_mtype, RandomType
from dryadic.features.mutations import MuType

from ..subgrouping_threshold import base_dir
from ..utilities.data_dirs import choose_source
from ..utilities.misc import choose_label_colour
from ..utilities.colour_maps import variant_clrs

import os
import argparse
from pathlib import Path
import bz2
import dill as pickle
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.use('Agg')
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plot_dir = os.path.join(base_dir, 'plots', 'aucs')


def plot_sub_comparison(orig_aucs, auc_vals, orig_phenos, pheno_dict, args):
    fig, ax = plt.subplots(figsize=(13, 8))

    base_mtype = MuType({('Gene', args.gene): pnt_mtype})
    base_size = orig_phenos[base_mtype].sum()
    ax.scatter(base_size, orig_aucs[base_mtype], c='none',
               edgecolors=variant_clrs['Point'], s=139, alpha=0.57,
               linewidths=3.7)

    orig_df = pd.DataFrame({mtype: {'Size': orig_phenos[mtype].sum()}
                            for mtype in orig_aucs.index
                            if not isinstance(mtype, RandomType)}).T
    orig_df['AUC'] = orig_aucs[orig_df.index]

    rand_df = pd.DataFrame({mtype: {'Size': mtype.size_dist}
                            for mtype in orig_aucs.index
                            if isinstance(mtype, RandomType)}).T
    rand_df['AUC'] = orig_aucs[rand_df.index]

    orig_df = orig_df.loc[orig_df.Size < base_size]
    rand_df = rand_df.loc[rand_df.Size < base_size]
    orig_plt = orig_df.groupby('Size')['AUC'].max()
    rand_plt = rand_df.groupby('Size')['AUC'].max()

    lgnd_ptchs = [
        Patch(color=variant_clrs['Point'], alpha=0.43,
              label="{} subgroupings".format(args.gene)),
        Patch(color='0.41', alpha=0.43,
              label="size-matched random\nsubsets of {}".format(args.gene))
        ]

    ax.plot(orig_plt.index, orig_plt.values, marker='o', markersize=6,
            linewidth=3.7, c=variant_clrs['Point'], alpha=0.23)
    ax.plot(rand_plt.index, rand_plt.values, marker='o', markersize=6,
            linewidth=3.7, c='0.41', alpha=0.23)

    for annt_lbl in ['PolyPhen', 'SIFT', 'VAF', 'depth']:
        annt_mtypes = {mtype for mtype in auc_vals.index
                       if mtype.annot == annt_lbl}

        if annt_mtypes:
            annt_clr = choose_label_colour(annt_lbl, clr_seed=9930)
            lgnd_ptchs += [Patch(color=annt_clr, alpha=0.43,
                                 label="{} thresholds".format(annt_lbl))]

            annt_sizes = pd.Series({mtype: pheno_dict[mtype].sum()
                                    for mtype in annt_mtypes})
            annt_plt = sorted(annt_mtypes,
                              key=lambda mtype: annt_sizes[mtype])

            ax.plot(annt_sizes[annt_plt], auc_vals[annt_plt], marker='o',
                    markersize=6, linewidth=2.3, c=annt_clr, alpha=0.47)

    plt_lgnd = ax.legend(handles=lgnd_ptchs, frameon=False, fontsize=19,
                         ncol=2, loc=4, handletextpad=0.7,
                         bbox_to_anchor=(0.96, 0.015))
    ax.add_artist(plt_lgnd)

    ax.set_xlabel("# of Mutated Samples", size=26, weight='semibold')
    ax.set_ylabel("AUC", size=26, weight='semibold')

    xlims = ax.get_xlim()
    ax.plot(xlims, [1, 1], color='black', linewidth=1.5, alpha=0.89)
    ax.set_xlim(xlims)

    ymin = min(0.77, orig_plt.min() - 0.11,
               rand_plt.min() - 0.11, auc_vals.min() - 0.11)
    ax.set_ylim([ymin, 1.004])

    plt.savefig(os.path.join(plot_dir, args.gene,
                             "{}__sub-comparison_{}.svg".format(
                                 args.cohort, args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        'plot_aucs',
        description="Plots comparisons of performances of classifier tasks."
        )

    parser.add_argument('gene', help='a mutated gene')
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
        raise ValueError("No subvariant testing experiment output found "
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

    auc_vals = auc_vals[[
        mtype for mtype in auc_vals.index
        if (tuple(mtype.base_mtype.label_iter())[0] == args.gene
            and (tuple(mtype.base_mtype.subtype_iter())[0][1]
                 & copy_mtype).is_empty())
        ]]

    if len(auc_vals) == 0:
        raise ValueError("No subgrouping threshold experiment output found "
                         "for gene {} in cohort {} for "
                         "classifier {} !".format(args.gene, args.cohort,
                                                  args.classif))

    orig_phns = dict()
    auc_df = dict()

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

    orig_aucs = pd.concat(auc_df.values())
    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)

    orig_aucs = orig_aucs[[
        ((isinstance(mtype, RandomType) and isinstance(mtype.size_dist, int)
          and mtype.base_mtype is not None
          and tuple(mtype.base_mtype.label_iter())[0] == args.gene)
         or (not isinstance(mtype, RandomType)
             and tuple(mtype.label_iter())[0] == args.gene
             and (tuple(mtype.subtype_iter())[0][1] & copy_mtype).is_empty()))
        for mtype in orig_aucs.index
        ]]

    plot_sub_comparison(orig_aucs, auc_vals, orig_phns, phn_dict, args)


if __name__ == '__main__':
    main()

