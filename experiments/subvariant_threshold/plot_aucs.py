
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'], 'HetMan',
                        'subvariant_threshold')
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '..', '..')])
plot_dir = os.path.join(base_dir, 'plots', 'aucs')

from HetMan.experiments.subvariant_infer import variant_clrs
from HetMan.experiments.subvariant_tour.plot_aucs import choose_gene_colour

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_infer.utils import Mcomb, ExMcomb
from HetMan.experiments.subvariant_infer import copy_mtype
from HetMan.experiments.subvariant_tour import pnt_mtype
from dryadic.features.mutations import MuType

import argparse
from pathlib import Path
import bz2
import dill as pickle
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.edgecolor']='white'


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
            annt_clr = choose_gene_colour(annt_lbl, clr_seed=9930)
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
        "Plots classification performance using mutation subtypes of a gene "
        "in a cohort chosen by using thresholds versus subtypes chosen by "
        "other means."
        )

    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('cohort', type=str, help="which TCGA cohort was used")
    parser.add_argument('classif', help='a mutation classifier')

    args = parser.parse_args()
    os.makedirs(os.path.join(plot_dir, args.gene), exist_ok=True)
    orig_dir = os.path.join(Path(base_dir).parent,
                            'subvariant_infer', args.gene)

    with bz2.BZ2File(os.path.join(
            base_dir, "out-pheno__{}__{}.p.gz".format(
                args.cohort, args.classif)
            )) as fl:
        phn_dict = pickle.load(fl)

    with bz2.BZ2File(os.path.join(
            base_dir, "out-aucs__{}__{}.p.gz".format(
                args.cohort, args.classif)
            )) as fl:
        auc_dict = pickle.load(fl)

    with bz2.BZ2File(os.path.join(
            orig_dir, "out-simil__{}__{}.p.gz".format(
                args.cohort, args.classif)
            )) as fl:
        orig_phns, orig_aucs = pickle.load(fl)[:2]

    auc_vals = pd.Series({mtype: auc_val
                          for mtype, auc_val in auc_dict.items()
                          if mtype.base_mtype.get_labels()[0] == args.gene})

    orig_aucs = orig_aucs.loc[[
        ((isinstance(mtype, RandomType) and isinstance(mtype.size_dist, int))
         or (not isinstance(mtype, (RandomType, Mcomb, ExMcomb))
             and mtype.get_labels()[0] == args.gene
             and (mtype.subtype_list()[0][1] & copy_mtype).is_empty()))
        for mtype in orig_aucs.index
        ], 'All']

    plot_sub_comparison(orig_aucs, auc_vals, orig_phns, phn_dict, args)


if __name__ == '__main__':
    main()

