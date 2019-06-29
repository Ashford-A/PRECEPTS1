
import os
import sys

base_dir = os.path.join(os.environ['DATADIR'],
                        'HetMan', 'variant_mutex')
sys.path.extend([os.path.join(os.path.dirname(__file__), '../../..')])
plot_dir = os.path.join(base_dir, 'plots', 'interaction')

from HetMan.experiments.variant_mutex import *
from HetMan.experiments.subvariant_infer import variant_mtypes, variant_clrs
mtype_dict = dict(variant_mtypes)

import argparse
import dill as pickle
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def get_mtype_clr(mtype):
    sub_mtype = mtype.subtype_list()[0][1]

    if not (mtype_dict['Loss'] & sub_mtype).is_empty():
        mtype_clr = variant_clrs['Loss']
    elif not (mtype_dict['Gain'] & sub_mtype).is_empty():
        mtype_clr = variant_clrs['Gain']

    elif sub_mtype == mtype_dict['Point']:
        mtype_clr = variant_clrs['Point']
    elif sub_mtype.cur_level in {'Form', 'Exon', 'Protein'}:
        mtype_clr = variant_clrs['Point']

    else:
        raise ValueError("Unrecognized type of mutation {} !".format(mtype))

    return mtype_clr


def create_twotone_circle(xy, clrs, scale=1, **ptchargs):
    BEZR = 0.2652031
    SQRT2 = np.sqrt(0.5)
    BEZR45 = SQRT2 * BEZR
    GAPADJ = 1 / 79

    path_codes = [Path.MOVETO] + [Path.CURVE4] * 12 + [Path.CLOSEPOLY]
    path_verts = np.array([[0.0, -1.0], [BEZR, -1.0],
                           [SQRT2-BEZR45, -SQRT2-BEZR45], [SQRT2, -SQRT2],
                           [SQRT2+BEZR45, -SQRT2+BEZR45], [1.0, -BEZR],
                           [1.0, 0.0], [1.0, BEZR],
                           [SQRT2+BEZR45, SQRT2-BEZR45], [SQRT2, SQRT2],
                           [SQRT2-BEZR45, SQRT2+BEZR45], [BEZR, 1.0],
                           [0.0, 1.0], [0.0, -1.0]])

    circ_ptchs = [PathPatch(Path((path_verts - [GAPADJ, 0]) * [2 * i - 1, 1]
                                 * scale + np.array(xy),
                                 path_codes),
                            facecolor=clr, edgecolor='none', alpha=0.29,
                            **ptchargs)
                  for i, clr in enumerate(clrs)]

    return circ_ptchs


def plot_mutual_similarity(use_mtypes, stat_dict, mutex_dict, siml_dict,
                           args):
    fig_size = (11, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    plot_df = pd.DataFrame({
        'Occur': pd.Series(mutex_dict)[use_mtypes],
        'Simil': pd.Series({mtypes: siml_dict[mtypes].loc['Other'].mean()
                            for mtypes in use_mtypes})
        })

    plot_lims = plot_df.quantile(q=[0, 1])
    plot_rngs = plot_lims.diff().iloc[1]

    xy_scale = np.array([1, 2 ** np.log2(plot_rngs).diff()[-1]
                         * 2 ** -np.diff(np.log2(fig_size))])
    xy_scale /= (np.prod(plot_rngs) ** -0.76) * 19

    for (mtype1, mtype2), (occur_val, simil_val) in plot_df.iterrows():
        plt_size = (stat_dict[mtype1].mean() * stat_dict[mtype2].mean())
        plt_size = (plt_size ** 0.25) * (plot_df.shape[0] ** -0.19)
 
        for ptch in create_twotone_circle((occur_val, simil_val),
                                          [get_mtype_clr(mtype1),
                                           get_mtype_clr(mtype2)],
                                          scale=xy_scale * plt_size):
            ax.add_artist(ptch)

    ax.set_xlim(*(plot_lims.Occur
                  + [plot_rngs.Occur / -13, plot_rngs.Occur / 31]))
    ax.set_ylim(*(plot_lims.Simil
                  + [plot_rngs.Simil / -31, plot_rngs.Simil / 13]))

    ax.text(plot_rngs.Occur / -203, plot_lims.Simil[1] + plot_rngs.Simil / 15,
            "significant exclusivity \u2190", size=12, ha='right', va='top')
    ax.text(plot_rngs.Occur / 203, plot_lims.Simil[1] + plot_rngs.Simil / 15,
            "\u2192 significant overlap", size=12, ha='left', va='top')

    ax.text(plot_lims.Occur[0] - plot_rngs.Occur / 15, plot_rngs.Simil / -203,
            "             opposite \u2190\ndownstream effects",
            size=12, rotation=90, ha='left', va='top')
    ax.text(plot_lims.Occur[0] - plot_rngs.Occur / 15, plot_rngs.Simil / 203,
            "\u2192 similar\n   downstream effects",
            size=12, rotation=90, ha='left', va='bottom')

    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)
    ax.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)

    plt.xlabel("Genomic Co-occurence",
               size=17, weight='semibold')
    plt.ylabel("Transcriptomic Similarity", size=17, weight='semibold')

    plt.savefig(os.path.join(plot_dir,
                             "{}__samps-{}".format(args.cohort,
                                                   args.samp_cutoff),
                             "mutual-simil_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_mutual_synergy(use_mtypes, stat_dict, mutex_dict, siml_dict, args):
    fig_size = (11, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    plot_df = pd.DataFrame({
        'Occur': pd.Series(mutex_dict)[use_mtypes],
        'Syner': pd.Series({mtypes: siml_dict[mtypes].loc['Both'].mean()
                            for mtypes in use_mtypes})
        })

    plot_lims = plot_df.quantile(q=[0, 1])
    plot_rngs = plot_lims.diff().iloc[1]

    xy_scale = np.array([1, 2 ** np.log2(plot_rngs).diff()[-1]
                         * 2 ** -np.diff(np.log2(fig_size))])
    xy_scale /= (np.prod(plot_rngs) ** -0.76) * 19

    for (mtype1, mtype2), (occur_val, syner_val) in plot_df.iterrows():
        plt_size = (stat_dict[mtype1].mean() * stat_dict[mtype2].mean())
        plt_size = (plt_size ** 0.25) * (len(use_mtypes) ** -0.19)
 
        for ptch in create_twotone_circle((occur_val, syner_val),
                                          [get_mtype_clr(mtype1),
                                           get_mtype_clr(mtype2)],
                                          scale=xy_scale * plt_size):
            ax.add_artist(ptch)

    ax.set_xlim(*(plot_lims.Occur
                  + [plot_rngs.Occur / -13, plot_rngs.Occur / 31]))
    ax.set_ylim(*(plot_lims.Syner
                  + [plot_rngs.Syner / -31, plot_rngs.Syner / 13]))

    ax.text(plot_rngs.Occur / -203, plot_lims.Syner[1] + plot_rngs.Syner / 15,
            "significant exclusivity \u2190", size=12, ha='right', va='top')
    ax.text(plot_rngs.Occur / 203, plot_lims.Syner[1] + plot_rngs.Syner / 15,
            "\u2192 significant overlap", size=12, ha='left', va='top')

    ax.text(plot_lims.Occur[0] - plot_rngs.Occur / 15,
            1 - plot_rngs.Syner / 203,
            "          discordant \u2190\ndownstream effects",
            size=12, rotation=90, ha='left', va='top')
    ax.text(plot_lims.Occur[0] - plot_rngs.Occur / 15,
            1 + plot_rngs.Syner / 203,
            "\u2192 synergystic\n   downstream effects",
            size=12, rotation=90, ha='left', va='bottom')

    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)
    ax.axhline(1, color='black', linewidth=0.7, linestyle='--', alpha=0.35)
    ax.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)

    plt.xlabel("Genomic Co-occurence",
               size=17, weight='semibold')
    plt.ylabel("Transcriptomic Synergy", size=17, weight='semibold')

    plt.savefig(os.path.join(plot_dir,
                             "{}__samps-{}".format(args.cohort,
                                                   args.samp_cutoff),
                             "mutual-synergy_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_similarity_synergy(use_mtypes, stat_dict, siml_dict, args):
    fig_size = (11, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    plot_df = pd.DataFrame({
        'Simil': pd.Series({mtypes: siml_dict[mtypes].loc['Other'].mean()
                            for mtypes in use_mtypes}),
        'Syner': pd.Series({mtypes: siml_dict[mtypes].loc['Both'].mean()
                            for mtypes in use_mtypes})
        })

    plot_lims = plot_df.quantile(q=[0, 1])
    plot_rngs = plot_lims.diff().iloc[1]

    xy_scale = np.array([1, 2 ** np.log2(plot_rngs).diff()[-1]
                         * 2 ** -np.diff(np.log2(fig_size))])
    xy_scale /= (np.prod(plot_rngs) ** -0.76) * 19

    for (mtype1, mtype2), (simil_val, syner_val) in plot_df.iterrows():
        plt_size = (stat_dict[mtype1].mean() * stat_dict[mtype2].mean())
        plt_size = (plt_size ** 0.25) * (plot_df.shape[0] ** -0.19)
 
        for ptch in create_twotone_circle((simil_val, syner_val),
                                          [get_mtype_clr(mtype1),
                                           get_mtype_clr(mtype2)],
                                          scale=xy_scale * plt_size):
            ax.add_artist(ptch)

    ax.set_xlim(*(plot_lims.Simil
                  + [plot_rngs.Simil / -13, plot_rngs.Simil / 31]))
    ax.set_ylim(*(plot_lims.Syner
                  + [plot_rngs.Syner / -31, plot_rngs.Syner / 13]))

    ax.text(plot_lims.Simil[0] - plot_rngs.Simil / 15,
            1 - plot_rngs.Syner / 203,
            "          discordant \u2190\ndownstream effects",
            size=12, rotation=90, ha='left', va='top')
    ax.text(plot_lims.Simil[0] - plot_rngs.Simil / 15,
            1 + plot_rngs.Syner / 203,
            "\u2192 synergystic\n   downstream effects",
            size=12, rotation=90, ha='left', va='bottom')

    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.axhline(1, color='black', linewidth=0.7, linestyle='--', alpha=0.35)
    ax.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)

    plt.xlabel("Transcriptomic Similarity", size=17, weight='semibold')
    plt.ylabel("Transcriptomic Synergy", size=17, weight='semibold')

    plt.savefig(os.path.join(plot_dir,
                             "{}__samps-{}".format(args.cohort,
                                                   args.samp_cutoff),
                             "simil-syner_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def plot_similarity_symmetry(use_mtypes, stat_dict, siml_dict, args):
    fig, ax = plt.subplots(figsize=(9, 9))

    plot_df = pd.DataFrame({
        'Simil1': pd.Series({mtypes: siml_dict[mtypes].loc['Other', 'Mtype1']
                             for mtypes in use_mtypes}),
        'Simil2': pd.Series({mtypes: siml_dict[mtypes].loc['Other', 'Mtype2']
                             for mtypes in use_mtypes})
        })

    plot_lim = np.percentile(plot_df, q=[0, 100])
    plot_rng = plot_lim[1] - plot_lim[0]

    for (mtype1, mtype2), (simil_val1, simil_val2) in plot_df.iterrows():
        plt_size = (stat_dict[mtype1].mean() * stat_dict[mtype2].mean())
        plt_size = (plt_size ** 0.25) * (plot_df.shape[0] ** -0.19) / 1.43
        plt_size /= (plot_rng ** -1.52) * 19
 
        for ptch in create_twotone_circle((simil_val1, simil_val2),
                                          [get_mtype_clr(mtype1),
                                           get_mtype_clr(mtype2)],
                                          scale=plt_size):
            ax.add_artist(ptch)

    ax.set_xlim(*(plot_lim + [plot_rng / -13, plot_rng / 31]))
    ax.set_ylim(*(plot_lim + [plot_rng / -13, plot_rng / 31]))
    ax.plot(*([plot_lim[0] - 1, plot_lim[1] + 1] * 2),
            linewidth=1.1, linestyle=':', color='#550000', alpha=0.31)

    ax.text(plot_lim[0] - plot_rng / 15, plot_rng / -203,
            "            opposite \u2190\ndownstream effects",
            size=12, rotation=90, ha='left', va='top')
    ax.text(plot_lim[0] - plot_rng / 15, plot_rng / 203,
            "\u2192 similar\n   downstream effects",
            size=12, rotation=90, ha='left', va='bottom')

    plt.xticks(size=10)
    plt.yticks(size=10)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)
    ax.axvline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)

    plt.xlabel("Transcriptomic Similarity", size=17, weight='semibold')
    plt.ylabel("Transcriptomic Similarity", size=17, weight='semibold')

    plt.savefig(os.path.join(plot_dir,
                             "{}__samps-{}".format(args.cohort,
                                                   args.samp_cutoff),
                             "simil-symmetry_{}.svg".format(args.classif)),
                bbox_inches='tight', format='svg')

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('samp_cutoff')
    parser.add_argument('classif', help='a mutation classifier')

    # parse command line arguments, create directory where plots will be saved
    args = parser.parse_args()
    out_tag = "{}__samps-{}".format(args.cohort, args.samp_cutoff)
    os.makedirs(os.path.join(plot_dir, out_tag), exist_ok=True)

    with open(os.path.join(base_dir, out_tag,
                           "out-data__{}.p".format(args.classif)),
              'rb') as f:
        out_infer = pickle.load(f)['Infer']

    with open(os.path.join(base_dir, out_tag,
                           "out-simil__{}.p".format(args.classif)),
              'rb') as f:
        stat_dict, auc_dict, mutex_dict, siml_dict = pickle.load(f)

    auc_df = (pd.DataFrame(auc_dict) >= 0.8).all(axis=0)
    use_mtypes = auc_df.index[auc_df]

    plot_mutual_similarity(use_mtypes, stat_dict, mutex_dict, siml_dict, args)
    plot_mutual_synergy(use_mtypes, stat_dict, mutex_dict, siml_dict, args)
    plot_similarity_synergy(use_mtypes, stat_dict, siml_dict, args)
    plot_similarity_symmetry(use_mtypes, stat_dict, siml_dict, args)


if __name__ == '__main__':
    main()

