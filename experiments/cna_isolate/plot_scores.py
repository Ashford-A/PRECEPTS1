
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'scores')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.cna_isolate.fit_isolate import load_infer_output
from HetMan.experiments.cna_isolate.plot_aucs import get_aucs

import numpy as np
from sklearn.preprocessing import quantile_transform
import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"
copy_dir = '/home/users/grzadkow/compbio/input-data/firehose'

wt_cmap = sns.light_palette('0.09', as_cmap=True)
loss_cmap = sns.light_palette('#750027', as_cmap=True)
gap_cmap = sns.light_palette('#857800', as_cmap=True)
gain_cmap = sns.light_palette('#003455', as_cmap=True)


def plot_cna_scores(iso_vals, args, cdata):
    fig, ax = plt.subplots(figsize=(16, 11))

    low_ctf, high_ctf = iso_vals.name
    cna_vals = cdata.copy_data.loc[cdata.subset_samps(), args.gene]
    iso_means = iso_vals.apply(np.mean).values
    y_bound = np.max(np.absolute(np.percentile(iso_means, q=(0.1, 99.9))))

    use_mtype = MuType({('Gene', args.gene): None})
    mut_stat = np.array(cdata.train_pheno(use_mtype))
    gap_stat = cdata.train_pheno({'Gene': args.gene, 'CNA': 'Range',
                                  'Cutoff': (low_ctf, high_ctf)})

    zero_qnt = np.mean(cna_vals < 0)
    cna_qnts = quantile_transform(
        cna_vals.copy().values.reshape(-1, 1)).flatten()

    if low_ctf < 0:
        loss_ctf = low_ctf
        wt_ctf = high_ctf
        gain_ctf = -high_ctf

        wt_stat = cdata.train_pheno({'Gene': args.gene, 'CNA': 'Range',
                                     'Cutoff': (high_ctf, -high_ctf)})

    else:
        loss_ctf = -low_ctf
        wt_ctf = low_ctf
        gain_ctf = high_ctf

        wt_stat = cdata.train_pheno({'Gene': args.gene, 'CNA': 'Range',
                                     'Cutoff': (-low_ctf, low_ctf)})

    loss_stat = cdata.train_pheno({'Gene': args.gene, 'CNA': 'Loss',
                                   'Cutoff': loss_ctf})
    gain_stat = cdata.train_pheno({'Gene': args.gene, 'CNA': 'Gain',
                                   'Cutoff': gain_ctf})

    loss_qnt = np.mean(cna_vals < loss_ctf)
    wt_qnt = np.mean(cna_vals < wt_ctf)
    gain_qnt = np.mean(cna_vals < gain_ctf)

    if np.any(loss_stat):
        sns.kdeplot(
            cna_qnts[loss_stat & ~mut_stat], iso_means[loss_stat & ~mut_stat],
            cmap=loss_cmap, shade=True, shade_lowest=False,
            alpha=0.73, bw=loss_qnt / 7, gridsize=250, n_levels=11, cut=0
            )

    if np.any(gap_stat):
        sns.kdeplot(
            cna_qnts[gap_stat & ~mut_stat], iso_means[gap_stat & ~mut_stat],
            cmap=gap_cmap, shade=True, shade_lowest=False, alpha=0.73,
            bw=(wt_qnt - loss_qnt) / 7, gridsize=250, n_levels=11, cut=0
            )

    sns.kdeplot(cna_qnts[wt_stat & ~mut_stat],
                iso_means[wt_stat & ~mut_stat],
                cmap=wt_cmap, shade=True, shade_lowest=False, alpha=0.73,
                bw=(gain_qnt - wt_qnt) / 7, gridsize=250, n_levels=11, cut=0)

    if np.any(gain_stat):
        sns.kdeplot(
            cna_qnts[gain_stat & ~mut_stat], iso_means[gain_stat & ~mut_stat],
            cmap=gain_cmap, shade=True, shade_lowest=False, alpha=0.73,
            bw=(1 - gain_qnt) / 7, gridsize=250, n_levels=11, cut=0
            )

    ax.scatter(cna_qnts[mut_stat], iso_means[mut_stat],
               s=13, c='#550000', alpha=0.37)

    plt.xlabel("{} {} Gistic Score Rank".format(args.cohort, args.gene),
               fontsize=23, weight='semibold')
    plt.ylabel("Inferred {} CNA Score".format(args.gene),
               fontsize=23, weight='semibold')

    plt.xticks([loss_qnt, wt_qnt, zero_qnt, gain_qnt],
               ['Loss Cutoff ({:+.3f})'.format(low_ctf),
                'WT Cutoff ({:+.3f})'.format(high_ctf),
                'GISTIC = 0', 'Gain Cutoff ({:+.3f})'.format(-high_ctf)],
               fontsize=15, ha='right', rotation=38)

    ax.axvline(x=loss_qnt, ymin=-y_bound * 2, ymax=y_bound * 2,
               ls='--', lw=3.3, c=loss_cmap(50))
    ax.axvline(x=wt_qnt, ymin=-y_bound * 2, ymax=y_bound * 2,
               ls='--', lw=3.3, c=wt_cmap(50))
    ax.axvline(x=zero_qnt, ymin=-y_bound * 2, ymax=y_bound * 2,
               ls=':', lw=0.9, c='black')
    ax.axvline(x=gain_qnt, ymin=-y_bound * 2, ymax=y_bound * 2,
               ls='--', lw=3.3, c=gain_cmap(50))

    ax.grid(False, which='major', axis='x')
    plt.ylim(-y_bound * 1.29, y_bound * 1.29)

    if low_ctf < 0:
        lgnd_lbls = [
            'CNA Loss used in training', 'CNA WT used in training',
            'CNA Loss withheld in training', 'CNA Gain withheld in training'
            ]

    else:
        lgnd_lbls = [
            'CNA Loss withheld in training', 'CNA WT used in training',
            'CNA Gain withheld in training', 'CNA Gain used in training'
            ]

    lgnd_lbls += ['Mutants withheld in training']
    plt.legend([Patch(color=loss_cmap(100), alpha=0.73),
                Patch(color=wt_cmap(100), alpha=0.73),
                Patch(color=gap_cmap(100), alpha=0.73),
                Patch(color=gain_cmap(100), alpha=0.73),
                Line2D([0], [0], lw=0, marker='o', markersize=14, alpha=0.57,
                       markerfacecolor='#550000', markeredgecolor='#550000')],
               lgnd_lbls, fontsize=17, loc=8, ncol=2)

    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}__ctfs_{:.3f}_{:.3f}.png".format(
                         args.cohort, args.gene, args.classif,
                         low_ctf, high_ctf)),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the inferred CNA scores for a cohort's samples against their "
        "actual CNA scores for a given set of cutoffs."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', var_source='mc3', expr_dir=firehose_dir,
        copy_source='Firehose', copy_dir=copy_dir, copy_discrete=False,
        syn=syn, cv_prop=1.0
        )
    
    iso_df = load_infer_output(
        os.path.join(base_dir, 'output',
                     args.cohort, args.gene, args.classif)
        )

    loss_df, gain_df = get_aucs(iso_df, args, cdata)
    plot_cna_scores(iso_df.loc[loss_df['CNA'].idxmax(), :], args, cdata)
    plot_cna_scores(iso_df.loc[gain_df['CNA'].idxmax(), :], args, cdata)

    plot_cna_scores(iso_df.loc[(loss_df['CNA'] - loss_df['Mut']).idxmax(), :],
                    args, cdata)


if __name__ == '__main__':
    main()

