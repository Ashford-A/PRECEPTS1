
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'auc')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.cna_isolate.fit_isolate import load_infer_output

import numpy as np
import pandas as pd

import argparse
import synapseclient

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"
copy_dir = '/home/users/grzadkow/compbio/input-data/firehose'


def get_aucs(iso_df, args, cdata):
    base_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', args.gene): None})))

    loss_aucs = {ctf: {'CNA': None, 'Mut': None} for ctf in iso_df.index
                 if ctf[0] < 0}
    gain_aucs = {ctf: {'CNA': None, 'Mut': None} for ctf in iso_df.index
                 if ctf[0] > 0}

    for low_ctf, high_ctf in iso_df.index:
        use_vals = iso_df.loc[(low_ctf, high_ctf), :].values

        if low_ctf < 0:
            cna_pheno = np.array(cdata.train_pheno(
                {'Gene': args.gene, 'CNA': 'Loss', 'Cutoff': low_ctf}))

            wt_stat = ~base_pheno & np.array(cdata.train_pheno(
                {'Gene': args.gene, 'CNA': 'Range', 
                 'Cutoff': (high_ctf, -high_ctf)}
                ))

        else:
            cna_pheno = np.array(cdata.train_pheno(
                {'Gene': args.gene, 'CNA': 'Gain', 'Cutoff': high_ctf}))

            wt_stat = ~base_pheno & np.array(cdata.train_pheno(
                {'Gene': args.gene, 'CNA': 'Range', 
                 'Cutoff': (-low_ctf, low_ctf)}
                ))

        wt_vals = np.concatenate(use_vals[wt_stat])
        cna_vals = np.concatenate(use_vals[cna_pheno & ~base_pheno])
        mut_vals = np.concatenate(use_vals[~cna_pheno & base_pheno])

        cna_auc = np.greater.outer(cna_vals, wt_vals).mean()
        mut_auc = np.greater.outer(mut_vals, wt_vals).mean()

        if low_ctf < 0:
            loss_aucs[low_ctf, high_ctf]['CNA'] = cna_auc
            loss_aucs[low_ctf, high_ctf]['Mut'] = mut_auc

        else:
            gain_aucs[low_ctf, high_ctf]['CNA'] = cna_auc
            gain_aucs[low_ctf, high_ctf]['Mut'] = mut_auc

    loss_df = pd.DataFrame.from_dict(loss_aucs, orient='index')
    gain_df = pd.DataFrame.from_dict(gain_aucs, orient='index')

    return loss_df, gain_df


def plot_cutoff_aucs(loss_df, gain_df, args, cdata):
    fig, (loss_ax, gain_ax) = plt.subplots(
        figsize=(19, 9), ncols=2, gridspec_kw={'width_ratios': [1, 1.1]})

    loss_ctfs = loss_df.index.levels[0] | loss_df.index.levels[1]
    loss_lbls = ['{:+.3f}'.format(ctf) for ctf in sorted(loss_ctfs)]
    gain_ctfs = gain_df.index.levels[0] | gain_df.index.levels[1]
    gain_lbls = ['{:+.3f}'.format(ctf) for ctf in sorted(gain_ctfs)]

    use_cmap = sns.diverging_palette(11, 238, s=87, l=37, sep=57,
                                     as_cmap=True)

    plt_loss = pd.DataFrame(0.0, index=loss_ctfs, columns=loss_ctfs)
    plt_loss.iloc[:-1, 1:] += loss_df['CNA'].unstack().fillna(0.0)
    plt_loss.iloc[1:, :-1] += loss_df['Mut'].unstack().transpose().fillna(0.0)

    plt_gain = pd.DataFrame(0.0, index=gain_ctfs, columns=gain_ctfs)
    plt_gain.iloc[1:, :-1] += gain_df['CNA'].unstack().transpose().fillna(0.0)
    plt_gain.iloc[:-1, 1:] += gain_df['Mut'].unstack().fillna(0.0)
    plt_gain = plt_gain.iloc[::-1, ::-1]

    for i in range(plt_loss.shape[0]):
        plt_loss.iloc[i, i] = 0.5
    for i in range(plt_gain.shape[0]):
        plt_gain.iloc[i, i] = 0.5

    sns.heatmap(plt_loss, cmap=use_cmap, vmin=0.0, vmax=1.0, center=0.5,
                xticklabels=loss_lbls, yticklabels=loss_lbls,
                ax=loss_ax, cbar=False, square=True)
    sns.heatmap(plt_gain, cmap=use_cmap, vmin=0.0, vmax=1.0, center=0.5,
                xticklabels=gain_lbls, yticklabels=gain_lbls, ax=gain_ax,
                square=True, cbar_kws={'ticks': np.arange(0, 1.1, 0.1)})

    loss_ax.set_title('Loss CNAs', size=27, weight='semibold')
    gain_ax.set_title('Gain CNAs', size=27, weight='semibold')

    loss_ax.set_xlabel('Wild-Type Cutoff', size=22, weight='semibold')
    loss_ax.set_ylabel('CNA Cutoff', size=22, weight='semibold')
    gain_ax.set_xlabel('Wild-Type Cutoff', size=22, weight='semibold')
    gain_ax.set_ylabel('CNA Cutoff', size=22, weight='semibold')

    loss_ax.tick_params(labelsize=13)
    gain_ax.tick_params(labelsize=13)

    cbar = gain_ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=19)
    cbar.set_label(label='AUC', size=28)

    fig.tight_layout()
    plt.savefig(
        os.path.join(plot_dir,
                     "{}_{}_{}.png".format(
                         args.cohort, args.gene, args.classif)),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the success of classifying a gene's CNA status in a given "
        "cohort using different cutoffs for determining CNA status."
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

    loss_df, gain_df = get_aucs(
        load_infer_output(os.path.join(base_dir, 'output',
                                       args.cohort, args.gene, args.classif)),
        args, cdata
        )

    plot_cutoff_aucs(loss_df, gain_df, args, cdata)


if __name__ == '__main__':
    main()

