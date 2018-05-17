
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'labels')

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.stan_test.distr.fit_models import load_output

import argparse
import synapseclient
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'
wt_clr = '0.29'
mut_clr = sns.hls_palette(1, l=.51, s=.88)[0]


def plot_label_distribution(out_data, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))

    # get the median mutation score for each sample across cross-validation
    # runs, use the range of these scores to set plotting parameters
    out_meds = np.percentile(out_data, q=50, axis=1)
    kern_bw = (np.max(out_meds) - np.min(out_meds)) / 38
    plt_xmax = np.max(np.absolute(out_data)) * 1.1

    # get mutation status for the given gene in the given TCGA cohort
    use_mtype = MuType({('Gene', args.gene): None})
    mtype_stat = np.array(cdata.train_pheno(use_mtype))

    # calculates the classifier AUC for predicting mutation status based on
    # its inferred labels for each cross-validation run
    label_aucs = np.apply_along_axis(
        lambda vals: np.greater.outer(
            vals[mtype_stat], vals[~mtype_stat]).mean(),
        axis=0, arr=out_data
        )

    # plots distribution of wild-type label medians
    ax = sns.kdeplot(out_meds[~mtype_stat], color=wt_clr, alpha=0.7,
                     shade=False, linewidth=3.4, bw=kern_bw, gridsize=1000,
                     label='Wild-Type')

    # plots distribution of mutant label medians
    ax = sns.kdeplot(out_meds[mtype_stat], color=mut_clr, alpha=0.7,
                     shade=False, linewidth=3.4, bw=kern_bw, gridsize=1000,
                     label='{} Mutant'.format(args.gene))

    # plots distribution of wild-type and mutant labels individually for
    # each cross-validation run
    for i in range(out_data.shape[1]):
        ax = sns.kdeplot(out_data[~mtype_stat, i],
                         shade=True, alpha=0.04, linewidth=0, color=wt_clr,
                         bw=kern_bw, gridsize=1000)
        ax = sns.kdeplot(out_data[mtype_stat, i],
                         shade=True, alpha=0.04, linewidth=0, color=mut_clr,
                         bw=kern_bw, gridsize=1000)

    # display interquartile range of cross-validation run AUCs
    ax.text(-plt_xmax * 0.96, ax.get_ylim()[1] * 0.92,
            "AUCs: {:.3f} - {:.3f}".format(
                *np.percentile(label_aucs, q=(25, 75))),
            size=15)

    # set plot legend and axis characteristics
    plt.legend(frameon=False, prop={'size': 17})
    plt.xlim(-plt_xmax, plt_xmax)
    plt.xlabel('Inferred Mutation Score', fontsize=19, weight='semibold')
    plt.ylabel('Density', fontsize=19, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     'distribution__{}-{}_{}-{}.png'.format(
                         args.model_name, args.solve_method,
                         args.cohort, args.gene
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def plot_label_stability(out_data, args, cdata):
    fig, ax = plt.subplots(figsize=(13, 8))

    wt_cmap = sns.light_palette(wt_clr, as_cmap=True)
    mut_cmap = sns.light_palette(mut_clr, as_cmap=True)

    use_mtype = MuType({('Gene', args.gene): None})
    mtype_stat = np.array(cdata.train_pheno(use_mtype))
    out_means = np.mean(out_data, axis=1)
    out_sds = np.std(out_data, axis=1)
    plt_xmax = np.max(np.absolute(out_means)) * 1.1

    ax = sns.kdeplot(out_means[~mtype_stat], out_sds[~mtype_stat],
                     cmap=wt_cmap, linewidths=2.1, alpha=0.8,
                     gridsize=1000, n_levels=34)
    ax.text(np.percentile(out_means[~mtype_stat], q=39),
            np.percentile(out_sds[~mtype_stat], q=99.3),
            "Wild-Type", size=17, color=wt_clr)

    ax = sns.kdeplot(out_means[mtype_stat], out_sds[mtype_stat],
                     cmap=mut_cmap, linewidths=2.1, alpha=0.8,
                     gridsize=1000, n_levels=34)
    ax.text(np.percentile(out_means[mtype_stat], q=61),
            np.percentile(out_sds[mtype_stat], q=99.3),
            "{} Mutant".format(args.gene), size=17, color=mut_clr)

    plt.xlim(-plt_xmax, plt_xmax)
    plt.ylim(0, np.max(out_sds) * 1.05)
    plt.xlabel('Mutation Score CV Mean', fontsize=19, weight='semibold')
    plt.ylabel('Mutation Score CV SD', fontsize=19, weight='semibold')

    fig.savefig(
        os.path.join(plot_dir,
                     'stability__{}-{}_{}-{}.png'.format(
                         args.model_name, args.solve_method,
                         args.cohort, args.gene
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the distributions of perturbation scores separated by mutation "
        "status as inferred by a Stan mutation classifier trained on a gene "
        "in a given TCGA cohort."
        )

    # positional command-line arguments regarding the Stan model used to
    # obtain the sample mutation scores
    parser.add_argument('model_name', type=str, help="label of a Stan model")
    parser.add_argument('solve_method', type=str,
                        help=("method used to obtain estimates for the "
                              "parameters of the model"))

    # positional command line arguments regarding the samples and the mutation
    # classification task on which the model was trained
    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('gene', type=str, help="a mutated gene")

    # parse command line arguments, ensure directory where plots will be saved
    # exists, load inferred mutation scores from each cross-validation run
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)
    infer_mat = load_output(args.model_name, args.solve_method,
                            args.cohort, args.gene)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/CompBio'
                                '/mgrzad/input-data/synapse')
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', expr_dir=firehose_dir, var_source='mc3',
        syn=syn, cv_prop=1.0
        )

    plot_label_distribution(infer_mat, args, cdata)
    plot_label_stability(infer_mat, args, cdata)


if __name__ == "__main__":
    main()

