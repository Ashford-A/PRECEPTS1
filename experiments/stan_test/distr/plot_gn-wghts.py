
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'gene-weights')

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.experiments.stan_test.distr.fit_models import load_vars

import argparse
import synapseclient
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'


def plot_weights_cov(weights_df, args, cdata):
    fig, (ax_neg, ax_pos) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2,
                                         sharex=False, sharey=True)

    gn_means = weights_df.mean(axis=1)
    gn_stds = np.log10(weights_df.std(axis=1))
    gn_strengths = np.log10(np.absolute(gn_means))
    gn_directs = np.sign(gn_means)

    strgth_range = np.percentile(gn_strengths, q=(0.5, 100))
    kern_bw = (strgth_range[1] - strgth_range[0]) / 43

    sns.kdeplot(gn_strengths[gn_means < 0], gn_stds[gn_means < 0], ax=ax_neg,
                cmap='Reds', alpha=0.7, shade=True, shade_lowest=False,
                bw=kern_bw, n_levels=50, gridsize=250)
    sns.kdeplot(gn_strengths[gn_means > 0], gn_stds[gn_means > 0], ax=ax_pos,
                cmap='Blues', alpha=0.7, shade=True, shade_lowest=False,
                bw=kern_bw, n_levels=50, gridsize=250)
    
    ax_neg.set_xlim([strgth_range[0], strgth_range[1]])
    ax_pos.set_xlim([strgth_range[1], strgth_range[0]])

    fig.savefig(
        os.path.join(plot_dir,
                     'cov__{}-{}_{}-{}.png'.format(
                         args.model_name, args.solve_method,
                         args.cohort, args.gene
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the distributions of gene weight coefficients inferred by a "
        "given Stan classifier trained to predict the mutation status of a "
        "gene in a given TCGA cohort."
        )

    parser.add_argument('model_name', type=str, help="label of a Stan model")
    parser.add_argument('solve_method', type=str,
                        help=("method used to obtain estimates for the "
                              "parameters of the model"))

    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('gene', type=str, help="a mutated gene")

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)
    vars_dict = load_vars(args.model_name, args.solve_method,
                          args.cohort, args.gene)

    if 'gn_wghts' not in vars_dict:
        raise ValueError("Can only plot inferred gene weights for a model "
                         "that includes them as variables!")

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

    wghts_df = pd.DataFrame(vars_dict['gn_wghts'],
                            index=sorted(cdata.genes - {args.gene}))
    plot_weights_cov(wghts_df, args, cdata)


if __name__ == "__main__":
    main()

