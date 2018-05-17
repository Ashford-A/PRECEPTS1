
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'expression')

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.stan_test.distr.fit_models import load_output

import numpy as np
import pandas as pd

import argparse
import synapseclient
from itertools import chain, combinations

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'


def plot_subtype_expression(out_data, args, cdata, use_levels):
    fig, ax = plt.subplots(figsize=(13, 8))

    use_phenos = {
        mtype: np.array(cdata.train_pheno(mtype))
        for mtype in cdata.train_mut.branchtypes(sub_levels=use_levels)
        }

    use_phenos = {mtype: use_pheno for mtype, use_pheno in use_phenos.items()
                  if np.sum(use_pheno) >= 15}
    subtype_cmap = sns.hls_palette(len(use_phenos), l=.54, s=.87)

    out_means = np.mean(out_data, axis=1)
    plt_xmax = np.max(np.absolute(out_means)) * 1.13

    all_mtype = MuType(cdata.train_mut.allkey())
    gene_expr, all_pheno = cdata.train_data(all_mtype,
                                            include_genes={args.gene})
    gene_expr = np.array(gene_expr[args.gene])

    ax = sns.kdeplot(out_means[~all_pheno], gene_expr[~all_pheno],
                     cmap=sns.light_palette('0.5', as_cmap=True),
                     linewidths=1.7, alpha=0.7, gridsize=500, n_levels=32)
    ax.text(np.min(out_means[~all_pheno]), np.min(gene_expr[~all_pheno]),
            'Wild-Type', size=15, color='0.5')

    for (mtype, pheno), use_clr in zip(use_phenos.items(), subtype_cmap):
        use_cmap = sns.light_palette(use_clr, as_cmap=True)

        ax = sns.kdeplot(out_means[pheno], gene_expr[pheno],
                         cmap=use_cmap, linewidths=3.6, alpha=0.4,
                         gridsize=500, n_levels=3)

    plt.xlim(-plt_xmax, plt_xmax)
    plt.xlabel('Mutation Score CV Mean', fontsize=18, weight='semibold')
    plt.ylabel('{} RNA-seq Expression'.format(args.gene),
               fontsize=18, weight='semibold')

    plt.legend([Line2D([0], [0], color=use_clr, lw=6.8)
                for use_clr in subtype_cmap],
               [str(mtype) for mtype in use_phenos],
               fontsize=11, ncol=2, frameon=False)

    fig.savefig(
        os.path.join(plot_dir,
                     'subtypes__{}-{}_{}-{}_levels_{}.png'.format(
                         args.model_name, args.solve_method,
                         args.cohort, args.gene, '__'.join(use_levels)
                        )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the distribution of labels by mutation subtype returned by a "
        "Stan classifier trained to predict all the mutations for a given "
        "gene in a TCGA cohort."
        )

    parser.add_argument('model_name', type=str, help="label of a Stan model")
    parser.add_argument('solve_method', type=str,
                        help=("method used to obtain estimates for the "
                              "parameters of the model"))

    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('gene', type=str, help="a mutated gene")

    parser.add_argument('mut_levels', nargs='*',
                        default=['Form_base', 'Exon'],
                        help="which mutation annotation levels to consider")

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
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=args.mut_levels,
        expr_source='Firehose', expr_dir=firehose_dir, var_source='mc3',
        syn=syn, cv_prop=1.0
        )

    for use_levels in chain.from_iterable(combinations(args.mut_levels, r)
                                          for r in range(
                                              1, len(args.mut_levels) + 1)):
        plot_subtype_expression(infer_mat, args, cdata, use_levels)


if __name__ == "__main__":
    main()

