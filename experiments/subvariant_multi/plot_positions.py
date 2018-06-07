
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'position')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output

import argparse
import synapseclient
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.lines import Line2D

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def plot_position(infer_vals, args, cdata, mtype1, mtype2):
    fig, ax = plt.subplots(figsize=(15, 14))

    base_pheno = np.array(cdata.train_pheno(
        MuType({('Gene', args.gene): None})))
    pheno1 = np.array(cdata.train_pheno(mtype1))
    pheno2 = np.array(cdata.train_pheno(mtype2))

    use_vals = [np.concatenate(infer_vals[pheno].values)
                for pheno in [~base_pheno, pheno2, pheno1,
                              base_pheno & (~pheno1 & ~pheno2)]]

    auc_mtype1 = np.greater.outer(use_vals[2][:, 0], use_vals[0][:, 0]).mean()
    auc_mtype2 = np.greater.outer(use_vals[1][:, 1], use_vals[0][:, 1]).mean()

    plt_bound = np.max(np.absolute([
        infer_vals.apply(np.min).quantile(0.01),
        infer_vals.apply(np.max).quantile(0.99), 1.1
        ]))

    use_clrs = ['0.5', '#9B5500', '#044063', '#5C0165']
    use_lws = [2.1, None, None, 2.4]
    use_lvls = [19, 5, 5, 14]
    use_alphas = [0.43, 0.55, 0.55, 0.65]
    shade_stat = [False, True, True, False]

    for vals, use_clr, use_lw, use_alpha, use_lvl, shd in zip(
            use_vals, use_clrs, use_lws, use_alphas, use_lvls, shade_stat):

        sns.kdeplot(vals[:, 0], vals[:, 1],
                    cmap=sns.light_palette(use_clr, as_cmap=True), shade=shd,
                    alpha=use_alpha, shade_lowest=False, linewidths=use_lw,
                    bw=plt_bound / 37, gridsize=250, n_levels=use_lvl)

    ax.text(plt_bound * -0.86, plt_bound * 0.27,
            "AUC: {:.3f}".format(auc_mtype2),
            size=24, color='#9B5500', alpha=0.76)
    ax.text(plt_bound * 0.51, plt_bound * -0.53,
            "AUC: {:.3f}".format(auc_mtype1),
            size=24, color='#044063', alpha=0.76)

    plt.legend(
        [Line2D([0], [0], color=use_clr, lw=9.2) for use_clr in use_clrs],
        ["Wild-Type", str(mtype2), str(mtype1),
         "Remaining {} Mutants".format(args.gene)],
        fontsize=21, loc=4
        )

    plt.xlim(-plt_bound, plt_bound)
    plt.ylim(-plt_bound, plt_bound)

    plt.xlabel('Inferred {} Score'.format(mtype1), size=28, weight='semibold')
    plt.ylabel('Inferred {} Score'.format(mtype2), size=28, weight='semibold')

    plt.savefig(
        os.path.join(plot_dir, args.cohort, args.gene, args.mut_levels,
                     "{}__xx__{}__{}__{}.png".format(
                         mtype1.get_label(), mtype2.get_label(),
                         args.model_name, args.solve_method
                        )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot the positions predicted for each sample in a given cohort by a "
        "multi-task model trained on pairs of mutation subtypes of a gene in "
        "two-dimensional inferred label space."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('mut_levels',
                        help='a set of mutation annotation levels')

    parser.add_argument('model_name', help='a Stan multi-task learning model')
    parser.add_argument('solve_method', choices=['optim', 'variat', 'sampl'],
                        help='method used to obtain Stan parameter estimates')

    # parse command-line arguments, create directory where plots will be saved
    args = parser.parse_args()
    os.makedirs(
        os.path.join(plot_dir, args.cohort, args.gene, args.mut_levels),
        exist_ok=True
        )

    multi_df = load_infer_output(os.path.join(
        base_dir, 'output', args.cohort, args.gene, args.mut_levels,
        args.model_name, args.solve_method
        ))

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=[args.gene],
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    for (mtype1, mtype2), infer_vals in multi_df.iterrows():
        plot_position(infer_vals, args, cdata, mtype1, mtype2)


if __name__ == '__main__':
    main()

