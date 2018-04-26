
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'tuning')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.describe.transformers import *

import synapseclient
import argparse
import numpy as np
from itertools import product

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def plot_tuning_gene(cdata, args, tune_params, pca_comps=(0, 1)):
    tune_size1 = len(tune_params[0][1])
    tune_size2 = len(tune_params[1][1])

    fig, axarr = plt.subplots(nrows=tune_size1, ncols=tune_size2,
                              figsize=(tune_size2 * 5 - 1, tune_size1 * 5))
    fig.tight_layout(pad=1.6)

    for ax in axarr.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pca_comps = np.array(pca_comps)
    trans_dict = dict()
    base_pheno = np.array(cdata.train_pheno(MuType(cdata.train_mut.allkey())))

    mut_clr = sns.light_palette((1/3, 0, 0), input="rgb",
                                n_colors=5, reverse=True)[1]

    for prms in product(*[[(x[0], y) for y in x[1]]
                          for x in tune_params[:2]]):

        mut_trans = eval(args.transform)().set_params(
            **dict(prms + (('fit__random_state', 903), )))
        trans_dict[prms] = mut_trans.fit_transform_coh(cdata)[:, pca_comps]

    for i in range(tune_size1):
        axarr[i, 0].set_ylabel(
            '{}: {}'.format(tune_params[0][0], tune_params[0][1][i]), size=21)

    for j in range(tune_size2):
        axarr[tune_size1 - 1, j].set_xlabel(
            '{}: {}'.format(tune_params[1][0], tune_params[1][1][j]), size=21)

    for i, j in product(range(tune_size1), range(tune_size2)):
        trans_expr = trans_dict[((tune_params[0][0], tune_params[0][1][i]),
                                 (tune_params[1][0], tune_params[1][1][j]))]

        axarr[i, j].scatter(
            trans_expr[~base_pheno, 0], trans_expr[~base_pheno, 1],
            marker='o', s=12, c='0.4', alpha=0.25, edgecolor='none'
            )

        axarr[i, j].scatter(
            trans_expr[base_pheno, 0], trans_expr[base_pheno, 1],
            marker='o', s=35, c=mut_clr, alpha=0.4, edgecolor='none'
            )

    fig.savefig(os.path.join(
        plot_dir, "{}__gene_{}_{}__comps_{}-{}__{}.png".format(
            args.transform, args.gene, args.cohort,
            pca_comps[0], pca_comps[1], tune_params[-1][1]
            )),
        dpi=250, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('transform', type=str)
    parser.add_argument('cohort', type=str, help='a cohort in TCGA')
    parser.add_argument('gene', type=str)

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', expr_dir=firehose_dir, cv_prop=1.0, syn=syn
        )

    tune_params = (('fit__n_neighbors', (5, 10, 15)),
                   ('fit__metric', ('euclidean', 'correlation', 'cosine',
                                    'manhattan', 'chebyshev')),
                   ('lbl', 'base2'))
    #tune_params = (('fit__learning_rate', (50, 200, 750)),
    #               ('fit__perplexity', (5, 15, 30, 40, 50)),
    #               ('lbl', 'base'))

    plot_tuning_gene(cdata, args, tune_params)


if __name__ == "__main__":
    main()

