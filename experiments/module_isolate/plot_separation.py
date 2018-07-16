
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'separation')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.module_isolate import *

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output

import argparse
import synapseclient
import numpy as np
import pandas as pd
from itertools import product

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def get_separation(iso_df, args, cdata):
    base_phenos = {
        gene: np.array(cdata.train_pheno(MuType({('Gene', gene): None})))
        for gene in args.genes
        }
    module_pheno = np.array(
        cdata.train_pheno(MuType({('Gene', tuple(args.genes)): None})))

    auc_list = pd.Series(index=iso_df.index, dtype=np.float)
    sep_dict = {gene: pd.Series(dtype=np.float) for gene in args.genes}
    prop_list = pd.Series(index=iso_df.index, dtype=np.float)

    for mtype, iso_vals in iso_df.iterrows():
        cur_pheno = np.array(cdata.train_pheno(mtype))

        none_vals = np.concatenate(iso_vals[~module_pheno].values)
        cur_vals = np.concatenate(iso_vals[cur_pheno].values)
        auc_list[mtype] = np.less.outer(none_vals, cur_vals).mean()
        prop_list[mtype] = np.sum(cur_pheno) / np.sum(module_pheno)

        for gene, base_pheno in base_phenos.items():
            rest_stat = base_pheno & ~cur_pheno

            if np.any(rest_stat):
                rest_vals = np.concatenate(iso_vals[rest_stat].values)
                sep_dict[gene][mtype] = np.less.outer(
                    none_vals, rest_vals).mean()

    return auc_list, sep_dict, prop_list


def plot_separation(auc_vals, sep_dict, prop_vals, args, cdata):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'
    fig, axarr = plt.subplots(figsize=(15, 14),
                              nrows=len(args.genes), ncols=len(args.genes))

    for (i, gene2), (j, gene1) in product(enumerate(args.genes), repeat=2):
        aucs_use = auc_vals.loc[
            [mtype for mtype in auc_vals.index
             if (mtype in sep_dict[gene2]
                 and mtype.subtype_list()[0][0] == gene1)]
            ]

        sep_vals = sep_dict[gene2][aucs_use.index]
        axarr[i, j].scatter(aucs_use, sep_vals,
                            s=prop_vals * 91, c='black', alpha=0.17)
        axarr[i, j].plot([-1, 2], [-1, 2], color='#550000',
                         linewidth=1.6, linestyle='--', alpha=0.6)
 
        axarr[i, j].tick_params(length=5.1, labelsize=16)
        axarr[i, j].set_xlim(min(aucs_use.min(), sep_vals.min()) - 0.02, 1)
        axarr[i, j].set_ylim(min(aucs_use.min(), sep_vals.min()) - 0.02, 1)

        if i == (len(args.genes) - 1):
            axarr[i, j].set_xlabel('Isolated {} Mutation AUC'.format(gene1),
                                   fontsize=20, weight='semibold')

        if j == 0:
            axarr[i, j].set_ylabel('Remaining {} AUC'.format(gene2),
                                   fontsize=20, weight='semibold')

    fig.tight_layout(w_pad=0.4, h_pad=0.4)
    fig.savefig(os.path.join(
        plot_dir, "remaining-auc__{}__{}__{}__samps_{}__{}.png".format(
            args.cohort, '_'.join(sorted(args.genes)),
            args.classif, args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot how well expression signatures separate isolated mutation "
        "subtypes from non-mutated samples relative to how they separate "
        "module-mutated samples not belonging to the subtype."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', type=str,
                        help='a set of mutation annotation levels')
    parser.add_argument('genes', type=str, nargs='+',
                        help='a list of mutated genes')
    parser.add_argument('--samp_cutoff', type=int, default=25)
 
    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=args.genes,
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    infer_df = load_infer_output(
        os.path.join(base_dir, 'output', args.cohort,
                     '_'.join(sorted(args.genes)), args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        )
    auc_vals, sep_dict, prop_vals = get_separation(infer_df, args, cdata)

    plot_separation(auc_vals, sep_dict, prop_vals, args, cdata)


if __name__ == '__main__':
    main()

