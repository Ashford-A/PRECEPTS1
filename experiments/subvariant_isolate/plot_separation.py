
import os
base_dir = os.path.dirname(__file__)
plot_dir = os.path.join(base_dir, 'plots', 'separation')

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.subvariant_isolate import firehose_dir

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities import load_infer_output

import argparse
import synapseclient
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def get_separation(iso_df, args, cdata):
    base_pheno = np.array(cdata.train_pheno(MuType(cdata.train_mut.allkey())))

    auc_list = pd.Series(index=iso_df.index, dtype=np.float)
    sep_list = pd.Series(index=iso_df.index, dtype=np.float)
    prop_list = pd.Series(index=iso_df.index, dtype=np.float)

    for mtype, iso_vals in iso_df.iterrows():
        cur_pheno = np.array(cdata.train_pheno(mtype))

        none_vals = np.concatenate(iso_vals[~base_pheno].values)
        cur_vals = np.concatenate(iso_vals[cur_pheno].values)
        rest_vals = np.concatenate(iso_vals[base_pheno & ~cur_pheno].values)

        auc_list[mtype] = np.less.outer(none_vals, cur_vals).mean()
        sep_list[mtype] = np.less.outer(none_vals, rest_vals).mean()
        prop_list[mtype] = np.sum(cur_pheno) / np.sum(base_pheno)

    return auc_list, sep_list, prop_list


def plot_separation(auc_vals, sep_vals, prop_vals, args, cdata):
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.edgecolor'] = '0.05'

    fig, ax = plt.subplots(figsize=(15, 14))
    ax.scatter(auc_vals, sep_vals, s=prop_vals * 131, c='black', alpha=0.19)
    ax.plot([-1, 2], [-1, 2],
            linewidth=1.7, linestyle='--', color='#550000', alpha=0.6)

    ax.tick_params(length=4.3, labelsize=17)
    ax.set_xlim(min(auc_vals.min(), sep_vals.min()) - 0.02, 1)
    ax.set_ylim(min(auc_vals.min(), sep_vals.min()) - 0.02, 1)

    ax.set_xlabel('Isolated Mutation AUC', fontsize=22, weight='semibold')
    ax.set_ylabel('Remaining {} AUC'.format(args.gene),
                  fontsize=22, weight='semibold')

    fig.savefig(os.path.join(
        plot_dir, "remaining-auc__{}_{}__{}__samps_{}__{}.png".format(
            args.cohort, args.gene, args.classif,
            args.samp_cutoff, args.mut_levels
            )),
        dpi=300, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        "Plot how well expression signatures separate isolated mutation "
        "subtypes from non-mutated samples relative to how they separate "
        "mutated samples non belonging to the subtype."
        )

    parser.add_argument('cohort', help='a TCGA cohort')
    parser.add_argument('gene', help='a mutated gene')
    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument('mut_levels', default='Form_base__Exon',
                        help='a set of mutation annotation levels')
    parser.add_argument('samp_cutoff', default=20)
    
    parser.add_argument(
        'syn_root', type=str,
        help="the root cache directory for data downloaded from Synapse"
        )

    args = parser.parse_args()
    os.makedirs(plot_dir, exist_ok=True)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = args.syn_root
    syn.login()

    cdata = MutationCohort(cohort=args.cohort, mut_genes=[args.gene],
                           mut_levels=args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_prop=1.0)

    infer_df = load_infer_output(
        os.path.join(base_dir, 'output', args.cohort, args.gene, args.classif,
                     'samps_{}'.format(args.samp_cutoff), args.mut_levels)
        )
    auc_vals, sep_vals, prop_vals = get_separation(infer_df, args, cdata)

    plot_separation(auc_vals, sep_vals, prop_vals, args, cdata)


if __name__ == '__main__':
    main()

