
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

import pandas as pd
from HetMan.features.cohorts.tcga import MutationCohort

import argparse
import synapseclient
import dill as pickle


def get_cohort_data(expr_source, syn_root, cohort, samp_cutoff):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    expr_dir = pd.read_csv(
        open(os.path.join(base_dir, 'expr_sources.txt'), 'r'),
        sep='\t', header=None, index_col=0
        ).loc[expr_source].iloc[0]
 
    cdata = MutationCohort(
        cohort=cohort, mut_genes=None, mut_levels=['Gene'], cv_prop=1.0,
        expr_source=expr_source, expr_dir=expr_dir, var_source='mc3',
        syn=syn, samp_cutoff=samp_cutoff
        )

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str, choices=['Firehose', 'toil'],
                        help="which TCGA expression data source to use")
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")

    parser.add_argument(
        'syn_root', type=str,
        help="the root cache directory for data downloaded from Synapse"
        )

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup')
    os.makedirs(out_path, exist_ok=True)

    cdata = get_cohort_data(args.expr_source, args.syn_root,
                            args.cohort, args.samp_cutoff)
    gene_list = {gene for gene, muts in cdata.train_mut
                 if (args.samp_cutoff <= len(muts)
                     <= (len(cdata.samples) - args.samp_cutoff))}

    pickle.dump(
        sorted(gene_list),
        open(os.path.join(out_path,
                          "genes-list_{}__{}__samps-{}.p".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
             'wb')
        )

    with open(os.path.join(out_path,
                          "genes-count_{}__{}__samps-{}.txt".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
              'w') as fl:

        fl.write(str(len(gene_list)))


if __name__ == '__main__':
    main()

