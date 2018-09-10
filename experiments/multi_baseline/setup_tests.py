
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.multi_baseline import *
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_, and_
from itertools import combinations as combn


def get_cohort_data(expr_source, cohort, samp_cutoff,
                    cv_prop=1.0, cv_seed=None):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'Sanger CGC', 'Foundation One', 'MSK-IMPACT']]
            == 'Yes').sum(axis=1) == 4
        ]

    source_info = expr_source.split('__')
    source_base = source_info[0]
    collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

    cdata = MutationCohort(
        cohort=cohort, mut_genes=use_genes.tolist(),
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source=source_base, var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=expr_sources[expr_source],
        copy_dir=copy_dir, collapse_txs=collapse_txs,
        syn=syn, cv_prop=cv_prop, cv_seed=cv_seed
        )

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        choices=list(expr_sources.keys()),
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup')
    os.makedirs(out_path, exist_ok=True)
    cdata = get_cohort_data(args.expr_source, args.cohort, args.samp_cutoff)

    muts_list = reduce(
        or_,
        [{MuType({('Gene', gene): mtype})
          for mtype in muts['Point'].branchtypes(min_size=args.samp_cutoff)}
         for gene, muts in cdata.train_mut
         if ('Scale', 'Point') in muts.allkey()]
        )

    muts_list |= {MuType({('Gene', gene): {('Copy', 'HomDel'): None}})
                  for gene, muts in cdata.train_mut
                  if (('Scale', 'Copy') in muts.allkey()
                      and ('Copy', 'HomDel') in muts['Copy'].allkey()
                      and len(muts['Copy']['HomDel']) >= args.samp_cutoff)}

    muts_list |= {MuType({('Gene', gene): {('Copy', 'HomGain'): None}})
                  for gene, muts in cdata.train_mut
                  if (('Scale', 'Copy') in muts.allkey()
                      and ('Copy', 'HomGain') in muts['Copy'].allkey()
                      and len(muts['Copy']['HomGain']) >= args.samp_cutoff)}

    muts_list |= {MuType({('Gene', gene): {('Scale', 'Point'): None}})
                  for gene, muts in cdata.train_mut
                  if (('Scale', 'Point') in muts.allkey()
                      and len(muts['Point'].allkey()) > 1
                      and len(muts['Point']) >= args.samp_cutoff)}

    muts_list = {mtype for mtype in muts_list
                 if (len(mtype.get_samples(cdata.train_mut))
                     <= (len(cdata.samples) - args.samp_cutoff))}

    combs_list = {mtypes for mtypes in combn(muts_list, 2)
                  if (reduce(and_, mtypes).is_empty()
                      and len(reduce(and_, [mtype.get_samples(cdata.train_mut)
                                            for mtype in mtypes])) == 0)}

    pickle.dump(
        sorted(combs_list),
        open(os.path.join(out_path,
                          "combs-list_{}__{}__samps-{}.p".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
             'wb')
        )

    with open(os.path.join(out_path,
                          "combs-count_{}__{}__samps-{}.txt".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
              'w') as fl:

        fl.write(str(len(combs_list)))


if __name__ == '__main__':
    main()

