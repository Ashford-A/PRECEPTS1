
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import list_cohorts, MutationCohort
from HetMan.experiments.transfer_baseline import *
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import pandas as pd
import dill as pickle

from functools import reduce
from operator import or_
from itertools import combinations as combn


def get_cohort_data(syn, expr_source, cohort, samp_cutoff,
                    cv_prop=1.0, cv_seed=None):

    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
    use_genes = gene_df.index[
        (gene_df.loc[
            :, ['Vogelstein', 'Sanger CGC', 'Foundation One', 'MSK-IMPACT']]
            == 'Yes').all(axis=1)
        ]

    cdata = MutationCohort(cohort=cohort, mut_genes=use_genes.tolist(),
                           mut_levels=['Gene', 'Form_base', 'Protein'],
                           expr_source=expr_source, var_source='mc3',
                           copy_source='Firehose', annot_file=annot_file,
                           expr_dir=expr_sources[expr_source],
                           copy_dir=copy_dir, syn=syn,
                           cv_prop=cv_prop, cv_seed=cv_seed)

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str, choices=['Firehose', 'toil'],
                        help="which TCGA expression data source to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup')
    os.makedirs(out_path, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cohorts = list_cohorts(args.expr_source,
                           expr_dir=expr_sources[args.expr_source],
                           copy_dir=copy_dir)
    cohorts -= {'GBM', 'LGG', 'COAD', 'READ', 'KIRC', 'KICH', 'KIRP'}

    muts_dict = {cohort: set() for cohort in cohorts}
    for cohort in cohorts:
        print(cohort)
        cdata = get_cohort_data(
            syn, args.expr_source, cohort, args.samp_cutoff)

        if len(cdata.samples) >= 250:
            muts_list = reduce(or_,
                               [{MuType({('Gene', gene): mtype})
                                 for mtype in muts['Point'].branchtypes(
                                     min_size=args.samp_cutoff)}
                                for gene, muts in cdata.train_mut
                                if ('Scale', 'Point') in muts.allkey()])

            muts_list |= {
                MuType({('Gene', gene): {('Copy', 'HomDel'): None}})
                for gene, muts in cdata.train_mut
                if (('Scale', 'Copy') in muts.allkey()
                    and ('Copy', 'HomDel') in muts['Copy'].allkey()
                    and len(muts['Copy']['HomDel']) >= args.samp_cutoff)
                }
 
            muts_list |= {
                MuType({('Gene', gene): {('Copy', 'HomGain'): None}})
                for gene, muts in cdata.train_mut
                if (('Scale', 'Copy') in muts.allkey()
                    and ('Copy', 'HomGain') in muts['Copy'].allkey()
                    and len(muts['Copy']['HomGain']) >= args.samp_cutoff)
                }
 
            muts_dict[cohort] = {
                mtype for mtype in muts_list
                if (len(mtype.get_samples(cdata.train_mut))
                    <= (len(cdata.samples) - args.samp_cutoff))
                }
            print(len(muts_dict[cohort]))

            muts_combs = {(coh1, coh2): mtypes1 & mtypes2
                          for (coh1, mtypes1), (coh2, mtypes2)
                          in combn(muts_dict.items(), 2) if mtypes1 & mtypes2}
            print('--total: {}'.format(sum(len(mtypes)
                                           for mtypes in muts_combs.values())))

    pickle.dump(
        muts_combs,
        open(os.path.join(out_path,
                          "combs-list_{}__samps-{}.p".format(
                              args.expr_source, args.samp_cutoff
                            )),
             'wb')
        )

    with open(os.path.join(out_path,
                          "combs-count_{}__samps-{}.txt".format(
                              args.expr_source, args.samp_cutoff
                            )),
              'w') as fl:

        fl.write(str(sum(len(mtypes) for mtypes in muts_combs.values())))


if __name__ == '__main__':
    main()

