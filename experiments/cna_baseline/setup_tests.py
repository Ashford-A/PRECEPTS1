
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.cna_baseline import *

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
import synapseclient
import dill as pickle


def get_cohort_data(expr_source, cohort,
                    samp_cutoff, train_prop=1.0, cv_seed=0):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(
        cohort=cohort, mut_genes=None, mut_levels=['Gene', 'Form'],
        expr_source=expr_source, expr_dir=expr_sources[expr_source],
        var_source='mc3', copy_dir=copy_dir, copy_source='Firehose', syn=syn,
        copy_discrete=True, samp_cutoff=0, cv_prop=train_prop, cv_seed=cv_seed
        )

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str, choices=['Firehose', 'toil'],
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

    cna_mtypes = {MuType({('Form', 'HomDel'): None}),
                  MuType({('Form', 'HomGain'): None}),
                  MuType({('Form', ('HomDel', 'HetDel')): None}),
                  MuType({('Form', ('HomGain', 'HetGain')): None}),
                  MuType({('Form', ('HomGain', 'HomDel')): None})}

    samp_min = args.samp_cutoff
    samp_max = len(cdata.samples) - args.samp_cutoff
    use_mtypes = set()

    for gene, muts in cdata.train_mut:
        use_mtypes |= {
            MuType({('Gene', gene): cna_mtype}) for cna_mtype in cna_mtypes
            if samp_min <= len(cna_mtype.get_samples(muts)) <= samp_max
            }

    pickle.dump(
        sorted(use_mtypes),
        open(os.path.join(out_path,
                          "mtypes-list_{}__{}__samps-{}.p".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
             'wb')
        )

    with open(os.path.join(out_path,
                          "mtypes-count_{}__{}__samps-{}.txt".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
              'w') as fl:

        fl.write(str(len(use_mtypes)))


if __name__ == '__main__':
    main()

