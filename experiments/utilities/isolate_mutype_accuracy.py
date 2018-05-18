
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts import VariantCohort
from HetMan.predict.classifiers import *

import numpy as np
import pandas as pd

import synapseclient
import dill as pickle

import argparse
from glob import glob
from operator import or_
from functools import reduce


firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_output(out_dir):

    out_list = [
        pd.concat([pd.DataFrame.from_dict(pickle.load(
            open(os.path.join(out_dir, "results/out__cv-{}_task-{}.p".format(
                cv_id, task_id)), 'rb')
            )['Acc'], orient='index') for task_id in range(10)], axis=0)
        for cv_id in range(5)
        ]

    return out_list


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Test a classifier's ability to predict the presence "
                     "of a list of sub-types.")
        )

    # positional command line arguments
    parser.add_argument('mtype_dir', type=str,
                        help='the folder where sub-types are stored')
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')
    parser.add_argument('base_gene', type=str,
                        help='the gene to cross with respect to')

    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument(
        '--tune_splits', type=int, default=8,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=24,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=12,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    if args.verbose:
        print("Starting testing for directory\n{}\nwith "
              "cross-validation ID {} and task ID {} ...".format(
                  args.mtype_dir, args.cv_id, args.task_id))

    mtype_list = sorted(pickle.load(
        open(os.path.join(args.mtype_dir, 'tmp', 'mtype_list.p'), 'rb')))

    # loads the pipeline used for classifying variants, gets the mutated
    # genes for each variant under consideration
    mut_clf = eval(args.classif)
    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                             for mtype in mtype_list]) | {args.base_gene}

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=list(use_genes),
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', data_dir=firehose_dir, syn=syn,
        cv_seed=(args.cv_id + 53) * 7, cv_prop=2/3
        )

    base_mtype = MuType({('Gene', args.base_gene): None})
    base_train_samps = base_mtype.get_samples(cdata.train_mut)
    base_test_samps = base_mtype.get_samples(cdata.test_mut)

    if args.verbose:
        print("Loaded {} sub-types over {} genes which will be tested using "
              "classifier {} in cohort {} with {} samples.".format(
                    len(mtype_list), len(use_genes), args.classif,
                    args.cohort, len(cdata.samples)
                    ))

    out_acc = {mtype: {} for mtype in mtype_list}

    for i, mtype in enumerate(mtype_list):
        if (i % 10) == args.task_id:

            if args.verbose:
                print("Testing {} ...".format(mtype))

            ex_genes = set(gn for gn, _ in mtype.subtype_list())
            clf = mut_clf()

            cur_train_samps = mtype.get_samples(cdata.train_mut)
            cur_test_samps = mtype.get_samples(cdata.test_mut)

            clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                         tune_splits=args.tune_splits,
                         test_count=args.test_count,
                         parallel_jobs=args.parallel_jobs)

            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype]['Base'] = clf.eval_coh(
                cdata, mtype, exclude_genes=ex_genes)

            if (len(cur_train_samps - base_train_samps) > 3
                    and len(cur_test_samps - base_test_samps) > 3):

                print("Null test {}".format(mtype))
                clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                             tune_splits=args.tune_splits,
                             exclude_samps=base_train_samps,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                clf.fit_coh(cdata, mtype,
                        exclude_genes=ex_genes, exclude_samps=base_train_samps)
                out_acc[mtype]['Null'] = clf.eval_coh(
                    cdata, mtype,
                    exclude_genes=ex_genes, exclude_samps=base_test_samps
                    )

            if (len(cur_train_samps & base_train_samps) > 3
                    and len(cur_test_samps & base_test_samps) > 3):

                print("Mut test {}".format(mtype))
                clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                             tune_splits=args.tune_splits,
                             include_samps=base_train_samps,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                clf.fit_coh(cdata, mtype,
                        exclude_genes=ex_genes, include_samps=base_train_samps)
                out_acc[mtype]['Mut'] = clf.eval_coh(
                    cdata, mtype,
                    exclude_genes=ex_genes, include_samps=base_test_samps
                    )

            if (len(cur_train_samps - base_train_samps) > 3
                    and len(cur_test_samps & base_test_samps) > 3):

                print("Null cross {}".format(mtype))
                clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                             tune_splits=args.tune_splits,
                             exclude_samps=base_train_samps,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                clf.fit_coh(cdata, mtype,
                        exclude_genes=ex_genes, exclude_samps=base_train_samps)
                out_acc[mtype]['NullX'] = clf.eval_coh(
                    cdata, mtype,
                    exclude_genes=ex_genes, include_samps=base_test_samps
                    )

            if (len(cur_train_samps & base_train_samps) > 3
                    and len(cur_test_samps - base_test_samps) > 3):

                print("Mut cross {}".format(mtype))
                clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                             tune_splits=args.tune_splits,
                             include_samps=base_train_samps,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                clf.fit_coh(cdata, mtype,
                        exclude_genes=ex_genes, include_samps=base_train_samps)
                out_acc[mtype]['MutX'] = clf.eval_coh(
                    cdata, mtype,
                    exclude_genes=ex_genes, exclude_samps=base_test_samps
                    )

        else:
            del(out_acc[mtype])

    # saves the performance measurements for each variant to file
    out_file = os.path.join(
        args.mtype_dir, 'results',
        'out__cv-{}_task-{}.p'.format(args.cv_id, args.task_id)
        )
    pickle.dump({'Acc': out_acc,
                 'Info': {'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                 open(out_file, 'wb'))


if __name__ == "__main__":
    main()

