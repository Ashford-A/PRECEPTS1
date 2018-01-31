
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.mut import VariantCohort
from HetMan.predict.stan_margins import *

import synapseclient
from importlib import import_module
import dill as pickle
import pandas as pd

import argparse
from operator import or_
from functools import reduce

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_accuracies(model_name, cohort, solve_type):

    out_lists = [
        [pickle.load(
            open(os.path.join(base_dir, "models",
                              model_name, cohort, solve_type, "results",
                              "out__cv-{}_task-{}.p".format(cv_id, task_id)),
                 'rb')
            )['Acc'] for task_id in range(10)]
        for cv_id in range(5)
        ]

    out_data = pd.concat(
        [pd.concat(pd.DataFrame.from_dict(out_dict, orient='index')
                   for out_dict in out_list)
         for out_list in out_lists],
        axis=1
        )

    return out_data


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Find the signatures a classifier predicts for a list "
                     "of sub-types.")
        )

    parser.add_argument('out_tag', type=str, help='label of a model')
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('solve_type', type=str, help='how to solve the model')

    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=12,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=8,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    stan_model = import_module(
        'HetMan.experiments.stan_test.genes.models.{}.model'.format(
            args.out_tag)
        )

    if args.verbose:
        print("Starting signature portrayal for cross-validation ID {} and "
              "task ID {} ...".format(args.cv_id, args.task_id))
        print('Using the following Stan model:\n\n{}'.format(
            stan_model.model_code))

    out_path = os.path.join(base_dir, 'models',
                            args.out_tag, args.cohort, args.solve_type)
    mtype_list = sorted(pickle.load(
        open(os.path.join(base_dir, 'setup',
                          '{}__mtype_list.p'.format(args.cohort)), 'rb')
        ))

    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                             for mtype in mtype_list])

    if len(use_genes) != len(mtype_list):
        raise ValueError("Each sub-type to be tested must correspond to "
                         "exactly one unique gene!")

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation ID for this sub-job
    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=use_genes, mut_levels=['Gene'],
        expr_source='Firehose', data_dir=firehose_dir, syn=syn,
        cv_prop=0.8, cv_seed=151 + 19 * args.cv_id
        )

    if args.solve_type == 'optim':
        class UseStan(StanOptimizing, stan_model.ModelClass):
            pass

    elif args.solve_type == 'variat':
        class UseStan(StanVariational, stan_model.ModelClass):
            pass

    elif args.solve_type == 'sampling':
        class UseStan(StanSampling, stan_model.ModelClass):
            pass

    else:
        raise ValueError("Unrecognized 'solve_type' argument!")

    clf_stan = StanPipe(UseStan(stan_model.model_code))
    out_acc = {mtype: -1 for mtype in mtype_list}

    # for each sub-variant, check if it has been assigned to this sub-job
    # according to the given task ID
    for i, mtype in enumerate(mtype_list):
        if i % 10 == args.task_id:
            ex_genes = set(gn for gn, _ in mtype.subtype_list())

            if args.verbose:
                print("Testing {} ...".format(mtype))

            clf_stan.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype] = clf_stan.eval_coh(cdata, mtype,
                                               exclude_genes=ex_genes)

            print(out_acc[mtype])

        else:
            del(out_acc[mtype])

    pickle.dump({'Acc': out_acc,
                 'Info': {'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                open(os.path.join(
                    out_path, 'results', 'out__cv-{}_task-{}.p'.format(
                        args.cv_id, args.task_id)
                    ), 'wb')
               )


if __name__ == "__main__":
    main()

