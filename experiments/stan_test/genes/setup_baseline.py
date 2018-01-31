
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.mut import VariantCohort
from HetMan.predict.basic.classifiers import *

import pandas as pd
import synapseclient
import dill as pickle
import argparse
from operator import or_

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_accuracies(base_alg, cohort):

    out_lists = [
        [pickle.load(
            open(os.path.join(base_dir, "setup", "baseline_perf",
                              "{}_{}__cv_{}__task_{}.p".format(
                                  cohort, base_alg, cv_id, task_id)),
                 'rb')
            ) for task_id in range(4)]
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
        description='Set up searching for sub-types to detect.'
        )

    # positional command line arguments
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('baseline', type=str, help='baseline algorithm')

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

    # optional command line argument controlling verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments, get the directory where found sub-types
    # will be saved for future use
    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'setup')
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, 'baseline_perf'), exist_ok=True)

    mtype_list = sorted(pickle.load(
        open(os.path.join(out_path,
                          '{}__mtype_list.p'.format(args.cohort)), 'rb')
        ))

    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                             for mtype in mtype_list])

    # log into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=use_genes, mut_levels=['Gene'],
        expr_source='Firehose', data_dir=firehose_dir, syn=syn,
        cv_prop=0.8, cv_seed=151 + 19 * args.cv_id
        )

    clf_base = eval(args.baseline)()
    out_acc = {mtype: -1 for mtype in mtype_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if i % 4 == args.task_id:
            ex_genes = set(gn for gn, _ in mtype.subtype_list())

            if args.verbose:
                print("Testing {} ...".format(mtype))

            clf_base.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                              tune_splits=args.tune_splits,
                              test_count=args.test_count,
                              parallel_jobs=args.parallel_jobs)

            clf_base.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype] = clf_base.eval_coh(cdata, mtype,
                                               exclude_genes=ex_genes)
            print(out_acc[mtype])

        else:
            del(out_acc[mtype])

    # save the list of found non-duplicate sub-types to file
    pickle.dump(
        out_acc,
        open(os.path.join(out_path, 'baseline_perf',
                          '{}_{}__cv_{}__task_{}.p'.format(
                              args.cohort, args.baseline,
                              args.cv_id, args.task_id
                            )), 'wb')
        )


if __name__ == '__main__':
    main()

