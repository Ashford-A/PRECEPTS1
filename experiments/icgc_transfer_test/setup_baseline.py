
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts.icgc import MutationCohort as ICGCcohort
from HetMan.predict.basic.classifiers import *

import argparse
import synapseclient
import dill as pickle
import pandas as pd

icgc_data_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'


def load_accuracies(base_alg):

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
    parser.add_argument('classif', type=str, help='classification algorithm')
    parser.add_argument('cv_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=24,
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

    cohort_genes = pickle.load(
        open(os.path.join(base_dir, 'setup', 'cohort_genes.p'), 'rb'))
    use_genes = sorted(set(gn for _, gn in cohort_genes))
    
    cdata_icgc = ICGCcohort(
        'PACA-AU', icgc_data_dir, mut_genes=None, samp_cutoff=[1/12, 11/12],
        cv_prop=0.75, cv_seed=(args.cv_id * 9999) + 3
        )

    mut_clf = eval(args.classif)()
    out_acc = {gene: -1 for gene in use_genes}

    # for each sub-variant, check if it has been assigned to this task
    for gene in use_genes:
        use_mtype = MuType({('Gene', gene): None})
        
        if args.verbose:
            print("Testing {} ...".format(use_mtype))
        
        mut_clf.tune_coh(cdata_icgc, use_mtype, exclude_genes={gene},
                         tune_splits=args.tune_splits,
                         test_count=args.test_count,
                         parallel_jobs=args.parallel_jobs)

        mut_clf.fit_coh(cdata_icgc, use_mtype, exclude_genes={gene})
        out_acc[gene] = mut_clf.eval_coh(cdata_icgc, use_mtype,
                                         exclude_genes={gene})

    pickle.dump(
        out_acc,
        open(os.path.join(out_path,
                          '{}__cv_{}.p'.format(args.classif, args.cv_id)
                            ), 'wb')
        )


if __name__ == '__main__':
    main()

