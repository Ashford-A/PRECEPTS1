
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts.icgc import MutationCohort as ICGCcohort
from HetMan.predict.basic.classifiers import *

import argparse
import dill as pickle
import pandas as pd

icgc_data_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/ICGC'


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Do baseline classifier testing by training and "
                     "testing within ICGC PACA-AU.")
        )

    # positional command line arguments
    parser.add_argument('classif', type=str, help='classification algorithm')
    parser.add_argument('mtypes', type=str, help='a list of mutation types')
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

    cohort_mtypes = pickle.load(
        open(os.path.join(base_dir, 'setup',
                          'cohort_{}.p'.format(args.mtypes)),
             'rb')
        )

    if args.mtypes == 'genes':
        use_mtypes = sorted(set(MuType({('Gene', gn): None})
                                for _, gn in cohort_mtypes))

    else:
        use_mtypes = sorted(set(mtype for _, mtype in cohort_mtypes))
    
    cdata_icgc = ICGCcohort(
        'PACA-AU', icgc_data_dir, mut_genes=None, samp_cutoff=[1/12, 11/12],
        cv_prop=0.75, cv_seed=(args.cv_id * 9999) + 3
        )

    mut_clf = eval(args.classif)()
    out_acc = {mtype: -1 for mtype in use_mtypes}

    for use_mtype in use_mtypes:
        if args.verbose:
            print("Testing {} ...".format(use_mtype))

        use_gns = set(gn for gn, _ in use_mtype.subtype_list())
        mut_clf.tune_coh(cdata_icgc, use_mtype, exclude_genes=use_gns,
                         tune_splits=args.tune_splits,
                         test_count=args.test_count,
                         parallel_jobs=args.parallel_jobs)

        mut_clf.fit_coh(cdata_icgc, use_mtype, exclude_genes=use_gns)
        out_acc[use_mtype] = mut_clf.eval_coh(cdata_icgc, use_mtype,
                                              exclude_genes=use_gns)

    pickle.dump(
        out_acc,
        open(os.path.join(out_path,
                          '{}_{}__cv_{}.p'.format(args.classif, args.mtypes,
                                                  args.cv_id)
                            ), 'wb')
        )


if __name__ == '__main__':
    main()

