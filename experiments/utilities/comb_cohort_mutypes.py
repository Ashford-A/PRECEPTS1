
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

from operator import or_, add
from functools import reduce
from itertools import combinations as combn

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_output(out_dir):
    
    out_list = [
        pickle.load(open(
            os.path.join(out_dir, "results", "out__cv-{}.p".format(cv_id)),
            'rb'))['Infer']
        for cv_id in range(10)
        ]
    out_dict = {}

    for mtypes in out_list[0]:
        out_dict[mtypes] = {}

        for cx_type in out_list[0][mtypes]:
            out_dict[mtypes][cx_type] = np.array([
                np.mean(reduce(add, x)) for x in zip(*[ols[mtypes][cx_type]
                                                       for ols in out_list])
                ])

    return pd.DataFrame.from_dict(out_dict).T


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
    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')

    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=16,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=16,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments and show info about where the
    # input sub-types are stored and which subset of them will be tested
    args = parser.parse_args()

    mtype_list = sorted(pickle.load(
        open(os.path.join(args.mtype_dir, 'tmp', 'mtype_list.p'), 'rb')))

    # loads the pipeline used for classifying variants, gets the mutated
    # genes for each variant under consideration
    mut_clf = eval(args.classif)
    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                             for mtype in mtype_list])

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=use_genes,
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', data_dir=firehose_dir, syn=syn,
        cv_seed=(args.cv_id + 12) * 71, cv_prop=1.0
        )

    infer_mats = {(mtype1, mtype2): {}
                  for mtype1, mtype2 in combn(mtype_list, 2)}
    for mtype1, mtype2 in combn(mtype_list, 2):
        print('{} and {}'.format(mtype1, mtype2))

        ex_genes = set(gn for gn, _ in mtype1.subtype_list())
        ex_genes |= set(gn for gn, _ in mtype2.subtype_list())
        clf = mut_clf()

        samps1 = mtype1.get_samples(cdata.train_mut)
        samps2 = mtype2.get_samples(cdata.train_mut)
        if len(samps1 | samps2) <= (len(cdata.samples) - 10):

            if 10 <= len(samps1 & samps2):

                clf.tune_coh(cdata, mtype1, exclude_genes=ex_genes,
                             exclude_samps=samps1 ^ samps2,
                             tune_splits=args.tune_splits,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                infer_mats[(mtype1, mtype2)]['Both'] = clf.infer_coh(
                    cdata, mtype1, exclude_genes=ex_genes,
                    force_test_samps=samps1 ^ samps2,
                    infer_splits=20, infer_folds=4,
                    parallel_jobs=args.parallel_jobs
                    )

            if 10 <= len(samps1 - samps2):

                clf.tune_coh(cdata, mtype1, exclude_genes=ex_genes,
                             exclude_samps=samps2,
                             tune_splits=args.tune_splits,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                infer_mats[(mtype1, mtype2)]['Mtype1'] = clf.infer_coh(
                    cdata, mtype1, exclude_genes=ex_genes,
                    force_test_samps=samps2,
                    infer_splits=20, infer_folds=4,
                    parallel_jobs=args.parallel_jobs
                    )


            if 10 <= len(samps2 - samps1):

                clf.tune_coh(cdata, mtype2, exclude_genes=ex_genes,
                             exclude_samps=samps1,
                             tune_splits=args.tune_splits,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                infer_mats[(mtype1, mtype2)]['Mtype2'] = clf.infer_coh(
                    cdata, mtype2, exclude_genes=ex_genes,
                    force_test_samps=samps1,
                    infer_splits=20, infer_folds=4,
                    parallel_jobs=args.parallel_jobs
                    )

            if (mtype1.get_levels() == mtype2.get_levels()
                    and 10 <= len(samps2 ^ samps1)):

                clf.tune_coh(cdata, mtype1 | mtype2, exclude_genes=ex_genes,
                             exclude_samps=samps1 & samps2,
                             tune_splits=args.tune_splits,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

                infer_mats[(mtype1, mtype2)]['Diff'] = clf.infer_coh(
                    cdata, mtype1 | mtype2, exclude_genes=ex_genes,
                    force_test_samps=samps1 & samps2,
                    infer_splits=20, infer_folds=4,
                    parallel_jobs=args.parallel_jobs
                    )

    # saves the performance measurements for each variant to file
    out_file = os.path.join(
        args.mtype_dir, 'results',
        'out__cv-{}.p'.format(args.cv_id)
        )
    pickle.dump({'Infer': infer_mats,
                 'Info': {'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                 open(out_file, 'wb'))


if __name__ == "__main__":
    main()

