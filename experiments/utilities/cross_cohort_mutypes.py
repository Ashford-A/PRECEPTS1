
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.predict.basic.classifiers import *

import numpy as np
import pandas as pd

import synapseclient
import dill as pickle

import argparse
from glob import glob
from operator import or_
from functools import reduce

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def get_output_files(out_dir):
    file_list = glob(os.path.join(out_dir, 'out__task-*.p'))

    base_names = [os.path.basename(fl).split('out__')[1] for fl in file_list]
    task_ids = [int(nm.split('task-')[1].split('.p')[0]) for nm in base_names]

    return file_list, task_ids


def load_output(out_dir):
    """Gets the cross-validated AUCs of a set of tested MuTypes.

    Args:
        out_dir (str): The directory where the results were saved.

    Returns:
        out_data (pd.DataFrame), shape = [n_MuTypes, n_cvs]
            How well the given classifier was able to predict the presence
            of each mutation sub-type in each cross-validation run.

    Examples:
        >>> out_data = test_output("HetMan/experiments/subvariant_detection/"
        >>>                        "output/PAAD/rForest/search")

    """
    file_list, task_ids = get_output_files(out_dir)

    return pd.concat([
        pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Cross'],
                               orient='index')
        for fl in file_list
        ]).applymap(lambda x: [[z[0] for z in y] for y in x])


def load_parameters(out_dir):
    file_list, cv_ids, _ = get_output_files(out_dir)

    out_dict = {cv_id: pd.DataFrame([]) for cv_id in set(cv_ids)}
    for cv_id, fl in zip(cv_ids, file_list):

        out_dict[cv_id] = pd.concat([
            out_dict[cv_id],
            pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Par'],
                                   orient='index')
            ])

    return pd.concat(out_dict, axis=1, join='outer')


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Test a classifier's ability to predict the presence "
                     "of a list of sub-types.")
        )

    # positional command line arguments for where input data and output
    # data is to be stored
    parser.add_argument('mtype_file', type=str,
                        help='the pickle file where sub-types are stored')
    parser.add_argument('out_dir', type=str,
                        help='where to save the output of testing sub-types')

    # positional arguments for which cohort of samples and which mutation
    # classifier to use for testing
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')

    # positional arguments controlling CV and task selection
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument('--use_genes', type=str, default=None,
                        help='specify which gene the mutations belong to')

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )

    parser.add_argument(
        '--mut_levels', type=str, default='Form_base__Exon',
        help='a set of mutation levels to consider'
        )

    # optional arguments controlling how classifier tuning is to be performed
    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=16,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=4,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    if args.verbose:
        print("Starting testing for sub-types in\n{}\nwith "
              "task ID {} ...".format(
                  args.mtype_file, args.task_id))

    pair_list = pickle.load(open(args.mtype_file, 'rb'))
    out_file = os.path.join(args.out_dir,
                            'out__task-{}.p'.format(
                                args.task_id))

    mut_clf = eval(args.classif)

    if args.use_genes is None:
        use_genes = reduce(
            or_, [set(gn for gn, _ in mtype1.subtype_list())
                  | set(gn for gn, _ in mtype2.subtype_list())
                  for mtype1, mtype2 in pair_list]
            )

    else:
        use_genes = {args.use_genes}

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    cdata = MutationCohort(cohort=args.cohort, mut_genes=list(use_genes),
                           mut_levels=['Gene'] + args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_seed=9999, cv_prop=1.0)

    base_mtype = MuType({('Gene', tuple(use_genes)): None})
    base_samps = base_mtype.get_samples(cdata.train_mut)

    if args.verbose:
        print("Loaded {} sub-type pairs over {} genes which will be crossed "
              "using classifier {} in cohort {} with {} samples.".format(
                    len(pair_list), len(use_genes), args.classif,
                    args.cohort, len(cdata.samples)
                    ))

    # initialize the dictionaries that will store classification
    # performances and hyper-parameter values
    out_cross = {mtypes: [None, None] for mtypes in pair_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, (mtype1, mtype2) in enumerate(pair_list):
        if (i % args.task_count) == args.task_id:
            clf = mut_clf()

            if args.verbose:
                print("Crossing {} and {} ...".format(mtype1, mtype2))

            samps1 = mtype1.get_samples(cdata.train_mut)
            samps2 = mtype2.get_samples(cdata.train_mut)
            ex_samps = base_samps - (samps1 | samps2)

            if len(samps1 | samps2) <= (len(cdata.samples) - 10):

                if 10 <= len(samps1 - samps2):
                    clf.tune_coh(cdata, mtype1, exclude_genes=use_genes,
                                 exclude_samps=ex_samps,
                                 tune_splits=args.tune_splits,
                                 test_count=args.test_count,
                                 parallel_jobs=args.parallel_jobs)
                    
                    out_cross[(mtype1, mtype2)][0] = clf.infer_coh(
                        cdata, mtype1, exclude_genes=use_genes,
                        force_test_samps=ex_samps,
                        infer_splits=40, infer_folds=4,
                        parallel_jobs=args.parallel_jobs
                        )

                if 10 <= len(samps2 - samps1):
                    clf.tune_coh(cdata, mtype2, exclude_genes=use_genes,
                                 exclude_samps=ex_samps,
                                 tune_splits=args.tune_splits,
                                 test_count=args.test_count,
                                 parallel_jobs=args.parallel_jobs)
                    
                    out_cross[(mtype1, mtype2)][1] = clf.infer_coh(
                        cdata, mtype2, exclude_genes=use_genes,
                        force_test_samps=ex_samps,
                        infer_splits=40, infer_folds=4,
                        parallel_jobs=args.parallel_jobs
                        )

        else:
            del(out_cross[(mtype1, mtype2)])

    # saves the performance measurements and tuned hyper-parameter values
    # for each sub-type to file
    pickle.dump({'Cross': out_cross,
                 'Info': {'TunePriors': mut_clf.tune_priors,
                          'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                 open(out_file, 'wb'))


if __name__ == "__main__":
    main()

