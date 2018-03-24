
"""Testing the classifiability of a list of variant types in a single cohort.

This script takes a list of variant sub-types and tests how well a given
classification pipeline can predict their presence in a cohort of samples.
A sub-type can be any subset of the mutations present in a gene or a group of
genes, as defined by shared properties. These properties can include form
(i.e. splice site mutations, missense mutations, frameshift mutations),
location (i.e. 5th exon, 123rd protein), PolyPhen score, and so on.

To allow for parallelization, we split the list of sub-types into equally
sized tasks that are each tested on a separate cluster array job. The split
is done by taking the modulus of each type's position in the given master list
of types. We repeat this process for multiple splits of the TCGA cohort into
training/testing cohorts, as defined by the given cross-validation ID.

Examples:
    python test_cohort_mutypes.py ../subv_search/setup/mtype_list.p ../output/ \
            BRCA Lasso 3 11
    python test_cohort_mutypes.py /home/experiments/clf_compare/mtype_list.p \
            SKCM rForest 0 8 --task_count=25

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts.tcga import MutationCohort
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
    file_list = glob(os.path.join(out_dir, 'out__cv-*_task-*.p'))
    base_names = [os.path.basename(fl).split('out__')[1] for fl in file_list]

    cv_ids = [int(nm.split('_')[0].split('cv-')[1]) for nm in base_names]
    task_ids = [int(nm.split('_')[1].split('task-')[1].split('.p')[0])
                for nm in base_names]

    return file_list, cv_ids, task_ids


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
    file_list, cv_ids, _ = get_output_files(out_dir)
    out_df = pd.DataFrame(columns=set(cv_ids))

    for use_cv in set(cv_ids):
        out_df[use_cv] = pd.concat(
            pd.Series(pickle.load(open(fl, 'rb'))['Acc'])
            for cv_id, fl in zip(cv_ids, file_list) if cv_id == use_cv
            )

    return out_df


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
    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
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
        '--parallel_jobs', type=int, default=8,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    if args.verbose:
        print("Starting testing for sub-types in\n{}\nwith "
              "cross-validation ID {} and task ID {} ...".format(
                  args.mtype_file, args.cv_id, args.task_id))

    mtype_list = sorted(pickle.load(open(args.mtype_file, 'rb')))
    out_file = os.path.join(args.out_dir,
                            'out__cv-{}_task-{}.p'.format(
                                args.cv_id, args.task_id))

    # loads the pipeline used for classifying variants, gets the mutated
    # genes for each variant under consideration
    mut_clf = eval(args.classif)
    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                             for mtype in mtype_list])

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=list(use_genes),
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', expr_dir=firehose_dir, syn=syn,
        cv_seed=(args.cv_id + 3) * 19, cv_prop=2.0/3
        )

    if args.verbose:
        print("Loaded {} sub-types over {} genes which will be tested using "
              "classifier {} in cohort {} with {} samples.".format(
                    len(mtype_list), len(use_genes), args.classif,
                    args.cohort, len(cdata.samples)
                    ))

    # initialize the dictionaries that will store classification
    # performances and hyper-parameter values
    out_acc = {mtype: -1 for mtype in mtype_list}
    out_par = {mtype: None for mtype in mtype_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:

            if args.verbose:
                print("Testing {} ...".format(mtype))

            # gets the genes that this variant mutates, initializes the
            # classification pipeline
            ex_genes = set(gn for gn, _ in mtype.subtype_list())
            clf = mut_clf()

            # tunes the classifier using the training cohort
            clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                         tune_splits=args.tune_splits,
                         test_count=args.test_count,
                         parallel_jobs=args.parallel_jobs)
            out_par[mtype] = {par: clf.get_params()[par]
                              for par, _ in clf.tune_priors}

            # fits the tuned classifier on the training cohort, evaluates its
            # performance on the testing cohort and saves the results
            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype] = clf.eval_coh(cdata, mtype,
                                          exclude_genes=ex_genes)

        else:
            del(out_acc[mtype])
            del(out_par[mtype])

    # saves the performance measurements and tuned hyper-parameter values
    # for each sub-type to file
    pickle.dump({'Acc': out_acc, 'Par': out_par,
                 'Info': {'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                 open(out_file, 'wb'))


if __name__ == "__main__":
    main()

