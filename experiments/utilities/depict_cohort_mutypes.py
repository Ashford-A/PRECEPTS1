
"""Finding the expression signature of a list of variant types in a cohort.

This script takes a list of variant sub-types and find the pattern of
expression perturbation that a given classification pipeline associates with
their presence in a cohort of samples. A sub-type can be any subset of the
mutations present in a gene or a group of genes, as defined by shared
properties. These properties can include form (i.e. splice site mutations,
missense mutations, frameshift mutations), location (i.e. 5th exon, 123rd
protein), PolyPhen score, and so on.

To allow for parallelization, we split the list of sub-types into equally
sized tasks that are each tested on a separate cluster array job. The split
is done by taking the modulus of each type's position in the given master list
of types. We repeat this process for multiple splits of the TCGA cohort into
training/testing cohorts, as defined by the given cross-validation ID.

Args:

Examples:

"""

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
from operator import or_, and_, add
from functools import reduce

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_output(out_dir):

    out_list = [
        [pickle.load(
            open(os.path.join(out_dir, "results/out__cv-{}_task-{}.p".format(
                cv_id, task_id)
                ), 'rb')
            ) for task_id in range(3)]
        for cv_id in range(10)
        ]

    acc_data = pd.concat(
        [pd.concat(pd.DataFrame.from_dict(x['Acc'], orient='index')
                   for x in ols)
         for ols in out_list],
        axis=1
        )

    coef_data = []
    for task_id in range(3):
        coef_list = [
            pd.DataFrame.from_dict(out_list[cv_id][task_id]['Coef'],
                                   orient='index').fillna(0.0)
            for cv_id in range(10)
            ]

        cmn_gns = reduce(or_, [x.columns for x in coef_list])
        cmn_types = reduce(and_, [x.index for x in coef_list])

        coef_mat = reduce(add, [ls.loc[cmn_types, cmn_gns].fillna(0.0)
                                for ls in coef_list])

        coef_data += [coef_mat / 10.0]

    coef_data = pd.concat(coef_data, axis=1).fillna(0.0)

    return acc_data, coef_data


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Find the signatures a classifier predicts for a list "
                     "of sub-types.")
        )

    parser.add_argument('mtype_dir', type=str,
                        help="the folder where sub-types are stored")
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')

    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument(
        '--tune_splits', type=int, default=8,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=32,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=16,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    if args.verbose:
        print("Starting signature portrayal for directory\n{}\nwith "
              "cross-validation ID {} and task ID {} ...".format(
                  args.mtype_dir, args.cv_id, args.task_id))

    # gets the directory where output will be saved and the name of the TCGA
    # cohort under consideration, loads the list of gene sub-variants 
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
        cohort=args.cohort, mut_genes=list(use_genes),
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', data_dir=firehose_dir,
        cv_seed=(args.cv_id + 5) * 31, cv_prop=0.9, syn=syn
        )

    if args.verbose:
        print("Loaded {} sub-types over {} genes whose expression signatures "
              "will be portrayed with respect to classifier {} in cohort {} "
              "with {} samples.".format(
                  len(mtype_list), len(use_genes), args.classif,
                  args.cohort, len(cdata.samples)
                  ))

    out_acc = {mtype: -1 for mtype in mtype_list}
    out_coef = {mtype: {} for mtype in mtype_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if i % 3 == args.task_id:

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

            # fits the tuned classifier on the training cohort, evaluates its
            # performance on the testing cohort and saves the results
            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype] = clf.eval_coh(cdata, mtype,
                                          exclude_genes=ex_genes)

            out_coef[mtype] = {
                gn: coef for gn, coef in clf.get_coef().items()
                if coef != 0
                }

        else:
            del(out_acc[mtype])
            del(out_coef[mtype])

    # saves the performance measurements for each variant to file
    out_file = os.path.join(
        args.mtype_dir, 'results',
        'out__cv-{}_task-{}.p'.format(args.cv_id, args.task_id)
        )
    pickle.dump({'Acc': out_acc, 'Coef': out_coef,
                 'Info': {'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count,
                          'ParallelJobs': args.parallel_jobs}},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main()

