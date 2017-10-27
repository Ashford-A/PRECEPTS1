
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
    python test_cohort_mutypes.py ../subv_search/output BRCA Lasso 3 11
    python test_cohort_mutypes.py /home/experiments/clf_compare/output \
            SKCM rForest 0 8

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
from operator import or_
from functools import reduce


firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_output(out_dir):
    """Gets the cross-validated AUCs of a set of tested MuTypes.

    Args:
        out_dir (str): The directory where the results were saved.

    Returns:
        out_data (pd.DataFrame), shape = [n_MuTypes, 5]
            How well the given classifier was able to predict the presence
            of each mutation sub-type in each cross-validation run.

    Examples:
        >>> out_data = test_output("HetMan/experiments/subvariant_detection/"
        >>>                        "output/PAAD/rForest/search")

    """

    # gets the list of output files for each cross-validation run and
    # reads in the data
    out_list = [
        [pickle.load(open(fl, 'rb'))
         for fl in glob(os.path.join(
             out_dir, "results/out__cv-{}_task-*".format(cv_id)))]
        for cv_id in range(5)
        ]

    # consolidates the output into a DataFrame and returns it
    return pd.concat(
        [pd.concat(pd.DataFrame.from_dict(x, orient='index') for x in ols)
         for ols in out_list],
        axis=1
        )


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Test a classifier's ability to predict the presence "
                     "of a list of sub-types.")
        )

    parser.add_argument('mtype_dir', type=str,
                        help='the folder where sub-types are stored')
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')

    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse the command line arguments and show info about where the
    # input sub-types are stored and which subset of them will be tested
    args = parser.parse_args()
    if args.verbose:
        print("Starting testing for directory\n{}\nwith "
              "cross-validation ID:{} and task ID:{} ...".format(
                  args.mtype_dir, args.cv_id, args.task_id))

    # gets the directory where output will be saved and the name of the TCGA
    # cohort under consideration, loads the list of gene sub-variants 
    mtype_list = pickle.load(
        open(os.path.join(args.mtype_dir, 'tmp', 'mtype_list.p'), 'rb'))

    # loads the pipeline used for classifying variants, gets the mutated
    # genes for each variant under consideration
    mut_clf = eval(args.classif)
    use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_iter())
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
        cv_seed=(args.cv_id + 3) * 19, syn=syn
        )

    if args.verbose:
        print("Loaded {} sub-types over {} genes in cohort {} with {} "
              "samples which will be tested using classifier {}.".format(
                    len(mtype_list), len(use_genes), args.cohort,
                    len(cdata.samples), args.classif
                    ))

    # intialize the dictionary that will store classification performances
    out_acc = {mtype: -1 for mtype in mtype_list}

    # for each sub-variants, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if i % 24 == args.task_id:

            if args.verbose:
                print("Testing {} ...".format(mtype))

            # gets the genes that this variant mutates, initializes the
            # classification pipeline
            ex_genes = set(gn for gn, _ in mtype.subtype_iter())
            clf = mut_clf()

            # tunes the classifier using the training cohort
            clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                         tune_splits=8, test_count=24, parallel_jobs=12)

            # fits the tuned classifier on the training cohort, evaluates its
            # performance on the testing cohort and saves the results
            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype] = clf.eval_coh(cdata, mtype,
                                          exclude_genes=ex_genes)

        else:
            del(out_acc[mtype])

    # saves the performance measurements for each variant to file
    out_file = os.path.join(
        args.mtype_dir, 'results',
        'out__cv-{}_task-{}.p'.format(args.cv_id, args.task_id)
        )
    pickle.dump(out_acc, open(out_file, 'wb'))


if __name__ == "__main__":
    main()

