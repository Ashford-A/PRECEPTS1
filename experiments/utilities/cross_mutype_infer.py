
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.predict.basic.classifiers import *

import synapseclient
import argparse
from glob import glob
import dill as pickle

import pandas as pd
from importlib import import_module
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

    Examples:
        >>> out_data = test_output("HetMan/experiments/subvariant_detection/"
        >>>                        "output/PAAD/rForest/search")

    """
    file_list, task_ids = get_output_files(out_dir)

    out_df = pd.concat([
        pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Cross'],
                               orient='index')
        for fl in file_list
        ]).applymap(lambda x: [y[0] for y in x])

    out_df.index = pd.MultiIndex.from_tuples(out_df.index)
    out_df = out_df.sort_index()

    return out_df


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        "Isolate the expression signatures of pairs of mutation subtypes "
        "against one another from their parent gene(s)' signature or that of "
        "a list of genes in a given TCGA cohort."
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

    parser.add_argument('--use_genes', type=str, default=None, nargs='+',
                        help='specify which gene(s) to isolate against')

    parser.add_argument(
        '--cv_id', type=int, default=4309,
        help='the random seed to use for cross-validation draws'
        )

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

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
        '--infer_splits', type=int, default=20,
        help='how many cohort splits to use for inference bootstrapping'
        )
    parser.add_argument(
        '--infer_folds', type=int, default=4,
        help=('how many parts to split the cohort into in each inference '
              'cross-validation run')
        )

    parser.add_argument(
        '--parallel_jobs', type=int, default=4,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    out_file = os.path.join(args.out_dir,
                            'out__task-{}.p'.format(args.task_id))

    if args.verbose:
        print("Starting cross-isolation for sub-types in\n{}\nthe results "
              "of which will be stored in\n{}\nin cohort {} with "
              "classifier <{}>.".format(args.mtype_file, args.out_dir,
                                        args.cohort, args.classif))

    pair_list = pickle.load(open(args.mtype_file, 'rb'))
    or_list = [mtype1 | mtype2 for mtype1, mtype2 in pair_list]
    use_lvls = []

    for lvls in reduce(or_, [{mtype.get_sorted_levels()}
                             for mtype in or_list]):
        for lvl in lvls:
            if lvl not in use_lvls:
                use_lvls.append(lvl)

    if args.use_genes is None:
        if set(mtype.cur_level for mtype in or_list) == {'Gene'}:
            use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                                     for mtype in or_list])

        else:
            raise ValueError(
                "A gene to isolate against must be given or the pairs of "
                "subtypes listed must each have <Gene> as their top level!"
                )

    else:
        use_genes = set(args.use_genes)
 
    if args.verbose:
        print("Subtypes at mutation annotation levels {} will be isolated "
              "against genes:\n{}".format(use_lvls, use_genes))

    if args.classif[:6] == 'Stan__':
        use_module = import_module('HetMan.experiments.utilities'
                                   '.stan_models.{}'.format(
                                       args.classif.split('Stan__')[1]))
        mut_clf = getattr(use_module, 'UsePipe')
    
    else:
        mut_clf = eval(args.classif)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=list(use_genes), mut_levels=use_lvls,
        expr_source='Firehose', expr_dir=firehose_dir,
        syn=syn, cv_seed=9999, cv_prop=1.0
        )

    if args.verbose:
        print("Loaded {} pairs of subtypes of which roughly {} will be "
              "isolated in cohort {} with {} samples.".format(
                  len(pair_list), len(pair_list) // args.task_count,
                  args.cohort, len(cdata.samples)
                ))

    out_cross = {(mtype1, mtype2): None for mtype1, mtype2 in pair_list}
    out_cross.update({(mtype2, mtype1): None for mtype1, mtype2 in pair_list})
    base_mtype = MuType({('Gene', tuple(use_genes)): None})
    base_samps = base_mtype.get_samples(cdata.train_mut)

    # for each subtype, check if it has been assigned to this task
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
                    
                    out_cross[(mtype1, mtype2)] = clf.infer_coh(
                        cdata, mtype1, exclude_genes=use_genes,
                        force_test_samps=ex_samps,
                        infer_splits=args.infer_splits,
                        infer_folds=args.infer_folds,
                        parallel_jobs=args.parallel_jobs
                        )

                if 10 <= len(samps2 - samps1):
                    clf.tune_coh(cdata, mtype2, exclude_genes=use_genes,
                                 exclude_samps=ex_samps,
                                 tune_splits=args.tune_splits,
                                 test_count=args.test_count,
                                 parallel_jobs=args.parallel_jobs)
                    
                    out_cross[(mtype2, mtype1)] = clf.infer_coh(
                        cdata, mtype2, exclude_genes=use_genes,
                        force_test_samps=ex_samps,
                        infer_splits=args.infer_splits,
                        infer_folds=args.infer_folds,
                        parallel_jobs=args.parallel_jobs
                        )

        else:
            del(out_cross[(mtype1, mtype2)])
            del(out_cross[(mtype2, mtype1)])

    pickle.dump(
        {'Cross': out_cross,
         'Info': {'TunePriors': mut_clf.tune_priors,
                  'TuneSplits': args.tune_splits,
                  'TestCount': args.test_count,
                  'InferFolds': args.infer_folds}},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

