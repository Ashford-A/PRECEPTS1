
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.predict.pipelines import MutPipe, MultiPipe

import numpy as np
import pandas as pd

import argparse
import synapseclient
from importlib import import_module
import dill as pickle

from glob import glob
from functools import reduce
from operator import or_

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
        pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Infer'],
                               orient='index')
        for fl in file_list
        ])


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
        description=("Test a Stan multitask model's inferred mutation scores "
                     "for a pair of mutation sub-types it is trained to "
                     "classify and compare them to the scores it infers for "
                     "the remaining mutated and wild-type samples for a "
                     "given gene in a TCGA cohort.")
        )

    # positional command line arguments for where input data and output
    # data is to be stored
    parser.add_argument('mtype_file', type=str,
                        help='the pickle file where sub-types are stored')
    parser.add_argument('out_dir', type=str,
                        help='where to save the output of testing sub-types')

    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('mut_levels', type=str,
                        help='a set of mutation levels to consider')

    parser.add_argument('model_name', type=str, help='a TCGA cohort')
    parser.add_argument('solve_method', type=str, help='a TCGA cohort')

    # positional arguments controlling CV and task selection
    parser.add_argument('task_id', type=int,
                        help='the subset of sub-types to assign to this task')

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )

    parser.add_argument('--use_gene', type=str, default=None,
                        help='specify which gene the mutations belong to')

    # optional arguments controlling how classifier tuning is to be performed
    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=12,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=12,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    if args.verbose:
        print("Starting testing for sub-types in\n{}\nwith "
              "task ID {} ...".format(
                  args.mtype_file, args.task_id))

    use_module = import_module('HetMan.experiments.subvariant_multi'
                               '.models.{}'.format(args.model_name))
    UsePipe = getattr(use_module, 'UsePipe')

    if args.solve_method == 'optim':
        clf_stan = getattr(use_module, 'UsePipe')(
            getattr(use_module, 'UseOptimizing')(
                model_code=getattr(use_module, 'use_model'))
            )

    elif args.solve_method == 'variat':
        clf_stan = getattr(use_module, 'UsePipe')(
            getattr(use_module, 'UseVariational')(
                model_code=getattr(use_module, 'use_model'))
            )

    elif args.solve_method == 'sampl':
        clf_stan = getattr(use_module, 'UsePipe')(
            getattr(use_module, 'UseSampling')(
                model_code=getattr(use_module, 'use_model'))
            )

    else:
        raise ValueError("Unrecognized <solve_method> argument!")

    pair_list = pickle.load(open(args.mtype_file, 'rb'))
    out_file = os.path.join(args.out_dir, '{}__out_task-{}.p'.format(
        clf_stan.named_steps['fit'].model_name, args.task_id))

    if args.verbose:
        print('Using the following Stan model:\n\n{}'.format(
            clf_stan.named_steps['fit'].model_code))

    if args.use_gene is None:
        base_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                                  for mtype in pair_list])

    else:
        base_genes = {args.use_gene}

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/CompBio'
                                '/mgrzad/input-data/synapse')
    syn.login()
    
    cdata = MutationCohort(cohort=args.cohort, mut_genes=list(base_genes),
                           mut_levels=args.mut_levels.split('__'),
                           expr_source='Firehose', expr_dir=firehose_dir,
                           syn=syn, cv_seed=9099, cv_prop=1.0)

    base_mtype = MuType({('Gene', tuple(base_genes)): None})
    if "Gene" in args.mut_levels:
        base_samps = base_mtype.get_samples(cdata.train_mut)
    else:
        base_samps = cdata.train_mut.get_samples()

    if args.verbose:
        print("Loaded {} sub-type pairs over {} genes which will be tested using "
              "classifier <Stan> in cohort {} with {} samples.".format(
                    len(pair_list), len(base_genes),
                    args.cohort, len(cdata.samples)
                    ))

    out_infer = {mtypes: None for mtypes in pair_list}
    out_par = {mtypes: None for mtypes in pair_list}
    out_vars = {mtypes: None for mtypes in pair_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, (mtype1, mtype2) in enumerate(pair_list):
        if (i % args.task_count) == args.task_id:
            if args.verbose:
                print("Testing {} x {} ...".format(mtype1, mtype2))

            ex_samps = base_samps - (mtype1.get_samples(cdata.train_mut)
                                     | mtype2.get_samples(cdata.train_mut))
            print(base_mtype)
            print(base_genes)
            print(len(base_samps))
            print(len(ex_samps))

            clf_stan.tune_coh(
                cdata, [mtype1, mtype2],
                exclude_genes=base_genes, exclude_samps=ex_samps,
                tune_splits=args.tune_splits, test_count=args.test_count,
                parallel_jobs=args.parallel_jobs
                )
            
            clf_stan.fit_coh(cdata, [mtype1, mtype2],
                             exclude_genes=base_genes, exclude_samps=ex_samps)
            clf_params = clf_stan.get_params()

            out_par[(mtype1, mtype2)] = {par: clf_params[par]
                                         for par, _ in clf_stan.tune_priors}
            out_vars[(mtype1, mtype2)] = (
                clf_stan.named_steps['fit'].get_var_means())

            out_infer[(mtype1, mtype2)] = clf_stan.infer_coh(
                cdata, [mtype1, mtype2], exclude_genes=base_genes,
                force_test_samps=ex_samps, infer_splits=60, infer_folds=4,
                parallel_jobs=args.parallel_jobs
                )

        else:
            del(out_infer[(mtype1, mtype2)])
            del(out_par[(mtype1, mtype2)])
            del(out_vars[(mtype1, mtype2)])

    pickle.dump(
        {'Infer': out_infer, 'Par': out_par, 'Vars': out_vars},
        open(os.path.join(args.out_dir,
                          'out__cv-{}.p'.format(args.task_id)), 'wb')
        )


if __name__ == "__main__":
    main()

