
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
import synapseclient
from importlib import import_module
import dill as pickle

from functools import reduce
from operator import or_

firehose_dir = "/home/exacloud/lustre1/share_your_data_here/precepts/firehose"


def main():
    parser = argparse.ArgumentParser(
        description=("Test a Stan multitask model's inferred mutation scores "
                     "for a pair of mutation sub-types it is trained to "
                     "classify and compare them to the scores it infers for "
                     "the remaining mutated and wild-type samples for a "
                     "given gene in a TCGA cohort.")
        )

    parser.add_argument('mtype_file', type=str,
                        help='the pickle file where sub-types are stored')
    parser.add_argument('out_dir', type=str,
                        help='where to save the output of testing sub-types')

    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('model_name', type=str, help='a TCGA cohort')
    parser.add_argument('solve_method', type=str, help='a TCGA cohort')

    parser.add_argument('--use_genes', type=str, default=None, nargs='+',
                        help='specify which gene(s) to isolate against')
    
    parser.add_argument(
        '--cv_id', type=int, default=8807,
        help='the random seed to use for cross-validation draws'
        )

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

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
        print("Starting multi-task isolation with Stan model <{}> for "
              "sub-types in\n{}\nthe results of which will be stored "
              "in\n{}\n".format(args.model_name, args.mtype_file, args.out_dir)
                )

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

    if args.verbose:
        print('Using the following Stan model:\n\n{}'.format(
            clf_stan.named_steps['fit'].model_code))

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/CompBio'
                                '/mgrzad/input-data/synapse')
    syn.login()
    
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=list(use_genes), mut_levels=use_lvls,
        expr_source='Firehose', expr_dir=firehose_dir,
        syn=syn, cv_seed=9099, cv_prop=1.0
        )
    
    if args.verbose:
        print("Loaded {} pairs of subtypes of which roughly {} will be "
              "isolated in cohort {} with {} samples.".format(
                  len(pair_list), len(pair_list) // args.task_count,
                  args.cohort, len(cdata.samples)
                ))

    out_multi = {mtypes: None for mtypes in pair_list}
    out_par = {mtypes: None for mtypes in pair_list}
    out_vars = {mtypes: None for mtypes in pair_list}

    base_mtype = MuType({('Gene', tuple(use_genes)): None})
    base_samps = base_mtype.get_samples(cdata.train_mut)

    # for each sub-variant, check if it has been assigned to this task
    for i, (mtype1, mtype2) in enumerate(pair_list):
        if (i % args.task_count) == args.task_id:
            if args.verbose:
                print("Isolating {} and {} ...".format(mtype1, mtype2))

            ex_samps = base_samps - (mtype1.get_samples(cdata.train_mut)
                                     | mtype2.get_samples(cdata.train_mut))

            clf_stan.tune_coh(
                cdata, [mtype1, mtype2],
                exclude_genes=use_genes, exclude_samps=ex_samps,
                tune_splits=args.tune_splits, test_count=args.test_count,
                parallel_jobs=args.parallel_jobs
                )
            
            clf_stan.fit_coh(cdata, [mtype1, mtype2],
                             exclude_genes=use_genes, exclude_samps=ex_samps)
            clf_params = clf_stan.get_params()

            out_par[(mtype1, mtype2)] = {par: clf_params[par]
                                         for par, _ in clf_stan.tune_priors}
            out_vars[(mtype1, mtype2)] = (
                clf_stan.named_steps['fit'].get_var_means())

            out_multi[(mtype1, mtype2)] = clf_stan.infer_coh(
                cdata, [mtype1, mtype2],
                exclude_genes=use_genes, force_test_samps=ex_samps,
                infer_splits=args.infer_splits, infer_folds=args.infer_folds,
                parallel_jobs=args.parallel_jobs
                )

        else:
            del(out_multi[(mtype1, mtype2)])
            del(out_par[(mtype1, mtype2)])
            del(out_vars[(mtype1, mtype2)])

    pickle.dump(
        {'Infer': out_multi, 'Par': out_par, 'Vars': out_vars,
         'Info': {'TunePriors': clf_stan.tune_priors,
                  'TuneSplits': args.tune_splits,
                  'TestCount': args.test_count}},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

