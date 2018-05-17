
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType

import argparse
from importlib import import_module
import synapseclient
import dill as pickle

import numpy as np
from functools import reduce
from operator import and_

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'


def load_output(model_name, solve_method, cohort, gene):
    out_lists = [
        pickle.load(open(os.path.join(
            base_dir, "output", model_name, solve_method, cohort, gene,
            "out__cv-{}.p".format(cv_id)
            ), 'rb'))['Infer']
        for cv_id in range(10)
        ]

    return np.concatenate(out_lists, axis=1)


def load_params(model_name, solve_method, cohort, gene):
    out_lists = [
        pickle.load(open(os.path.join(
            base_dir, "output", model_name, solve_method, cohort, gene,
            "out__cv-{}.p".format(cv_id)
            ), 'rb'))['Params']
        for cv_id in range(10)
        ]

    return {par: np.stack([ols[par] for ols in out_lists], axis=0).transpose()
            for par in reduce(and_, [ols.keys() for ols in out_lists])}


def load_vars(model_name, solve_method, cohort, gene):
    out_lists = [
        pickle.load(open(os.path.join(
            base_dir, "output", model_name, solve_method, cohort, gene,
            "out__cv-{}.p".format(cv_id)
            ), 'rb'))['Vars']
        for cv_id in range(10)
        ]

    return {var: np.stack([ols[var] for ols in out_lists], axis=0).transpose()
            for var in reduce(and_, [ols.keys() for ols in out_lists])}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str,
                        help='the name of a Stan model')
    parser.add_argument(
        'solve_method', type=str,
        help='the method used for optimizing the parameters of the Stan model'
        )

    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('gene', type=str, help='a gene with mutated samples')

    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    out_path = os.path.join(base_dir, 'output', args.model_name,
                            args.solve_method, args.cohort, args.gene)

    if args.verbose:
        print("Starting distribution testing for Stan model {} using "
              "optimization method {} on mutated gene {} in TCGA cohort {} "
              "for cross-validation ID {} ...".format(
                  args.model_name, args.solve_method,
                  args.cohort, args.gene, args.cv_id
                ))

    use_mtype = MuType({('Gene', args.gene): None})
    use_module = import_module('HetMan.experiments.stan_test'
                               '.distr.models.{}'.format(args.model_name))
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

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ('/home/exacloud/lustre1/CompBio'
                                '/mgrzad/input-data/synapse')
    syn.login()
    
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', expr_dir=firehose_dir, var_source='mc3',
        syn=syn, cv_prop=1.0, cv_seed=1298 + 93 * args.cv_id
        )

    clf_stan.tune_coh(cdata, use_mtype, exclude_genes={args.gene},
                      tune_splits=4, test_count=24, parallel_jobs=12)
    clf_stan.fit_coh(cdata, use_mtype, exclude_genes={args.gene})

    if clf_stan.tune_priors:
        clf_params = clf_stan.get_params()
    else:
        clf_params = None

    infer_mat = clf_stan.infer_coh(
        cdata, use_mtype, exclude_genes={args.gene},
        infer_splits=12, infer_folds=4, parallel_jobs=12
        )

    pickle.dump(
        {'Params': clf_params, 'Infer': infer_mat,
         'Vars': clf_stan.named_steps['fit'].get_var_means()},
        open(os.path.join(out_path, 'out__cv-{}.p'.format(args.cv_id)), 'wb')
        )


if __name__ == "__main__":
    main()

