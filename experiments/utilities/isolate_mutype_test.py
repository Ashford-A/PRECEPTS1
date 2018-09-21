
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.utilities.data_dirs import (
    expr_dir, copy_dir, annot_file)
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType
from HetMan.experiments.utilities.classifiers import *

import synapseclient
import argparse
from pathlib import Path
import dill as pickle

import pandas as pd
from importlib import import_module
from operator import or_
from functools import reduce
from sklearn.metrics import roc_auc_score, average_precision_score


def get_output_files(out_dir):
    out_path = Path(out_dir)
    out_files = tuple(out_path.glob('out__task-*__cv-*.p'))

    file_info = [fl.name.split('__') for fl in out_files]
    task_ids = [int(x[1].split('-')[1]) for x in file_info]
    cv_ids = [int(x[2].split('-')[1].split('.p')[0]) for x in file_info]

    return out_files, task_ids, cv_ids


def load_acc(out_dir):
    """Gets the cross-validated AUCs of a set of tested MuTypes.

    Args:
        out_dir (str): The directory where the results were saved.

    Examples:
        >>> out_data = load_output("HetMan/experiments/subvariant_detection/"
        >>>                        "output/PAAD/rForest/search")

    """
    file_list, task_ids, cv_ids = get_output_files(out_dir)
    out_data = [pickle.load(fl.open('rb')) for fl in file_list]

    out_auc, out_aupr = tuple(
        pd.concat([
            pd.concat([
                pd.DataFrame.from_dict({
                    mtype: pd.Series(dt)
                    for mtype, dt in out_dict[acc_type].items()
                    }, orient='index')
                for out_dict, task, cv in zip(out_data, task_ids, cv_ids)
                if task == use_task
                ], axis=1)
            for use_task in set(task_ids)
            ], axis=0)
        for acc_type in ['AUC', 'AUPR']
        )

    base_auc, base_aupr = tuple(
        pd.Series(out_df.loc[
            [mtype for mtype in out_df.index if isinstance(mtype, tuple)],
            out_df.columns == 0
            ].values[0])
        for out_df in [out_auc, out_aupr]
        )

    out_auc, out_aupr = tuple(
        out_df.loc[
            [mtype for mtype in out_df.index if isinstance(mtype, MuType)],
            out_df.columns.isin(['Base', 'Iso'])
            ]
        for out_df in [out_auc, out_aupr]
        )

    return base_auc, out_auc, base_aupr, out_aupr


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        "Isolate the expression signature of mutation subtypes from their "
        "parent gene(s)' signature or that of a list of genes in a given "
        "TCGA cohort."
        )

    # positional command line arguments for where input data and output
    # data is to be stored
    parser.add_argument('mtype_file', type=str,
                        help='the pickle file where sub-types are stored')
    parser.add_argument('out_dir', type=str,
                        help='where to save the output of testing sub-types')

    # positional arguments for which cohort of samples and which mutation
    # classifier to use for testing
    parser.add_argument('cohort', type=str, help="a TCGA cohort")
    parser.add_argument('classif', type=str,
                        help="a classifier in HetMan.predict.classifiers")
    parser.add_argument('cv_id', type=int,
                        help="random seed used for cross-validation draws")

    parser.add_argument('--use_genes', type=str, default=None, nargs='+',
                        help='specify which gene(s) to isolate against')

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    # optional arguments controlling how classifier tuning is to be performed
    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many cohort splits to use for tuning'
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
    out_file = os.path.join(args.out_dir,
                            'out__task-{}__cv-{}.p'.format(
                                args.task_id, args.cv_id))

    if args.verbose:
        print("Starting isolation for sub-types in\n{}\nthe results of "
              "which will be stored in\n{}\nwith classifier <{}>.".format(
                  args.mtype_file, args.out_dir, args.classif
                ))

    mtype_list = pickle.load(open(args.mtype_file, 'rb'))
    use_lvls = []

    for lvls in reduce(or_, [{mtype.get_sorted_levels()}
                             for mtype in mtype_list]):
        for lvl in lvls:
            if lvl not in ['Scale', 'Copy'] and lvl not in use_lvls:
                use_lvls.append(lvl)

    if args.use_genes is None:
        if set(mtype.cur_level for mtype in mtype_list) == {'Gene'}:
            use_genes = reduce(or_, [set(gn for gn, _ in mtype.subtype_list())
                                     for mtype in mtype_list])

        else:
            raise ValueError(
                "A gene to isolate against must be given or the subtypes "
                "listed must have <Gene> as their top level!"
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
    # cross-validation ID for this task
    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=list(use_genes), mut_levels=use_lvls,
        expr_source='Firehose', var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=expr_dir, copy_dir=copy_dir,
        syn=syn, cv_seed=args.cv_id, cv_prop=0.75
        )

    if args.verbose:
        print("Loaded {} subtypes of which roughly {} will be isolated in "
              "cohort {} with {} samples.".format(
                  len(mtype_list), len(mtype_list) // args.task_count,
                  args.cohort, len(cdata.samples)
                ))
 
    if 'Gene' in use_lvls:
        base_mtype = MuType({('Gene', tuple(use_genes)): {
            ('Scale', 'Point'): None}})
 
    else:
        base_mtype = MuType({('Scale', 'Point'): None})

    train_samps = base_mtype.get_samples(cdata.train_mut)
    test_samps = base_mtype.get_samples(cdata.test_mut)

    out_auc = {mtype: {'Base': None, 'Iso': None} for mtype in mtype_list}
    out_aupr = {mtype: {'Base': None, 'Iso': None} for mtype in mtype_list}
    out_params = {mtype: {'Base': None, 'Iso': None} for mtype in mtype_list}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            clf = mut_clf()

            if args.verbose:
                print("Isolating {} ...".format(mtype))

            clf.tune_coh(
                cdata, mtype, exclude_genes=use_genes,
                tune_splits=args.tune_splits, test_count=args.test_count,
                parallel_jobs=args.parallel_jobs
                )

            out_params[mtype]['Base'] = {par: clf.get_params()[par]
                                         for par, _ in mut_clf.tune_priors}
            clf.fit_coh(cdata, mtype, exclude_genes=use_genes)

            test_omics, test_pheno = cdata.test_data(
                mtype, exclude_genes=use_genes)
            pred_scores = clf.parse_preds(clf.predict_omic(test_omics))

            if len(set(test_pheno)) == 2:
                out_auc[mtype]['Base'] = roc_auc_score(
                    test_pheno, pred_scores)
                out_aupr[mtype]['Base'] = average_precision_score(
                    test_pheno, pred_scores)

            ex_train = train_samps - mtype.get_samples(cdata.train_mut)
            ex_test = test_samps - mtype.get_samples(cdata.test_mut)

            clf.tune_coh(
                cdata, mtype,
                exclude_genes=use_genes, exclude_samps=ex_train,
                tune_splits=args.tune_splits, test_count=args.test_count,
                parallel_jobs=args.parallel_jobs
                )

            out_params[mtype]['Iso'] = {par: clf.get_params()[par]
                                        for par, _ in mut_clf.tune_priors}
            clf.fit_coh(cdata, mtype,
                        exclude_genes=use_genes, exclude_samps=ex_train)

            test_omics, test_pheno = cdata.test_data(
                mtype, exclude_genes=use_genes, exclude_samps=ex_test)
            pred_scores = clf.parse_preds(clf.predict_omic(test_omics))

            if len(set(test_pheno)) == 2:
                out_auc[mtype]['Iso'] = roc_auc_score(test_pheno, pred_scores)
                out_aupr[mtype]['Iso'] = average_precision_score(
                    test_pheno, pred_scores)

        else:
            del(out_auc[mtype])
            del(out_aupr[mtype])
            del(out_params[mtype])

    pickle.dump(
        {'AUC': out_auc, 'AUPR': out_aupr,
         'Clf': mut_clf, 'Params': out_params,
         'Info': {'TunePriors': mut_clf.tune_priors,
                  'TuneSplits': args.tune_splits,
                  'TestCount': args.test_count}},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

