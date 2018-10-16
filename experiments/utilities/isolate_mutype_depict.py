
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

from HetMan.experiments.utilities.data_dirs import *
from HetMan.features.cohorts.tcga import MutationCohort
from dryadic.features.mutations import MuType
from HetMan.experiments.utilities.classifiers import *

import argparse
import synapseclient
from glob import glob
import dill as pickle

import pandas as pd
from functools import reduce
from operator import or_, and_, add
from sklearn.metrics import roc_auc_score, average_precision_score


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

    coef_data = pd.concat(coef_data, axis=0).fillna(0.0)

    return acc_data, coef_data


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Find the signatures a classifier predicts for a list "
                     "of sub-types.")
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
    parser.add_argument('cv_id', type=int,
                        help='the random seed to use for cross-validation draws')

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
                            'out__task-{}_cv-{}.p'.format(
                                args.task_id, args.cv_id))

    if args.verbose:
        print("Starting depiction for sub-types in\n{}\nthe results of "
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

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=list(use_genes), mut_levels=use_lvls,
        expr_source='Firehose', var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=expr_dir, copy_dir=copy_dir,
        syn=syn, cv_seed=((args.cv_id + 1) * 203) ** 2, cv_prop=0.8
        )

    if args.verbose:
        print("Loaded {} sub-types over {} genes whose expression signatures "
              "will be portrayed with respect to classifier {} in cohort {} "
              "with {} samples.".format(
                  len(mtype_list), len(use_genes), args.classif,
                  args.cohort, len(cdata.samples)
                  ))

    clf = eval(args.classif)
    out_coef = {mtype: {'Base': dict(), 'Iso': dict()}
                for mtype in mtype_list}

    out_auc = {mtype: {'Base': -1, 'Iso': -1} for mtype in mtype_list}
    out_aupr = {mtype: {'Base': -1, 'Iso': -1} for mtype in mtype_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            mut_clf = clf()

            if args.verbose:
                print("Depicting {} ...".format(mtype))

            if mtype.cur_level == 'Gene':
                cur_genes = set(gn for gn, _ in mtype.subtype_list())

                cur_chrs = {cdata.gene_annot[gene]['chr']
                            for gene in cur_genes}
                ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                            if annot['chr'] in cur_chrs}

            mut_clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                             tune_splits=args.tune_splits,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

            mut_clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_coef[mtype]['Base'] = {
                gn: cf for gn, cf in mut_clf.get_coef().items() if cf != 0}

            test_omics, test_pheno = cdata.test_data(mtype,
                                                     exclude_genes=ex_genes)
            pred_scores = mut_clf.parse_preds(
                mut_clf.predict_omic(test_omics))

            if len(set(test_pheno)) == 2:
                out_auc[mtype]['Base'] = roc_auc_score(
                    test_pheno, pred_scores)
                out_aupr[mtype]['Base'] = average_precision_score(
                    test_pheno, pred_scores)

            if args.use_genes is None:
                base_mtype = MuType({('Gene', tuple(cur_genes)): None})
                ex_train = base_mtype.get_samples(cdata.train_mut)
                ex_test = base_mtype.get_samples(cdata.test_mut)

                ex_train -= mtype.get_samples(cdata.train_mut)
                ex_test -= mtype.get_samples(cdata.test_mut)

            mut_clf.tune_coh(cdata, mtype,
                             exclude_genes=ex_genes, exclude_samps=ex_train,
                             tune_splits=args.tune_splits,
                             test_count=args.test_count,
                             parallel_jobs=args.parallel_jobs)

            mut_clf.fit_coh(cdata, mtype,
                            exclude_genes=ex_genes, exclude_samps=ex_train)
            out_coef[mtype]['Iso'] = {
                gn: cf for gn, cf in mut_clf.get_coef().items() if cf != 0}

            test_omics, test_pheno = cdata.test_data(
                mtype, exclude_genes=ex_genes, exclude_samps=ex_test)
            pred_scores = mut_clf.parse_preds(mut_clf.predict_omic(test_omics))

            if len(set(test_pheno)) == 2:
                out_auc[mtype]['Iso'] = roc_auc_score(test_pheno, pred_scores)
                out_aupr[mtype]['Iso'] = average_precision_score(
                    test_pheno, pred_scores)

        else:
            del(out_coef[mtype])
            del(out_auc[mtype])
            del(out_aupr[mtype])

    pickle.dump({'Acc': out_auc, 'AUPR': out_aupr, 'Coef': out_coef,
                 'Info': {'Clf': mut_clf, 'TunePriors': clf.tune_priors,
                          'TuneSplits': args.tune_splits,
                          'TestCount': args.test_count}},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main()

