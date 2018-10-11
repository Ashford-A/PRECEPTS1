
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import TransferMutationCohort
from HetMan.experiments.transfer_baseline import *
from HetMan.experiments.transfer_baseline.setup_tests import get_cohort_data
from dryadic.features.mutations import MuType

import argparse
from importlib import import_module
import synapseclient
import pandas as pd
import dill as pickle

import time
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as aupr_score
from operator import itemgetter


def load_output(expr_source, samp_cutoff, classif):
    out_dir = os.path.join(base_dir, "output",
                           "{}__samps-{}".format(expr_source, samp_cutoff),
                           classif)

    out_files = [(fl, int(fl.split('out__cv-')[1].split('_task-')[0]),
                  int(fl.split('_task-')[1].split('.p')[0]))
                  for fl in os.listdir(out_dir) if 'out__cv-' in fl]
    out_files = sorted(out_files, key=itemgetter(2, 1))
    
    out_df = pd.concat([
        pd.concat([
            pd.DataFrame.from_dict(pickle.load(open(os.path.join(out_dir, fl),
                                                    'rb')))
            for fl, _, task in out_files if task == task_id
            ], axis=1)
        for task_id in set([fl[2] for fl in out_files])
        ], axis=0)

    use_clf = set(out_df.Clf.values.ravel())
    if len(use_clf) != 1:
        raise ValueError("Each gene baseline testing experiment must be run "
                         "with exactly one classifier!")

    par_df = pd.concat(dict(
        out_df.Params.apply(
            lambda gn_pars: pd.DataFrame.from_records(tuple(gn_pars)), axis=1)
        ))

    par_df['CV'] = par_df.index.get_level_values(2)
    par_df.index = pd.MultiIndex.from_arrays(
        [par_df.index.get_level_values(0), par_df.index.get_level_values(1)])

    return out_df.AUC, out_df.AUPR, out_df.Time, par_df, tuple(use_clf)[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        choices=['Firehose', 'toil', 'toil_tx'],
                        help='which TCGA expression data source to use')

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('classif', type=str,
                        help='the name of a mutation classifier')
    
    parser.add_argument(
        '--cv_id', type=int, default=6732,
        help='the random seed to use for cross-validation draws'
        )
 
    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    # parse command-line arguments, create directory where to save results
    args = parser.parse_args()
    out_path = os.path.join(
        base_dir, 'output',
        '{}__samps-{}'.format(args.expr_source, args.samp_cutoff),
        args.classif
        )

    comb_list = pickle.load(
        open(os.path.join(base_dir, "setup",
                          "combs-list_{}__samps-{}.p".format(
                              args.expr_source, args.samp_cutoff)),
             'rb')
        )

    comb_list = sorted((cohs, mtype) for cohs, mtypes in comb_list.items()
                       for mtype in mtypes)
    task_size = len(comb_list) // args.task_count

    combs_use = comb_list[
        (args.task_id * task_size):((args.task_id + 1) * task_size)]
    if args.task_id < (len(comb_list) % args.task_count):
        combs_use += [comb_list[-(args.task_id + 1)]]

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    clf_info = args.classif.split('__')
    clf_module = import_module(
        'HetMan.experiments.transfer_baseline.models.{}'.format(clf_info[0]))
    mut_clf = getattr(clf_module, clf_info[1].capitalize())

    out_auc = {comb: {'train': dict(), 'test': dict()} for comb in combs_use}
    out_aupr = {comb: {'train': dict(), 'test': dict()} for comb in combs_use}
    out_params = {comb: None for comb in combs_use}
    out_time = {comb: None for comb in combs_use}

    for cur_cohs in {cohs for cohs, _ in combs_use}:
        if args.verbose:
            print("Transferring between cohort {} "
                  "and cohort {} ...".format(*cur_cohs))

        combs_cur = [(cohs, mtypes) for cohs, mtypes in combs_use
                     if cohs == cur_cohs]
        cur_genes = {mtype.subtype_list()[0][0] for _, mtype in combs_cur}

        cdata = TransferMutationCohort(
            cohorts=cur_cohs, mut_genes=list(cur_genes),
            mut_levels=['Gene', 'Form_base', 'Protein'],
            expr_sources=args.expr_source, var_sources='mc3',
            copy_sources='Firehose', annot_file=annot_file,
            expr_dir=expr_sources[args.expr_source], copy_dir=copy_dir,
            syn=syn, cv_prop=0.75, cv_seed=2079 + 57 * args.cv_id
            )

        for _, mtype in combs_cur:
            clf = mut_clf()
            if args.verbose:
                print("Testing {} ...".format(mtype))

            mut_gene = mtype.subtype_list()[0][0]
            ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                        if annot['chr'] == cdata.gene_annot[mut_gene]['chr']}

            clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                         tune_splits=4, test_count=36, parallel_jobs=12)
            out_params[cur_cohs, mtype] = {par: clf.get_params()[par]
                                           for par, _ in mut_clf.tune_priors}

            t_start = time.time()
            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            t_end = time.time()
            out_time[cur_cohs, mtype] = t_end - t_start

            pheno_list = dict()
            train_omics, pheno_list['train'] = cdata.train_data(
                mtype, exclude_genes=ex_genes)
            test_omics, pheno_list['test'] = cdata.test_data(
                mtype, exclude_genes=ex_genes)

            pred_scores = {'train': clf.predict_omic(train_omics),
                           'test': clf.predict_omic(test_omics)}
            samp_sizes = {
                'train': {coh: (len(mtype.get_samples(cdata.train_mut[coh]))
                                / len(cdata.train_samps[coh]))
                          for coh in cur_cohs},
                'test': {coh: (len(mtype.get_samples(cdata.test_mut[coh]))
                               / len(cdata.test_samps[coh]))
                         for coh in cur_cohs}
                }

            for samp_set, scores in pred_scores.items():
                for coh in cur_cohs:
                    if len(set(pheno_list[samp_set][coh])) == 2:
                        out_auc[cur_cohs, mtype][samp_set][coh] = auc_score(
                            pheno_list[samp_set][coh], scores[coh])
                        out_aupr[cur_cohs, mtype][samp_set][coh] = aupr_score(
                            pheno_list[samp_set][coh], scores[coh])
 
                    else:
                        out_auc[cur_cohs, mtype][samp_set][coh] = 0.5
                        out_aupr[cur_cohs, mtype][samp_set][coh] = (
                            samp_sizes[samp_set][coh])

    pickle.dump(
        {'AUC': out_auc, 'AUPR': out_aupr,
         'Clf': mut_clf, 'Params': out_params, 'Time': out_time},
        open(os.path.join(out_path,
                          'out__cv-{}_task-{}.p'.format(
                              args.cv_id, args.task_id)),
             'wb')
        )


if __name__ == "__main__":
    main()

