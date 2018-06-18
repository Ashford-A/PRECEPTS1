
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
import pandas as pd

import argparse
import synapseclient
from importlib import import_module
import dill as pickle

import time
from sklearn.metrics import roc_auc_score, average_precision_score
from operator import itemgetter


def load_output(expr_source, cohort, samp_cutoff, classif):
    out_dir = os.path.join(
        base_dir, "output", expr_source,
        "{}__samps-{}".format(cohort, samp_cutoff), classif
        )

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

    acc_df = out_df.loc[:, ['AUC', 'AUPR']]
    par_list = out_df.Params.unstack()
    par_list.index = par_list.index.droplevel(0)
    par_df = pd.DataFrame.from_dict(dict(par_list.iteritems())).stack()

    return acc_df, out_df.Time, par_df, tuple(use_clf)[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        choices=['Firehose', 'toil', 'toil_tx'],
                        help='which TCGA expression data source to use')
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")

    parser.add_argument(
        'syn_root', type=str,
        help="the root cache directory for data downloaded from Synapse"
        )

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
        base_dir, 'output', args.expr_source,
        '{}__samps-{}'.format(args.cohort, args.samp_cutoff), args.classif
        )

    gene_list = pickle.load(
        open(os.path.join(base_dir, "setup",
                          "genes-list_{}__{}__samps-{}.p".format(
                              args.expr_source, args.cohort,
                              args.samp_cutoff
                            )),
             'rb')
        )

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = args.syn_root
    syn.login()
 
    expr_dir = pd.read_csv(
        open(os.path.join(base_dir, 'expr_sources.txt'), 'r'),
        sep='\t', header=None, index_col=0
        ).loc[args.expr_source].iloc[0]

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=gene_list, mut_levels=['Gene'],
        expr_source=args.expr_source, expr_dir=expr_dir, var_source='mc3',
        syn=syn, cv_prop=0.75, cv_seed=2079 + 57 * args.cv_id
        )

    clf_info = args.classif.split('__')
    clf_module = import_module(
        'HetMan.experiments.gene_baseline.models.{}'.format(clf_info[0]))
    mut_clf = getattr(clf_module, clf_info[1].capitalize())

    out_auc = {mut_gene: None for mut_gene in gene_list}
    out_aupr = {mut_gene: None for mut_gene in gene_list}
    out_params = {mut_gene: None for mut_gene in gene_list}
    out_time = {mut_gene: None for mut_gene in gene_list}

    for i, mut_gene in enumerate(gene_list):
        if (i % args.task_count) == args.task_id:
            if args.verbose:
                print("Testing {} ...".format(mut_gene))

            clf = mut_clf()
            mtype = MuType({('Gene', mut_gene): None})

            clf.tune_coh(cdata, mtype, exclude_genes={mut_gene},
                         tune_splits=4, test_count=24, parallel_jobs=12)
            out_params[mut_gene] = {par: clf.get_params()[par]
                                    for par, _ in mut_clf.tune_priors}

            t_start = time.time()
            clf.fit_coh(cdata, mtype, exclude_genes={mut_gene})
            t_end = time.time()
            out_time[mut_gene] = t_end - t_start

            test_omics, test_pheno = cdata.test_data(
                mtype, exclude_genes={mut_gene})
            pred_scores = clf.predict_omic(test_omics)

            out_auc[mut_gene] = roc_auc_score(test_pheno, pred_scores)
            out_aupr[mut_gene] = average_precision_score(
                test_pheno, pred_scores)

        else:
            del(out_auc[mut_gene])
            del(out_aupr[mut_gene])
            del(out_params[mut_gene])
            del(out_time[mut_gene])

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

