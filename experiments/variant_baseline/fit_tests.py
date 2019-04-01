
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.setup_tests import get_cohort_data

import argparse
from importlib import import_module
import dill as pickle

import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as aupr_score


def load_variants_list(setup_dir, expr_source, cohort, samp_cutoff):
    var_file = "vars-list_{}__{}__samps-{}.p".format(
        expr_source, cohort, samp_cutoff)

    return pickle.load(open(os.path.join(setup_dir, 'setup', var_file), 'rb'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        choices=list(expr_sources.keys()),
                        help='which TCGA expression data source to use')
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")

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

    parser.add_argument('--setup_dir', type=str, default=base_dir)
    parser.add_argument('--out_dir', type=str, default=base_dir)

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    cdata = get_cohort_data(args.expr_source, args.cohort,
                            cv_prop=0.75, cv_seed=2079 + 57 * args.cv_id)

    # load list of variants whose presence will be predicted
    vars_list = load_variants_list(
        args.setup_dir, args.expr_source, args.cohort, args.samp_cutoff)

    clf_info = args.classif.split('__')
    clf_module = import_module(
        'HetMan.experiments.variant_baseline.models.{}'.format(clf_info[0]))
    mut_clf = getattr(clf_module, clf_info[1].capitalize())

    out_acc = {mtype: {'tune': {'mean': None, 'std': None},
                       'train': {'AUC': None, 'AUPR': None},
                       'test': {'AUC': None, 'AUPR': None}}
               for mtype in vars_list}

    out_params = {mtype: None for mtype in vars_list}
    out_time = {mtype: {'tune': {'fit': dict(), 'score': dict()},
                        'final': {'fit': None, 'score': None}}
                for mtype in vars_list}

    for i, mtype in enumerate(vars_list):
        if (i % args.task_count) == args.task_id:
            clf = mut_clf()

            if args.verbose:
                print("Testing {} ...".format(mtype))

            # get the gene that the variant is associated with and the list
            # of genes on the same chromosome as that gene
            var_gene = mtype.subtype_list()[0][0]
            ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                        if annot['Chr'] == cdata.gene_annot[var_gene]['Chr']}

            clf, cv_output = clf.tune_coh(
                cdata, mtype, exclude_genes=ex_genes,
                tune_splits=4, test_count=36, parallel_jobs=8
                )

            out_time[mtype]['tune']['fit']['avg'] = cv_output['mean_fit_time']
            out_time[mtype]['tune']['fit']['std'] = cv_output['std_fit_time']
            out_time[mtype]['tune']['score']['avg'] = cv_output[
                'mean_score_time']
            out_time[mtype]['tune']['score']['std'] = cv_output[
                'std_score_time']

            out_acc[mtype]['tune']['mean'] = cv_output['mean_test_score']
            out_acc[mtype]['tune']['std'] = cv_output['std_test_score']
            out_params[mtype] = {par: clf.get_params()[par]
                                 for par, _ in mut_clf.tune_priors}

            t_start = time.time()
            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            t_end = time.time()
            out_time[mtype]['final']['fit'] = t_end - t_start

            pheno_list = dict()
            train_omics, pheno_list['train'] = cdata.train_data(
                mtype, exclude_genes=ex_genes)
            test_omics, pheno_list['test'] = cdata.test_data(
                mtype, exclude_genes=ex_genes)

            t_start = time.time()
            pred_scores = {
                'train': clf.parse_preds(clf.predict_omic(train_omics)),
                'test': clf.parse_preds(clf.predict_omic(test_omics))
                }
            out_time[mtype]['final']['score'] = time.time() - t_start

            samp_sizes = {'train': (len(mtype.get_samples(cdata.train_mut))
                                    / len(cdata.train_samps)),
                          'test': (len(mtype.get_samples(cdata.test_mut))
                                   / len(cdata.test_samps))}

            for samp_set, scores in pred_scores.items():
                if len(set(pheno_list[samp_set])) == 2:
                    out_acc[mtype][samp_set]['AUC'] = roc_auc_score(
                        pheno_list[samp_set], scores)
                    out_acc[mtype][samp_set]['AUPR'] = aupr_score(
                        pheno_list[samp_set], scores)
 
                else:
                    out_acc[mtype][samp_set]['AUC'] = 0.5
                    out_acc[mtype][samp_set]['AUPR'] = samp_sizes[samp_set]

        else:
            del(out_acc[mtype])
            del(out_params[mtype])
            del(out_time[mtype])

    pickle.dump(
        {'Acc': out_acc, 'Clf': mut_clf,
         'Params': out_params, 'Time': out_time},
        open(os.path.join(args.out_dir, 'output',
                          'out__cv-{}_task-{}.p'.format(
                              args.cv_id, args.task_id)),
             'wb')
        )


if __name__ == "__main__":
    main()

