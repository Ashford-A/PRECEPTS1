
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.stan_baseline import *
import argparse
import dill as pickle
from importlib import import_module

import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as aupr_score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('use_gene', type=str,
                        help="the gene whose mutations are being classified")
    parser.add_argument('model', type=str,
                        help="the name of a mutation classifier")

    parser.add_argument('--use_dir', type=str, default=base_dir)
    parser.add_argument('--task_id', type=int, default=0,
                        help="the subset of subtypes to assign to this task")
    parser.add_argument('--cv_id', type=int, default=6072,
                        help="the seed to use for random sampling")

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')

    with open(os.path.join(setup_dir, "vars-list.p"), 'rb') as fl:
        vars_list = pickle.load(fl)

    with open(os.path.join(setup_dir, "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)
    cdata.update_seed(2079 + 57 * args.cv_id, test_prop=0.25)

    clf_info = args.model.split('__')
    clf_module = import_module(
        'HetMan.experiments.stan_baseline.models.{}'.format(clf_info[0]))
    use_clf = getattr(clf_module, clf_info[1].capitalize())

    out_acc = {fit_method: {'tune': {'mean': None, 'std': None},
                            'train': {'AUC': None, 'AUPR': None},
                            'test': {'AUC': None, 'AUPR': None}}
               for fit_method in ['optim', 'varit', 'sampl']}

    out_time = {fit_method: {'tune': {'fit': dict(), 'score': dict()},
                             'final': {'fit': None, 'score': None}}
                for fit_method in ['optim', 'varit', 'sampl']}

    out_params = {'optim': None, 'varit': None, 'sampl': None}
    out_scores = {fit_method: {'train': None, 'test': None}
                  for fit_method in ['optim', 'varit', 'sampl']}

    use_mtype = sorted(vars_list)[args.task_id]
    ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                if annot['Chr'] == cdata.gene_annot[args.use_gene]['Chr']}

    for fmth in ['optim', 'varit', 'sampl']:
        mut_clf = use_clf(fit_method=fmth)

        mut_clf, cv_output = mut_clf.tune_coh(
            cdata, use_mtype, exclude_feats=ex_genes,
            tune_splits=4, test_count=24, parallel_jobs=8
            )

        out_time[fmth]['tune']['fit']['avg'] = cv_output['mean_fit_time']
        out_time[fmth]['tune']['fit']['std'] = cv_output['std_fit_time']
        out_time[fmth]['tune']['score']['avg'] = cv_output['mean_score_time']
        out_time[fmth]['tune']['score']['std'] = cv_output['std_score_time']

        out_acc[fmth]['tune']['mean'] = cv_output['mean_test_score']
        out_acc[fmth]['tune']['std'] = cv_output['std_test_score']
        out_params[fmth] = {par: mut_clf.get_params()[par]
                            for par, _ in mut_clf.tune_priors}

        t_start = time.time()
        mut_clf.fit_coh(cdata, use_mtype, exclude_feats=ex_genes)
        t_end = time.time()
        out_time[fmth]['final']['fit'] = t_end - t_start

        pheno_list = dict()
        train_omics, pheno_list['train'] = cdata.train_data(
            use_mtype, exclude_feats=ex_genes)
        test_omics, pheno_list['test'] = cdata.test_data(
            use_mtype, exclude_feats=ex_genes)

        t_start = time.time()
        pred_scores = {
            'train': mut_clf.parse_preds(mut_clf.predict_omic(train_omics)),
            'test': mut_clf.parse_preds(mut_clf.predict_omic(test_omics))
            }
        out_time[fmth]['final']['score'] = time.time() - t_start

        samp_sizes = {'train': (sum(cdata.train_pheno(use_mtype))
                                / len(cdata.get_train_samples())),
                      'test': (sum(cdata.test_pheno(use_mtype))
                               / len(cdata.get_test_samples()))}

        for samp_set, scores in pred_scores.items():
            out_scores[fmth][samp_set] = scores

            if len(set(pheno_list[samp_set])) == 2:
                out_acc[fmth][samp_set]['AUC'] = roc_auc_score(
                    pheno_list[samp_set], scores)
                out_acc[fmth][samp_set]['AUPR'] = aupr_score(
                    pheno_list[samp_set], scores)
 
            else:
                out_acc[fmth][samp_set]['AUC'] = 0.5
                out_acc[fmth][samp_set]['AUPR'] = samp_sizes[samp_set]

        if fmth == 'sampl':
            out_sampl = mut_clf.named_steps['fit'].fit_obj.summary()

    with open(os.path.join(args.use_dir, 'output',
                           "out__cv-{}_task-{}.p".format(args.cv_id,
                                                         args.task_id)),
              'wb') as fl:
        pickle.dump({'Acc': out_acc, 'Clf': mut_clf, 'Scores': out_scores,
                     'Params': out_params, 'Time': out_time,
                     'Sampl': out_sampl}, fl)


if __name__ == "__main__":
    main()

