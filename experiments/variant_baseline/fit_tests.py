
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.variant_baseline import *
import argparse
import dill as pickle
from glob import glob
import random
from importlib import import_module

import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as aupr_score
from sklearn.externals.joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('classif', type=str,
                        help="the name of a mutation classifier")
    parser.add_argument('--use_dir', type=str, default=base_dir)

    parser.add_argument(
        '--task_count', type=int, default=10,
        help="how many parallel tasks the list of types to test is split into"
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help="the subset of subtypes to assign to this task")
    parser.add_argument('--cv_id', type=int, default=0,
                        help="the seed to use for random sampling")

    args = parser.parse_args()
    out_path = os.path.join(args.use_dir, 'setup')
    use_seed = 2079 + 57 * args.cv_id

    with open(os.path.join(out_path, "feat-list.p"), 'rb') as fl:
        feat_list = pickle.load(fl)
    with open(os.path.join(out_path, "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)
    with open(os.path.join(out_path, "vars-list.p"), 'rb') as fl:
        vars_list = pickle.load(fl)

    clf_info = args.classif.split('__')
    clf_module = import_module(
        'HetMan.experiments.variant_baseline.models.{}'.format(clf_info[0]))
    mut_clf = getattr(clf_module, clf_info[1].capitalize())()

    out_acc = {mtype: {'tune': {'mean': None, 'std': None},
                       'train': {'AUC': None, 'AUPR': None},
                       'test': {'AUC': None, 'AUPR': None}}
               for mtype in vars_list}

    out_params = {mtype: None for mtype in vars_list}
    out_scores = {mtype: None for mtype in vars_list}

    out_time = {mtype: {'tune': {'fit': dict(), 'score': dict()},
                        'final': {'fit': None, 'score': None}}
                for mtype in vars_list}

    coh_files = glob(os.path.join(out_path, "*__cohort-data.p"))
    coh_dict = {coh_fl.split('/setup/')[1].split('__')[0]: coh_fl
                for coh_fl in coh_files}
    out_trnsf = {mtype: dict() for mtype in vars_list}

    for i, mtype in enumerate(vars_list):
        if (i % args.task_count) == args.task_id:
            print("Testing {} ...".format(mtype))

            if (args.cv_id // 25) == 2:
                cdata.update_split(use_seed)

            elif (args.cv_id // 25) == 1:
                cdata_samps = cdata.get_samples()

                random.seed((args.cv_id // 5) * 1811 + 9)
                random.shuffle(cdata_samps)
                cdata.update_split(
                    use_seed, test_samps=cdata_samps[(args.cv_id % 5)::5])

            elif (args.cv_id // 25) == 0:
                cdata.update_split(use_seed, test_prop=0.2)

            else:
                raise ValueError("Invalid cross-validation id!")

            # get the gene that the variant is associated with and the list
            # of genes on the same chromosome as that gene
            var_gene = mtype.subtype_list()[0][0]
            ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                        if annot['Chr'] == cdata.gene_annot[var_gene]['Chr']}
            use_genes = feat_list - ex_genes

            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, include_feats=use_genes,
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
            out_params[mtype] = {par: mut_clf.get_params()[par]
                                 for par, _ in mut_clf.tune_priors}

            t_start = time.time()
            mut_clf.fit_coh(cdata, mtype, include_feats=use_genes)
            t_end = time.time()
            out_time[mtype]['final']['fit'] = t_end - t_start

            if (args.cv_id // 25) < 2:
                pheno_list = dict()
                train_omics, pheno_list['train'] = cdata.train_data(
                    mtype, include_feats=use_genes)
                test_omics, pheno_list['test'] = cdata.test_data(
                    mtype, include_feats=use_genes)

                t_start = time.time()
                pred_scores = {
                    'train': mut_clf.parse_preds(
                        mut_clf.predict_omic(train_omics)),
                    'test': mut_clf.parse_preds(
                        mut_clf.predict_omic(test_omics))
                    }

                out_time[mtype]['final']['score'] = time.time() - t_start
                out_scores[mtype] = pred_scores['test']

                samp_sizes = {'train': (sum(cdata.train_pheno(mtype))
                                        / len(cdata.get_train_samples())),
                              'test': (sum(cdata.test_pheno(mtype))
                                       / len(cdata.get_test_samples()))}

                for smp_set, scores in pred_scores.items():
                    if len(set(pheno_list[smp_set])) == 2:
                        out_acc[mtype][smp_set]['AUC'] = roc_auc_score(
                            pheno_list[smp_set], scores)
                        out_acc[mtype][smp_set]['AUPR'] = aupr_score(
                            pheno_list[smp_set], scores)
 
                    else:
                        out_acc[mtype][smp_set]['AUC'] = 0.5
                        out_acc[mtype][smp_set]['AUPR'] = samp_sizes[smp_set]

            else:
                out_scores[mtype] = [
                    mut_clf.parse_preds(vals) for vals in mut_clf.infer_coh(
                        cdata, mtype, include_feats=use_genes,
                        infer_splits=5, infer_folds=5, parallel_jobs=5
                        )
                    ]

            out_trnsf[mtype] = dict(zip(
                coh_dict.keys(),
                Parallel(n_jobs=8, pre_dispatch=8)(
                    delayed(mut_clf.predict_omic)(
                        pickle.load(open(coh_fl, 'rb')).train_data(
                            mtype, include_feats=use_genes)[0]
                        )
                    for coh_fl in coh_dict.values()
                    )
                ))

        else:
            del(out_acc[mtype])
            del(out_params[mtype])
            del(out_scores[mtype])
            del(out_time[mtype])
            del(out_trnsf[mtype])

    with open(os.path.join(args.use_dir, 'output',
                           "out__cv-{}_task-{}.p".format(args.cv_id,
                                                         args.task_id)),
              'wb') as fl:
        pickle.dump({'Acc': out_acc, 'Clf': mut_clf.__class__,
                     'Params': out_params, 'Time': out_time,
                     'Scores': out_scores, 'Transfer': out_trnsf}, fl)


if __name__ == "__main__":
    main()

