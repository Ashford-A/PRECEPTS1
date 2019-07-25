
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.copy_baseline import *
import argparse
import dill as pickle
from glob import glob
import random
from importlib import import_module

import time
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as msq_error
from sklearn.externals.joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('regress', type=str,
                        help="the name of a mutation regressor")
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
    with open(os.path.join(out_path, "gene-list.p"), 'rb') as fl:
        gene_list = pickle.load(fl)

    rgr_info = args.regress.split('__')
    rgr_module = import_module(
        'HetMan.experiments.copy_baseline.models.{}'.format(rgr_info[0]))
    mut_rgr = getattr(rgr_module, rgr_info[1].capitalize())()

    out_acc = {gene: {'tune': {'mean': None, 'std': None},
                      'train': {'Cor': None, 'MSE': None},
                      'test': {'Cor': None, 'MSE': None}}
               for gene in gene_list}

    out_params = {gene: None for gene in gene_list}
    out_scores = {gene: None for gene in gene_list}

    out_time = {gene: {'tune': {'fit': dict(), 'score': dict()},
                       'final': {'fit': None, 'score': None}}
                for gene in gene_list}

    coh_files = glob(os.path.join(out_path, "*__cohort-data.p"))
    coh_dict = {coh_fl.split('/setup/')[1].split('__')[0]: coh_fl
                for coh_fl in coh_files}
    out_trnsf = {gene: dict() for gene in gene_list}

    for i, copy_gene in enumerate(gene_list):
        if (i % args.task_count) == args.task_id:
            print("Testing {} ...".format(copy_gene))

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

            ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                        if annot['Chr'] == cdata.gene_annot[copy_gene]['Chr']}
            use_genes = feat_list - ex_genes

            mut_rgr, cv_output = mut_rgr.tune_coh(
                cdata, copy_gene, include_feats=use_genes,
                tune_splits=4, test_count=36, parallel_jobs=8
                )

            out_time[copy_gene]['tune']['fit']['avg'] = cv_output[
                'mean_fit_time']
            out_time[copy_gene]['tune']['fit']['std'] = cv_output[
                'std_fit_time']

            out_time[copy_gene]['tune']['score']['avg'] = cv_output[
                'mean_score_time']
            out_time[copy_gene]['tune']['score']['std'] = cv_output[
                'std_score_time']

            out_acc[copy_gene]['tune']['mean'] = cv_output['mean_test_score']
            out_acc[copy_gene]['tune']['std'] = cv_output['std_test_score']
            out_params[copy_gene] = {par: mut_rgr.get_params()[par]
                                     for par, _ in mut_rgr.tune_priors}

            t_start = time.time()
            mut_rgr.fit_coh(cdata, copy_gene, include_feats=use_genes)
            t_end = time.time()
            out_time[copy_gene]['final']['fit'] = t_end - t_start

            if (args.cv_id // 25) < 2:
                pheno_list = dict()
                train_omics, pheno_list['train'] = cdata.train_data(
                    copy_gene, include_feats=use_genes)
                test_omics, pheno_list['test'] = cdata.test_data(
                    copy_gene, include_feats=use_genes)

                t_start = time.time()
                pred_scores = {
                    'train': mut_rgr.parse_preds(
                        mut_rgr.predict_omic(train_omics)),
                    'test': mut_rgr.parse_preds(
                        mut_rgr.predict_omic(test_omics))
                    }

                out_time[copy_gene]['final']['score'] = time.time() - t_start
                out_scores[copy_gene] = pred_scores['test']

                for samp_set, scores in pred_scores.items():
                    if len(set(pheno_list[samp_set])) > 1:
                        out_acc[copy_gene][samp_set]['Cor'] = pearsonr(
                            pheno_list[samp_set], scores)[0]
                        out_acc[copy_gene][samp_set]['MSE'] = msq_error(
                            pheno_list[samp_set], scores)
 
                    else:
                        out_acc[copy_gene][samp_set]['Cor'] = 0
                        out_acc[copy_gene][samp_set]['MSE'] = None

            else:
                out_scores[copy_gene] = [
                    mut_rgr.parse_preds(vals)[0]
                    for vals in mut_rgr.infer_coh(
                        cdata, copy_gene, include_feats=use_genes,
                        infer_splits=5, infer_folds=5, parallel_jobs=5
                        )
                    ]

            out_trnsf[copy_gene] = dict(zip(coh_dict.keys(), [
                mut_rgr.parse_preds(vals) for vals in Parallel(
                    n_jobs=8, pre_dispatch=8)(delayed(mut_rgr.predict_omic)(
                        pickle.load(open(coh_fl, 'rb')).train_data(
                            include_feats=use_genes)[0]
                        )
                        for coh_fl in coh_dict.values())
                ]))

        else:
            del(out_acc[copy_gene])
            del(out_params[copy_gene])
            del(out_scores[copy_gene])
            del(out_time[copy_gene])
            del(out_trnsf[copy_gene])

    with open(os.path.join(args.use_dir, 'output',
                           "out__cv-{}_task-{}.p".format(args.cv_id,
                                                         args.task_id)),
              'wb') as fl:
        pickle.dump({'Acc': out_acc, 'Rgr': mut_rgr.__class__,
                     'Params': out_params, 'Time': out_time,
                     'Scores': out_scores, 'Transfer': out_trnsf}, fl)


if __name__ == "__main__":
    main()

