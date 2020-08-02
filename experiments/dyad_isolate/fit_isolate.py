
from ..utilities.handle_input import safe_load
from ..utilities.mutations import pnt_mtype, shal_mtype, ExMcomb
from ..subvariant_isolate.classifiers import *

import os
import argparse
import dill as pickle
import random

from functools import reduce
from operator import or_


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        "Trains classifiers to predict the presence of genes' mutation "
        "subgroupings in a tumor cohort while isolating this presence from "
        "that of other mutations on the same gene as well as a paired gene."
        )

    parser.add_argument('classif', type=str,
                        help="a classifier in HetMan.predict.classifiers")
    parser.add_argument('use_dir', type=str)

    parser.add_argument(
        '--task_count', type=int, default=1,
        help='how many parallel tasks the list of types to test is split into'
        )

    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')
    parser.add_argument('--cv_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')

    # load mutation subgroupings previously enumerated for testing
    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        muts_list = pickle.load(muts_f)

    # load tumour cohort paired expression and mutation data
    cdata = safe_load(os.path.join(setup_dir, "cohort-data.p.gz"),
                      retry_pause=31)

    base_tree = tuple(cdata.mtrees.values())[0]
    clf = eval(args.classif)
    mut_clf = clf()

    # split the tumour cohort into training and testing subsets
    use_seed = 13101 + 103 * args.cv_id
    cdata_samps = sorted(cdata.get_samples())
    random.seed((args.cv_id // 4) * 3901 + 23)
    random.shuffle(cdata_samps)
    cdata.update_split(use_seed, test_samps=cdata_samps[(args.cv_id % 4)::4])

    out_pars = {mut: {smps: {par: None for par, _ in mut_clf.tune_priors}
                      for smps in ['All', 'Iso', 'IsoShal']}
                for mut in muts_list}

    out_time = {mut: {smps: dict() for smps in ['All', 'Iso', 'IsoShal']}
                for mut in muts_list}
    out_acc = {mut: {smps: dict() for smps in ['All', 'Iso', 'IsoShal']}
               for mut in muts_list}
    out_pred = {mut: {smps: None for smps in ['All', 'Iso', 'IsoShal']}
                for mut in muts_list}

    # for each subgrouping, check if it has been assigned to this task
    for i, mut in enumerate(muts_list):
        if (i % args.task_count) == args.task_id:
            print("Isolating {} ...".format(mut))

            cur_genes = mut.get_labels()
            ex_genes = cdata.get_cis_genes('Chrm', cur_genes=cur_genes)
            gene_samps = reduce(or_, [base_tree[gene].get_samples()
                                      for gene in cur_genes])

            shal_samps = reduce(
                or_,
                [ExMcomb(pnt_mtype, shal_mtype).get_samples(base_tree[gene])
                 for gene in cur_genes]
                )
            mut_samps = mut.get_samples(cdata.mtrees)

            ex_dict = {'All': set(), 'Iso': gene_samps - mut_samps,
                       'IsoShal': gene_samps - mut_samps - shal_samps}

            for ex_lbl, ex_samps in ex_dict.items():
                mut_clf, cv_output = mut_clf.tune_coh(
                    cdata, mut, exclude_feats=ex_genes,
                    exclude_samps=ex_samps, tune_splits=4,
                    test_count=mut_clf.test_count, parallel_jobs=8
                    )

                # save the tuned values of the hyper-parameters
                clf_params = mut_clf.get_params()
                for par, _ in mut_clf.tune_priors:
                    out_pars[mut][ex_lbl][par] = clf_params[par]

                out_time[mut][ex_lbl]['avg'] = cv_output['mean_fit_time']
                out_time[mut][ex_lbl]['std'] = cv_output['std_fit_time']
                out_acc[mut][ex_lbl]['avg'] = cv_output['mean_test_score']
                out_acc[mut][ex_lbl]['std'] = cv_output['std_test_score']
                out_acc[mut][ex_lbl]['par'] = cv_output['params']

                mut_clf.fit_coh(cdata, mut, exclude_feats=ex_genes,
                                exclude_samps=ex_samps)

                # make predictions for samples in the testing cohort split
                out_pred[mut][ex_lbl] = {
                    'test': np.round(mut_clf.parse_preds(
                        mut_clf.predict_test(cdata, lbl_type='raw',
                                             exclude_feats=ex_genes)
                        ), 7)
                    }

                # make predictions for samples in the training cohort split
                # that were held out when isolating the given subgrouping
                if ex_samps & set(cdata.get_train_samples()):
                    out_pred[mut][ex_lbl]['train'] = np.round(
                        mut_clf.parse_preds(mut_clf.predict_train(
                            cdata, lbl_type='raw',
                            exclude_feats=ex_genes, include_samps=ex_samps
                            )),
                        7)

        else:
            del(out_pars[mut])
            del(out_time[mut])
            del(out_acc[mut])
            del(out_pred[mut])

    with open(os.path.join(args.use_dir, 'output',
                           "out__cv-{}_task-{}.p".format(
                               args.cv_id, args.task_id)),
              'wb') as fl:
        pickle.dump({'Pred': out_pred, 'Pars': out_pars, 'Time': out_time,
                     'Acc': out_acc, 'Clf': mut_clf.__class__},
                    fl, protocol=-1)


if __name__ == "__main__":
    main()

