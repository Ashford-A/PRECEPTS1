
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_tour.fit_tour import get_excluded_genes
from HetMan.experiments.subvariant_infer.fit_infer import transfer_model
from HetMan.experiments.subvariant_test.classifiers import *

import argparse
import dill as pickle
import time
import random

from pathlib import Path
from joblib import Parallel, delayed


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        "Isolate the expression signature of mutation subtypes from their "
        "parent gene(s)' signature or that of a list of genes in a given "
        "TCGA cohort."
        )

    # positional arguments for which cohort of samples and which mutation
    # classifier to use for testing
    parser.add_argument('classif', type=str,
                        help="a classifier in HetMan.predict.classifiers")
    parser.add_argument('--use_dir', type=str, default=base_dir)

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )

    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')
    parser.add_argument('--cv_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')
 
    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)
    with open(os.path.join(setup_dir, "cohort-data.p"), 'rb') as cdata_f:
        cdata = pickle.load(cdata_f)
    with open(os.path.join(setup_dir, "feat-list.p"), 'rb') as fl:
        feat_list = pickle.load(fl)

    clf = eval(args.classif)
    mut_clf = clf()
    coh_path = os.path.join(args.use_dir, 'setup', "cohort-data.p")
    use_seed = 9073 + 97 * args.cv_id

    cdata = None
    while cdata is None:
        try:
            with open(coh_path, 'rb') as cdata_f:
                cdata = pickle.load(cdata_f)

        except:
            print("Failed to load cohort data, trying again...")
            time.sleep(61)

    cdata_samps = cdata.get_samples()
    random.seed((args.cv_id // 4) * 7712 + 13)
    random.shuffle(cdata_samps)
    cdata.update_split(use_seed, test_samps=cdata_samps[(args.cv_id % 4)::4])

    mtype_genes = {mtype: mtype.get_labels()[0] for mtype in mtype_list
                   if not isinstance(mtype, RandomType)}

    out_pars = {mtype: {par: None for par, _ in mut_clf.tune_priors}
                for mtype in mtype_list}
    out_time = {mtype: dict() for mtype in mtype_list}
    out_acc = {mtype: dict() for mtype in mtype_list}
    out_pred = {mtype: dict() for mtype in mtype_list}

    coh_dict = {coh_fl.stem.split('__')[1]: coh_fl
                for coh_fl in Path(setup_dir).glob("cohort-data__*.p")}
    out_trnsf = {mtype: {coh: None for coh in coh_dict}
                 for mtype in mtype_list}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Testing {} ...".format(mtype))

            if not isinstance(mtype, RandomType):
                use_gene = mtype_genes[mtype]

            elif mtype.base_mtype is not None:
                use_gene = mtype.base_mtype.get_labels()[0]
            else:
                use_gene = random.choice(list(mtype_genes.values()))

            ex_genes = get_excluded_genes('Chrm', use_gene, cdata.gene_annot)
            use_feats = feat_list - ex_genes

            # tune the hyper-parameters of the classifier
            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, include_feats=use_feats,
                tune_splits=4, test_count=mut_clf.test_count, parallel_jobs=8
                )

            # save the tuned values of the hyper-parameters
            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_pars[mtype][par] = clf_params[par]

            out_time[mtype]['avg'] = cv_output['mean_fit_time']
            out_time[mtype]['std'] = cv_output['std_fit_time']
            out_acc[mtype]['avg'] = cv_output['mean_test_score']
            out_acc[mtype]['std'] = cv_output['std_test_score']
            out_acc[mtype]['par'] = cv_output['params']

            mut_clf.fit_coh(cdata, mtype, include_feats=use_feats)
            out_pred[mtype] = mut_clf.parse_preds(mut_clf.predict_test(
                cdata, lbl_type='raw', include_feats=use_feats))

            out_trnsf[mtype] = dict(zip(coh_dict.keys(), [
                mut_clf.parse_preds(vals) for vals in Parallel(
                    backend='threading', n_jobs=8, pre_dispatch=8)(
                        delayed(transfer_model)(trnsf_fl, mut_clf, use_feats)
                        for trnsf_fl in coh_dict.values())
                ]))

        else:
            del(out_pars[mtype])
            del(out_time[mtype])
            del(out_acc[mtype])
            del(out_pred[mtype])
            del(out_trnsf[mtype])

    with open(os.path.join(args.use_dir, 'output',
                           "out__cv-{}_task-{}.p".format(
                               args.cv_id, args.task_id)),
              'wb') as fl:
        pickle.dump({'Pred': out_pred, 'Pars': out_pars, 'Time': out_time,
                     'Acc': out_acc, 'Transfer': out_trnsf,
                     'Clf': mut_clf.__class__},
                    fl, protocol=-1)


if __name__ == "__main__":
    main()

