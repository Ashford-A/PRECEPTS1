
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_infer.utils import Mcomb, ExMcomb
from dryadic.learning.classifiers import *

import argparse
import dill as pickle
from glob import glob
from joblib import Parallel, delayed


def transfer_model(trnsf_fl, clf, use_feats):
    with open(trnsf_fl, 'rb') as f:
        trnsf_preds = clf.predict_omic(pickle.load(f).train_data(
            include_feats=use_feats)[0])

    return trnsf_preds


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

    args = parser.parse_args()
    setup_dir = os.path.join(args.use_dir, 'setup')

    with open(os.path.join(setup_dir, "feat-list.p"), 'rb') as fl:
        feat_list = pickle.load(fl)
    with open(os.path.join(setup_dir, "muts-list.p"), 'rb') as muts_f:
        mtype_list = pickle.load(muts_f)
    with open(os.path.join(setup_dir, "cohort-data.p"), 'rb') as cdata_f:
        cdata = pickle.load(cdata_f)

    clf = eval(args.classif)
    mut_clf = clf()
    use_gene = {mtype.get_labels()[0] for mtype in mtype_list
                if not isinstance(mtype, (ExMcomb, Mcomb, RandomType))}

    assert len(use_gene) == 1, ("List of mutations to test associated with "
                                "more than one gene!")
    use_gene = tuple(use_gene)[0]

    out_pars = {mtype: {smps: {par: None for par, _ in mut_clf.tune_priors}
                        for smps in ['All', 'Iso']}
                for mtype in mtype_list}

    out_time = {mtype: {smps: dict() for smps in ['All', 'Iso']}
                for mtype in mtype_list}
    out_acc = {mtype: {smps: dict() for smps in ['All', 'Iso']}
               for mtype in mtype_list}
    out_inf = {mtype: {'All': None, 'Iso': None} for mtype in mtype_list}

    coh_files = glob(os.path.join(setup_dir, "*__cohort-data.p"))
    coh_dict = {coh_fl.split('/setup/')[1].split('__')[0]: coh_fl
                for coh_fl in coh_files}

    out_trnsf = {mtype: {'All': {coh: None for coh in coh_dict},
                         'Iso': {coh: None for coh in coh_dict}}
                 for mtype in mtype_list}

    base_lvls = 'Gene', 'Scale', 'Copy', 'Exon', 'Location', 'Protein'
    use_chr = cdata.gene_annot[use_gene]['Chr']
    use_feats = feat_list - {gene for gene, annot in cdata.gene_annot.items()
                             if annot['Chr'] == use_chr}

    # for each subtype, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if (i % args.task_count) == args.task_id:
            print("Isolating {} ...".format(mtype))

            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, include_feats=use_feats,
                tune_splits=4, test_count=48, parallel_jobs=8
                )

            # save the tuned values of the hyper-parameters
            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_pars[mtype]['All'][par] = clf_params[par]

            out_time[mtype]['All']['avg'] = cv_output['mean_fit_time']
            out_time[mtype]['All']['std'] = cv_output['std_fit_time']
            out_acc[mtype]['All']['avg'] = cv_output['mean_test_score']
            out_acc[mtype]['All']['std'] = cv_output['std_test_score']
            out_acc[mtype]['All']['par'] = cv_output['params']

            out_inf[mtype]['All'] = mut_clf.infer_coh(
                cdata, mtype, include_feats=use_feats,
                infer_splits=80, infer_folds=4, parallel_jobs=8
                )
            mut_clf.fit_coh(cdata, mtype, include_feats=use_feats)

            out_trnsf[mtype]['All'] = dict(zip(coh_dict.keys(), [
                mut_clf.parse_preds(vals) for vals in Parallel(
                    backend='threading', n_jobs=8, pre_dispatch=8)(
                        delayed(transfer_model)(trnsf_fl, mut_clf, use_feats)
                        for trnsf_fl in coh_dict.values())
                ]))

            ex_samps = cdata.mtrees[base_lvls][use_gene].get_samples()
            ex_samps -= mtype.get_samples(cdata.mtrees[
                cdata.choose_mtree(mtype)])

            mut_clf, cv_output = mut_clf.tune_coh(
                cdata, mtype, include_feats=use_feats, exclude_samps=ex_samps,
                tune_splits=4, test_count=48, parallel_jobs=8
                )

            clf_params = mut_clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_pars[mtype]['Iso'][par] = clf_params[par]

            out_time[mtype]['Iso']['avg'] = cv_output['mean_fit_time']
            out_time[mtype]['Iso']['std'] = cv_output['std_fit_time']
            out_acc[mtype]['Iso']['avg'] = cv_output['mean_test_score']
            out_acc[mtype]['Iso']['std'] = cv_output['std_test_score']
            out_acc[mtype]['Iso']['par'] = cv_output['params']

            out_inf[mtype]['Iso'] = mut_clf.infer_coh(
                cdata, mtype,
                include_feats=use_feats, force_test_samps=ex_samps,
                infer_splits=80, infer_folds=4, parallel_jobs=8
                )

            mut_clf.fit_coh(cdata, mtype,
                            include_feats=use_feats, exclude_samps=ex_samps)

            out_trnsf[mtype]['Iso'] = dict(zip(coh_dict.keys(), [
                mut_clf.parse_preds(vals) for vals in Parallel(
                    backend='threading', n_jobs=8, pre_dispatch=8)(
                        delayed(transfer_model)(trnsf_fl, mut_clf, use_feats)
                        for trnsf_fl in coh_dict.values())
                ]))

        else:
            del(out_pars[mtype])
            del(out_time[mtype])
            del(out_acc[mtype])
            del(out_inf[mtype])
            del(out_trnsf[mtype])

    with open(os.path.join(args.use_dir, 'output',
                           "out_task-{}.p".format(args.task_id)),
              'wb') as fl:
        pickle.dump({'Infer': out_inf, 'Pars': out_pars, 'Time': out_time,
                     'Acc': out_acc, 'Transfer': out_trnsf,
                     'Clf': mut_clf.__class__},
                    fl, protocol=-1)


if __name__ == "__main__":
    main()

