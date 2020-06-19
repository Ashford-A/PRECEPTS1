
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

import argparse
import bz2
from pathlib import Path
import dill as pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random

from dryadic.features.mutations import MuType
from HetMan.experiments.utilities.mutations import (
    pnt_mtype, shal_mtype, ExMcomb)
from HetMan.experiments.subvariant_isolate.merge_isolate import (
    compare_muts, calculate_auc)

from itertools import cycle
from functools import reduce
from operator import add


def calculate_siml(base_mtype, phn_dict, ex_k, pred_vals):
    cur_genes = set(base_mtype.get_labels())

    none_mean = np.concatenate(pred_vals[~phn_dict[ex_k]].values).mean()
    base_mean = np.concatenate(pred_vals[phn_dict[base_mtype]].values).mean()
    cur_diff = base_mean - none_mean

    return {othr_mtype: (np.concatenate(pred_vals[phn].values).mean()
                         - none_mean) / cur_diff
            for othr_mtype, phn in phn_dict.items()
            if (isinstance(othr_mtype, ExMcomb)
                and set(othr_mtype.get_labels()) == cur_genes)}


def main():
    parser = argparse.ArgumentParser(
        "Processes and consolidates the distributed output of an iteration "
        "of the subgrouping isolation experiment for use in further analyses."
        )

    parser.add_argument('use_dir', type=str, default=base_dir)
    parser.add_argument('--ex_lbls', type=str, nargs='+',
                        choices=['All', 'Iso', 'IsoShal'],
                        default=['All', 'Iso', 'IsoShal'])
    parser.add_argument('--task_ids', type=int, nargs='+')
    args = parser.parse_args()

    # load the -omic datasets for this experiment's tumour cohort
    with bz2.BZ2File(os.path.join(args.use_dir, 'setup',
                                  "cohort-data.p.gz"), 'r') as f:
        cdata = pickle.load(f)

    # get the mutations present in this cohort sorted into the attribute
    # hierarchy used in this experiment as well as the subgroupings tested
    use_mtree = tuple(cdata.mtrees.values())[0]
    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    # get list of output files from all parallelized jobs
    file_list = tuple(Path(args.use_dir, 'output').glob("out__cv-*_task*.p"))
    file_dict = dict()

    for out_fl in file_list:
        fl_info = out_fl.stem.split("out__")[1]
        out_task = int(fl_info.split("task-")[1])

        if args.task_ids is None or out_task in args.task_ids:
            out_cv = int(fl_info.split("cv-")[1].split("_")[0])
            file_dict[out_fl] = out_task, out_cv

    assert (len(file_dict) % 40) == 0, "Missing output files detected!"
    task_count = 1
    with open(os.path.join(args.use_dir, 'setup', "tasks.txt"), 'r') as f:
        task_list = f.readline().strip()

        while task_list:
            task_count = max(task_count,
                             *[int(tsk) + 1 for tsk in task_list.split(' ')])
            task_list = f.readline().strip()

    if args.task_ids is None:
        use_tasks = set(range(task_count))
        out_tag = ''

    else:
        use_tasks = set(args.task_ids)
        out_tag = "_{}".format('-'.join([
            str(tsk) for tsk in sorted(use_tasks)]))

    file_sets = {
        cv_id: {out_fl for out_fl, (out_task, out_cv) in file_dict.items()
                if out_task in use_tasks and out_cv == cv_id}
        for cv_id in range(40)
        }

    # initialize object that will store raw experiment output data
    out_dfs = {k: {ex_lbl: [None for cv_id in range(40)]
                   for ex_lbl in args.ex_lbls}
               for k in ['Pars', 'Time', 'Acc']}
    out_clf = None
    out_tune = None

    use_muts = [mut for i, mut in enumerate(muts_list)
                if i % task_count in use_tasks]

    # initialize object that will store collated classifier scores
    pred_lists = {
        ex_lbl: [
            pd.DataFrame(index=use_muts,
                         columns=cdata.get_samples()).applymap(lambda x: [])
            for cv_id in range(40)
            ]
        for ex_lbl in args.ex_lbls
        }

    if 'Iso' in args.ex_lbls or 'IsoShal' in args.ex_lbls:
        mut_samps = {mut: mut.get_samples(use_mtree) for mut in use_muts}
        mut_genes = {mut: mut.get_labels()[0] for mut in use_muts}
        gene_samps = {gene: mtree.get_samples() for gene, mtree in use_mtree}

    if 'IsoShal' in args.ex_lbls:
        shal_samps = {gene: ExMcomb(pnt_mtype, shal_mtype).get_samples(mtree)
                      for gene, mtree in use_mtree}

    for cv_id, out_fls in file_sets.items():
        out_list = []

        for out_fl in out_fls:
             with open(out_fl, 'rb') as f:
                 out_list += [pickle.load(f)]

        for out_dicts in out_list:
            if out_clf is None:
                out_clf = out_dicts['Clf']

            else:
                assert out_clf == out_dicts['Clf'], (
                    "Each experiment must be run with the same classifier!")

            if out_tune is None:
                out_tune = out_dicts['Clf'].tune_priors

            else:
                assert out_tune == out_dicts['Clf'].tune_priors, (
                    "Each experiment must be run with exactly "
                    "one set of tuning priors!"
                    )

        # recover the cohort training/testing data split that was
        # used to generate the results in this file
        cdata_samps = sorted(cdata.get_samples())
        random.seed((cv_id // 4) * 3901 + 23)
        random.shuffle(cdata_samps)

        cdata.update_split(13101 + 103 * cv_id,
                           test_samps=cdata_samps[(cv_id % 4)::4])

        samps_dict = {'train': cdata.get_train_samples(),
                      'test': cdata.get_test_samples()}

        for ex_lbl in args.ex_lbls:
            out_preds = pd.DataFrame({
                mut: out_vals[ex_lbl] for out_dicts in out_list
                for mut, out_vals in out_dicts['Pred'].items()
                }).transpose()

            assert sorted(out_preds.index) == sorted(use_muts), (
                "Mutations with `{}` predictions for c-v fold <{}> don't "
                "match those enumerated during setup!".format(ex_lbl, cv_id)
                )

            pred_lists[ex_lbl][cv_id][samps_dict['test']] = pd.DataFrame(
                out_preds.test.values.tolist(),
                index=out_preds.index, columns=samps_dict['test']
                ).applymap(lambda x: [x])

            if 'train' in out_preds:
                train_mat = out_preds.train[~out_preds.train.isnull()]
            
                for mut, train_preds in train_mat.iteritems():
                    use_samps = gene_samps[mut_genes[mut]] - mut_samps[mut]
                    if ex_lbl == 'IsoShal':
                        use_samps -= shal_samps[mut_genes[mut]]

                    out_samps = sorted(use_samps & set(samps_dict['train']))
                    pred_lists[ex_lbl][cv_id].loc[mut][out_samps] = [
                        [x] for x in train_preds]

        for k in out_dfs:
            for ex_lbl in args.ex_lbls:
                out_dfs[k][ex_lbl][cv_id] = pd.DataFrame({
                    mut: out_vals[ex_lbl] for out_dicts in out_list
                    for mut, out_vals in out_dicts[k].items()
                    }).transpose()

    pred_dfs = {ex_lbl: reduce(add, pred_mats)
                for ex_lbl, pred_mats in pred_lists.items()}

    if 'All' in args.ex_lbls:
        assert (pred_dfs['All'].applymap(len) == 10).values.all(), (
            "Incorrect number of testing CV scores!")

    if 'Iso' in args.ex_lbls or 'IsoShal' in args.ex_lbls:
        for mut in pred_dfs[args.ex_lbls[0]].index:
            hld_samps = gene_samps[mut_genes[mut]] - mut_samps[mut]

            if 'Iso' in args.ex_lbls:
                assert (pred_dfs['Iso'].loc[
                    mut, set(cdata.get_samples()) - hld_samps].apply(len)
                    == 10).all(), ("Incorrect number of testing CV scores!")

                assert (pred_dfs['Iso'].loc[
                    mut, hld_samps].apply(len) == 40).all(), (
                        "Incorrect number of testing CV scores!")

            if 'IsoShal' in args.ex_lbls:
                hld_samps -= shal_samps[mut_genes[mut]]

                assert (pred_dfs['IsoShal'].loc[
                    mut, set(cdata.get_samples()) - hld_samps].apply(len)
                    == 10).all(), ("Incorrect number of testing CV scores!")

                assert (pred_dfs['IsoShal'].loc[
                    mut, hld_samps].apply(len) == 40).all(), (
                        "Incorrect number of testing CV scores!")

    pars_dfs = {ex_lbl: pd.concat(out_dfs['Pars'][ex_lbl], axis=1)
                for ex_lbl in args.ex_lbls}

    for pars_df in pars_dfs.values():
        assert pars_df.shape[1] == (40 * len(out_clf.tune_priors)), (
            "Tuned parameter values missing for some CVs!")

    time_dfs = {ex_lbl: pd.concat(out_dfs['Time'][ex_lbl], axis=1)
                for ex_lbl in args.ex_lbls}

    for time_df in time_dfs.values():
        assert time_df.shape[1] == 80, (
            "Algorithm fitting times missing for some CVs!")
        assert (time_df.applymap(len) == out_clf.test_count).values.all(), (
            "Algorithm fit times missing for some hyper-parameter values!")

    acc_dfs = {ex_lbl: pd.concat(out_dfs['Acc'][ex_lbl], axis=1)
               for ex_lbl in args.ex_lbls}

    for acc_df in acc_dfs.values():
        assert acc_df.shape[1] == 120, (
            "Algorithm tuning accuracies missing for some CVs!")
        assert (acc_df.applymap(len) == out_clf.test_count).values.all(), (
            "Algorithm tuning stats missing for some hyper-parameter values!")

    for out_dfs in [pred_dfs, pars_dfs, time_dfs, acc_dfs]:
        for out_df in out_dfs.values():
            assert compare_muts(out_df.index, use_muts), (
                "Mutations for which predictions were made do not match "
                "the list of mutations enumerated during setup!"
                )

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-pred{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(pred_dfs, fl, protocol=-1)

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-tune{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump([pars_dfs, time_dfs, acc_dfs, out_clf], fl, protocol=-1)

    cdata.update_split(test_prop=0)
    train_samps = np.array(cdata.get_train_samples())
    pheno_dict = {mut: np.array(cdata.train_pheno(mut)) for mut in use_muts}

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-pheno{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    # calculates AUCs for prediction tasks using scores from all
    # cross-validations concatenated together...
    auc_dicts = {
        ex_lbl: {
            'all': pd.Series(dict(zip(use_muts, Parallel(
                n_jobs=12, prefer='threads', pre_dispatch=120)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype],
                        pred_dfs[ex_lbl].loc[mtype][train_samps],
                        )
                    for mtype in use_muts
                    )
                ))),

            # ...and for each cross-validation run considered separately...
            'CV': pd.DataFrame.from_records(
                tuple(zip(cycle(use_muts), Parallel(
                    n_jobs=12, prefer='threads', pre_dispatch=120)(
                        delayed(calculate_auc)(
                            pheno_dict[mtype],
                            pred_dfs[ex_lbl].loc[mtype][train_samps],
                            cv_indx=cv_id
                            )
                        for cv_id in range(10) for mtype in use_muts
                        )
                    ))
                ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0],

            # ...and finally using the average of predicted scores for each
            # sample across CV runs
            'mean': pd.Series(dict(zip(use_muts, Parallel(
                n_jobs=12, prefer='threads', pre_dispatch=120)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype],
                        pred_dfs[ex_lbl].loc[mtype][train_samps],
                        use_mean=True
                        )
                    for mtype in use_muts
                    )
                )))
            }
        for ex_lbl in args.ex_lbls
        }

    for ex_lbl in args.ex_lbls:
        auc_dicts[ex_lbl]['CV'].name = None
        auc_dicts[ex_lbl]['CV'].index.name = None

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-aucs{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(auc_dicts, fl, protocol=-1)

    random.seed(9903)
    sub_inds = [random.choices([False, True], k=len(cdata.get_samples()))
                for _ in range(100)]

    conf_lists = {
        ex_lbl: {
            'mean': pd.DataFrame.from_records(
                tuple(zip(cycle(use_muts), Parallel(
                    n_jobs=12, prefer='threads', pre_dispatch=120)(
                        delayed(calculate_auc)(
                            pheno_dict[mtype][sub_indx],
                            pred_dfs[ex_lbl].loc[
                                mtype][train_samps[sub_indx]],
                            use_mean=True
                            )
                        for sub_indx in sub_inds for mtype in use_muts
                        )
                    ))
                ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0]
            }
        for ex_lbl in args.ex_lbls
        }

    for ex_lbl in args.ex_lbls:
        conf_lists[ex_lbl]['mean'].name = None
        conf_lists[ex_lbl]['mean'].index.name = None

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-conf{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(conf_lists, fl, protocol=-1)

    if 'Iso' in args.ex_lbls or 'IsoShal' in args.ex_lbls:
        ex_mtypes = [('Iso', MuType({})), ('IsoShal', shal_mtype)]
        use_exs = [(ex_lbl, ex_mtype) for ex_lbl, ex_mtype in ex_mtypes
                   if ex_lbl in args.ex_lbls]

        mcomb_lists = {
            'Iso': {mut for mut in use_muts
                    if (isinstance(mut, ExMcomb)
                        and not (mut.all_mtype & shal_mtype).is_empty())},
            'IsoShal': {mut for mut in use_muts
                        if (isinstance(mut, ExMcomb)
                            and (mut.all_mtype & shal_mtype).is_empty())}
            }

        for ex_lbl, ex_mtype in use_exs:
            for mcomb in mcomb_lists[ex_lbl]:
                all_mtype = MuType({('Gene', mut_genes[mcomb]): use_mtree[
                    mut_genes[mcomb]].allkey()})

                gene_ex = MuType({('Gene', mut_genes[mcomb]): ex_mtype})
                pheno_dict[ex_lbl, mcomb] = np.array(cdata.train_pheno(
                    all_mtype - gene_ex))

        siml_dicts = {
            ex_lbl: dict(zip(mcomb_lists[ex_lbl], Parallel(
                n_jobs=12, prefer='threads', pre_dispatch=120)(
                    delayed(calculate_siml)(
                        mcomb, pheno_dict, (ex_lbl, mcomb),
                        pred_dfs[ex_lbl].loc[mcomb][train_samps]
                        )
                    for mcomb in mcomb_lists[ex_lbl]
                    )
                ))
            for ex_lbl in set(args.ex_lbls) & {'Iso', 'IsoShal'}
            }

        with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                      "out-siml{}.p.gz".format(out_tag)),
                         'w') as fl:
            pickle.dump(siml_dicts, fl, protocol=-1)


if __name__ == "__main__":
    main()

