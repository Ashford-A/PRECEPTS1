
from ..subgrouping_tour import cis_lbls
from ..utilities.pipeline_setup import get_task_count
from ..utilities.misc import compare_muts
from ..subgrouping_test.gather_test import calculate_auc

import os
import argparse
import bz2
from pathlib import Path
import dill as pickle
from joblib import Parallel, delayed
import random

import numpy as np
import pandas as pd

from itertools import cycle, product
from itertools import combinations as combn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('use_dir', type=str)
    parser.add_argument('--task_ids', type=int, nargs='+')
    args = parser.parse_args()

    # load the -omic datasets for this experiment's tumour cohort
    with bz2.BZ2File(os.path.join(args.use_dir, 'setup',
                                  "cohort-data.p.gz"), 'r') as f:
        cdata = pickle.load(f)

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    # get list of output files from all parallelized jobs
    file_list = tuple(Path(args.use_dir, 'output').glob("out__cv-*_task*.p"))
    file_dict = dict()

    # filter output files according to whether they came from one of the
    # parallelized tasks assigned to this gather task
    for out_fl in file_list:
        fl_info = out_fl.stem.split("out__")[1]
        out_task = int(fl_info.split("task-")[1])

        # gets the parallelized task id and learning cross-validation fold
        # each output file corresponds to
        if args.task_ids is None or out_task in args.task_ids:
            out_cv = int(fl_info.split("cv-")[1].split("_")[0])
            file_dict[out_fl] = out_task, out_cv

    # find the number of parallelized tasks used in this run of the pipeline
    assert (len(file_dict) % 40) == 0, "Missing output files detected!"
    task_count = get_task_count(args.use_dir)

    if args.task_ids is None:
        use_tasks = set(range(task_count))
        out_tag = ''

    else:
        use_tasks = set(args.task_ids)
        out_tag = "_{}".format('-'.join([
            str(tsk) for tsk in sorted(use_tasks)]))

    # organize output files according to their cross-validation fold for
    # easier collation of output data across parallelized task ids
    file_sets = {
        cv_id: {out_fl for out_fl, (out_task, out_cv) in file_dict.items()
                if out_task in use_tasks and out_cv == cv_id}
        for cv_id in range(40)
        }

    # initialize object that will store raw experiment output data
    out_dfs = {k: {cis_lbl: [None for cv_id in range(40)]
                   for cis_lbl in cis_lbls}
               for k in ['Pred', 'Pars', 'Time', 'Acc']}
    out_clf = None
    out_tune = None

    random.seed(10301)
    random.shuffle(muts_list)

    use_muts = [mut for i, mut in enumerate(muts_list)
                if i % task_count in use_tasks]

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

        for k in out_dfs:
            for cis_lbl in cis_lbls:
                out_dfs[k][cis_lbl][cv_id] = pd.concat([
                    pd.DataFrame.from_dict({
                        mtype: out_dict[cis_lbl]
                        for mtype, out_dict in out_dicts[k].items()
                        }, orient='index')
                    for out_dicts in out_list
                    ])

                assert (sorted(out_dfs[k][cis_lbl][cv_id].index)
                        == sorted(use_muts)), (
                    "Mutations with predictions for c-v fold <{}> don't "
                    "match those enumerated during setup!".format(cv_id)
                    )

        # recover the cohort training/testing data split that was
        # used to generate the results in this file
        cdata_samps = sorted(cdata.get_samples())
        random.seed((cv_id // 4) * 7712 + 13)
        random.shuffle(cdata_samps)
        cdata.update_split(9073 + 97 * cv_id,
                           test_samps=cdata_samps[(cv_id % 4)::4])

        test_samps = cdata.get_test_samples()
        for cis_lbl in cis_lbls:
            out_dfs['Pred'][cis_lbl][cv_id].columns = test_samps

    pred_dfs = {cis_lbl: pd.concat(pred_mats, axis=1)
                for cis_lbl, pred_mats in out_dfs['Pred'].items()}

    for cis_lbl, pred_df in pred_dfs.items():
        assert all(smp in pred_df.columns for smp in cdata.get_samples()), (
            "Missing mutation scores for some samples in the cohort!")
        assert (pred_df.columns.value_counts() == 10).all(), (
            "Inconsistent number of CV scores across cohort samples!")

    pars_dfs = {cis_lbl: pd.concat(pars_mats, axis=1)
                for cis_lbl, pars_mats in out_dfs['Pars'].items()}
    for cis_lbl, pars_df in pars_dfs.items():
        assert pars_df.shape[1] == (40 * len(out_clf.tune_priors)), (
            "Tuned parameter values missing for some CVs!")

    time_dfs = {cis_lbl: pd.concat(time_mats, axis=1)
                for cis_lbl, time_mats in out_dfs['Time'].items()}

    for cis_lbl, time_df in time_dfs.items():
        assert time_df.shape[1] == 80, (
            "Model fitting times missing for some CVs!")
        assert (time_df.applymap(len) == out_clf.test_count).values.all(), (
            "Model fitting times missing for some hyper-parameter values!")

    acc_dfs = {cis_lbl: pd.concat(acc_mats, axis=1)
               for cis_lbl, acc_mats in out_dfs['Acc'].items()}

    for cis_lbl, acc_df in acc_dfs.items():
        assert acc_df.shape[1] == 120, (
            "Algorithm tuning accuracies missing for some CVs!")
        assert (acc_df.applymap(len) == out_clf.test_count).values.all(), (
            "Algorithm tuning stats missing for some hyper-parameter values!")

    for cis_lbl, pred_df in pred_dfs.items():
        assert compare_muts(pred_df.index, use_muts), (
            "Mutations for which predictions were made do not match the list "
            "of mutations enumerated during setup!"
            )

    for cis_lbl1, cis_lbl2 in combn(cis_lbls, 2):
        assert compare_muts(
            pred_dfs[cis_lbl1].index, pred_dfs[cis_lbl2].index,
            time_dfs[cis_lbl1].index, time_dfs[cis_lbl2].index
            ), ("Mutations tested using cis-exclusion strategy {} do "
                "not match those tested using strategy {}!".format(
                    cis_lbl1, cis_lbl2))

    for cis_lbl1, cis_lbl2 in product(cis_lbls, repeat=2):
        assert compare_muts(
            pred_dfs[cis_lbl1].index, pars_dfs[cis_lbl2].index,
            time_dfs[cis_lbl1].index, acc_dfs[cis_lbl2].index
            ), ("Mutations with predicted scores do not match those for "
                "which tuned hyper-parameter values are available!")

    pred_dfs = {
        cis_lbl: pd.DataFrame({
            mtype: pred_df.loc[mtype].groupby(level=0).apply(
                lambda x: x.values)
            for mtype in use_muts
            }).transpose()
        for cis_lbl, pred_df in pred_dfs.items()
        }

    for cis_lbl, pred_df in pred_dfs.items():
        assert (pred_df.applymap(len) == 10).values.all(), (
            "Incorrect number of testing CV scores for cis-exclusion "
            "label `{}`!".format(cis_lbl)
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
    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in use_muts}

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-pheno{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    auc_vals = pd.DataFrame({
        cis_lbl: pd.Series(dict(zip(use_muts, Parallel(
            n_jobs=12, prefer='threads', pre_dispatch=120)(
                delayed(calculate_auc)(
                    pheno_dict[mtype],
                    np.vstack(pred_df.loc[
                        mtype][train_samps].values).mean(axis=1)
                    )
                for mtype in use_muts
                )
            )))
        for cis_lbl, pred_df in pred_dfs.items()
        })

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-aucs{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(auc_vals, fl, protocol=-1)

    random.seed(7609)
    sub_inds = [random.choices([False, True], k=len(cdata.get_samples()))
                for _ in range(1000)]

    conf_dict = {
        cis_lbl: pd.DataFrame.from_records(
            tuple(zip(cycle(use_muts), Parallel(
                n_jobs=12, prefer='threads', pre_dispatch=120)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype][sub_indx],
                        np.vstack(pred_df.loc[
                            mtype][train_samps[sub_indx]].values).mean(axis=1)
                        )
                    for sub_indx in sub_inds for mtype in use_muts
                    )
                ))
            ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0]
        for cis_lbl, pred_df in pred_dfs.items()
        }

    with bz2.BZ2File(os.path.join(args.use_dir, 'merge',
                                  "out-conf{}.p.gz".format(out_tag)),
                     'w') as fl:
        pickle.dump(conf_dict, fl, protocol=-1)


if __name__ == "__main__":
    main()

