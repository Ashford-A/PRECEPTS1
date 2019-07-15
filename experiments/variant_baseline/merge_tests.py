
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.variant_baseline import *
import argparse
import dill as pickle
import bz2

import numpy as np
import pandas as pd
from glob import glob
from itertools import product
from operator import itemgetter
import random


class MergeError(Exception):
    pass


def merge_cohort_data(out_dir, use_seed=None):
    cdata_file = os.path.join(out_dir, "cohort-data.p")

    if os.path.isfile(cdata_file):
        with open(cdata_file, 'rb') as fl:
            cur_cdata = pickle.load(fl)
            cur_hash = cur_cdata.data_hash()
            cur_hash = tuple(cur_hash[0]), cur_hash[1]

    else:
        cur_hash = None

    new_files = glob(os.path.join(out_dir, "cohort-data__*.p"))
    new_mdls = [new_file.split("cohort-data__")[1].split(".p")[0]
                for new_file in new_files]

    new_cdatas = {new_mdl: pickle.load(open(new_file, 'rb'))
                  for new_mdl, new_file in zip(new_mdls, new_files)}
    new_chsums = {mdl: cdata.data_hash() for mdl, cdata in new_cdatas.items()}
    new_chsums = {k: (tuple(v[0]), v[1]) for k, v in new_chsums.items()}

    for mdl, cdata in new_cdatas.items():
        if cdata.get_seed() != use_seed:
            raise MergeError("Cohort for model {} does not have the correct "
                             "cross-validation seed!".format(mdl))

        if cdata.get_test_samples():
            raise MergeError("Cohort for model {} does not have an empty "
                             "testing sample set!".format(mdl))

    assert len(set(new_chsums.values())) <= 1, (
        "Inconsistent cohort hashes found for new "
        "experiments in {} !".format(out_dir)
        )

    if new_files:
        if cur_hash is not None:
            assert tuple(new_chsums.values())[0] == cur_hash, (
                "Cohort hash for new experiment in {} does not match hash "
                "for cached cohort!".format(out_dir)
                )
            use_cdata = cur_cdata

        else:
            use_cdata = tuple(new_cdatas.values())[0]
            with open(cdata_file, 'wb') as f:
                pickle.dump(use_cdata, f)

        for new_file in new_files:
            os.remove(new_file)

    else:
        if cur_hash is None:
            raise ValueError("No cohort datasets found in {}, has an "
                             "experiment with these parameters been run to "
                             "completion yet?".format(out_dir))

        else:
            use_cdata = cur_cdata

    return use_cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('use_dir', type=str, default=base_dir)
    args = parser.parse_args()

    out_files = [(fl, int(fl.split('out__cv-')[1].split('_task-')[0]),
                  int(fl.split('_task-')[1].split('.p')[0]))
                  for fl in os.listdir(os.path.join(args.use_dir, 'output'))
                 if 'out__cv-' in fl]

    out_files = sorted(out_files, key=itemgetter(1, 2))
    task_count = len(set(task for _, _, task in out_files))
    out_list = [pickle.load(open(os.path.join(args.use_dir, 'output', fl),
                                 'rb'))
                for fl, _, _ in out_files]

    use_clf = set(ols['Clf'] for ols in out_list)
    if len(use_clf) != 1:
        raise MergeError("Each experiment must be run "
                         "with exactly one classifier!")

    use_tpr = set(ols['Clf'].tune_priors for ols in out_list)
    if len(use_tpr) != 1:
        raise MergeError("Each experiment must be run with the same tuning "
                         "priors across all classifer instances!")

    with open(os.path.join(args.use_dir,
                           'setup', "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)
    with open(os.path.join(args.use_dir, 'setup', "vars-list.p"), 'rb') as fl:
        vars_list = pickle.load(fl)

    fit_acc = {
        samp_set: pd.concat([
            pd.concat([
                pd.DataFrame.from_dict({mtype: acc[samp_set]
                                        for mtype, acc in ols['Acc'].items()})
                for (_, cv_id, task_id), ols in zip(out_files, out_list)
                if task_id == use_task and (cv_id // 25) < 2
                ], axis=0)
            for use_task in range(task_count)
        ], axis=1).transpose()
        for samp_set in ['train', 'test']
        }

    assert fit_acc['train'].shape == (len(vars_list), 100), (
        "Training fit accuracies missing for some tested mutations!")
    assert fit_acc['test'].shape == (len(vars_list), 100), (
        "Testing fit accuracies missing for some tested mutations!")

    tune_list = tuple(product(*[
        vals for _, vals in tuple(use_clf)[0].tune_priors]))

    tune_acc = {
        stat_lbl: pd.concat([
            pd.concat([
                pd.DataFrame.from_dict({mtype: acc['tune'][stat_lbl]
                                        for mtype, acc in ols['Acc'].items()},
                                       orient='index', columns=tune_list)
                for (_, _, task_id), ols in zip(out_files, out_list)
                if task_id == use_task
                ], axis=1, sort=True)
            for use_task in range(task_count)
        ], axis=0) for stat_lbl in ['mean', 'std']
        }

    assert tune_acc['mean'].shape == (len(vars_list), len(tune_list) * 51), (
        "Mean of tuning test accuracies missing for some tested mutations!")
    assert tune_acc['std'].shape == (len(vars_list), len(tune_list) * 51), (
        "Variance of tuning test accuracies missing for some "
        "tested mutations!"
        )

    par_df = pd.concat([
        pd.concat([pd.DataFrame.from_dict(ols['Params'], orient='index')
                   for (_, _, task_id), ols in zip(out_files, out_list)
                   if task_id == use_task], axis=1)
        for use_task in range(task_count)
        ], axis=0)

    assert par_df.shape == (len(vars_list), 51), (
        "Tuned parameter selections missing for some tested mutations!")

    tune_time = {
        stage_lbl: {
            stat_lbl: pd.concat([
                pd.concat([
                    pd.DataFrame.from_dict({
                        mtype: tm['tune'][stage_lbl][stat_lbl]
                        for mtype, tm in ols['Time'].items()
                        }, orient='index', columns=tune_list)
                    for (_, _, task_id), ols in zip(out_files, out_list)
                    if task_id == use_task
                    ], axis=1, sort=True)
                for use_task in range(task_count)
                ], axis=0) for stat_lbl in ['avg', 'std']
            } for stage_lbl in ['fit', 'score']
        }

    trnsf_mtypes = {tuple(sorted(ols['Transfer'])) for ols in out_list}
    assert len(trnsf_mtypes) == 1, (
        "Some tested mutations missing from transferred scores!")
    assert tuple(trnsf_mtypes)[0] == tuple(sorted(vars_list)), (
        "Some tested mutations missing from transferred scores!")

    use_cohs = {tuple(sorted(out_trnsf.keys())) for ols in out_list
                for out_trnsf in ols['Transfer'].values()}
    assert len(use_cohs) == 1, ("Inconsistent cohorts used for transfer "
                                "testing across tested mutations!")
    coh_list = tuple(use_cohs)[0]

    trnsf_dict = {
        coh: pd.concat([pd.concat([
            pd.DataFrame({mtype: trnsf_vals[coh]
                          for mtype, trnsf_vals in ols['Transfer'].items()})
            for (_, _, task_id), ols in zip(out_files, out_list)
            if task_id == use_task], axis=1, sort=True)
            for use_task in range(task_count)], axis=0)
        for coh in coh_list
        }

    for coh, trnsf_vals in trnsf_dict.items():
        assert trnsf_vals.shape[1] == len(vars_list) * 51, (
            "Transfer predictions for some tested mutations missing "
            "for cohort {} !".format(coh)
            )

    score_list = [pd.DataFrame(index=cdata.get_samples())] * 51
    for cv_id in range(51):
        if (cv_id // 25) == 2:
            cdata.update_split(2079 + 57 * cv_id, test_prop=0)
            test_samps = cdata.get_train_samples()

        elif (cv_id // 25) == 1:
            cdata_samps = cdata.get_samples()
            random.seed((cv_id // 5) * 1811 + 9)
            random.shuffle(cdata_samps)

            cdata.update_split(2077 + 57 * cv_id,
                               test_samps=cdata_samps[(cv_id % 5)::5])
            test_samps = cdata.get_test_samples()

        else:
            cdata.update_split(2077 + 57 * cv_id, test_prop=0.2)
            test_samps = cdata.get_test_samples()

        score_list[cv_id] = pd.concat([
            pd.DataFrame(ols['Scores'], index=test_samps) for ols in out_list[
                (cv_id * task_count):((cv_id + 1) * task_count)]
            ])

    score_dict = {
        'random': pd.concat(
            score_list[:25], join='outer', axis=1, sort=True).groupby(
                level=0, axis=1).agg(lambda val_arr: val_arr.apply(
                    lambda x: list(x), axis=1)),

        'fivefold': pd.concat(
            score_list[25:50], join='outer', axis=1, sort=True).groupby(
                level=0, axis=1).agg(lambda val_arr: val_arr.apply(
                    lambda x: list(x), axis=1)),

        'infer': score_list[50]
        }

    for cv_mth, score_df in score_dict.items():
        assert sorted(score_df.columns) == sorted(vars_list), (
            "Mutations for which {} inferred scores were calculated does not "
            "match master list of mutations!".format(cv_mth)
            )

    assert score_dict['random'].shape[0] <= len(cdata.get_samples()), (
        "More samples with inferred scores using random CV folds "
        "than present in the dataset!"
        )

    for cv_mth in ['fivefold', 'infer']:
        assert (sorted(score_dict[cv_mth].index)
                == sorted(cdata.get_samples())), (
                    "Samples with {} inferred scores do not match those "
                    "present in the dataset!"
                    )

    assert set(score_dict['infer'].dtypes) == {np.dtype('float')}, (
        "Some inferred scores missing!")
    for cv_mth in ['random', 'fivefold']:
        assert (score_dict[cv_mth].applymap(len) == 25).values.all(), (
            "Some inferred scores missing for CV method {}!".format(cv_mth))

    assert (
        score_dict['random'].applymap(lambda x: pd.isnull(x).sum()).apply(
            set, axis=1).apply(len) == 1).values.all(), (
                "Inconsistent random CV sampling!")
    assert (score_dict['fivefold'].applymap(lambda x: pd.isnull(x).sum())
            == 20).values.all(), "Inconsistent fivefold CV sampling!"

    with bz2.BZ2File(os.path.join(args.use_dir, "out-data.p.gz"), 'w') as fl:
        pickle.dump({'Tune': {'Acc': tune_acc, 'Time': tune_time},
                     'Fit': fit_acc, 'Params': par_df,
                     'Clf': tuple(use_clf)[0], 'Trnsf': trnsf_dict,
                     'Scores': score_dict}, fl, protocol=-1)


if __name__ == "__main__":
    main()

