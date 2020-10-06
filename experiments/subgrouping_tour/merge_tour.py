
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random

from HetMan.experiments.subvariant_tour import cis_lbls
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.merge_test import (
    compare_muts, calculate_auc)

from itertools import product, cycle
from itertools import combinations as combn


def hash_cdata(cdata):
    cdata_hash = {'expr': tuple(cdata.data_hash()[0])}
    for lvls, mtree in cdata.mtrees.items():
        cdata_hash[lvls] = hash(mtree)

    return cdata_hash


def merge_cohort_data(out_dir, use_seed=None):
    cdata_file = os.path.join(out_dir, "cohort-data.p")

    if os.path.isfile(cdata_file):
        with open(cdata_file, 'rb') as fl:
            cur_cdata = pickle.load(fl)
            cur_hash = hash_cdata(cur_cdata)

    else:
        cur_hash = None

    #TODO: handle case where input `mut_lvls` is malformed
    # eg. Domain_SMART instead of Domain_SMART__Form_base
    new_files = glob(os.path.join(out_dir, "cohort-data__*.p"))
    new_mdls = [new_file.split("cohort-data__")[1].split(".p")[0]
                for new_file in new_files]
    new_cdatas = {new_mdl: None for new_mdl in new_mdls}
    new_chsums = {new_mdl: None for new_mdl in new_mdls}

    for new_mdl, new_file in zip(new_mdls, new_files):
        with open(new_file, 'rb') as f:
            new_cdatas[new_mdl] = pickle.load(f)
        new_chsums[new_mdl] = hash_cdata(new_cdatas[new_mdl])

    for mdl, cdata in new_cdatas.items():
        if cdata.get_seed() != use_seed:
            raise MergeError("Cohort for model {} does not have the correct "
                             "cross-validation seed!".format(mdl))

        if cdata.get_test_samples():
            raise MergeError("Cohort for model {} does not have an empty "
                             "testing sample set!".format(mdl))

    for (mdl1, chsum1), (mdl2, chsum2) in combn(new_chsums.items(), 2):
        assert chsum1['expr'] == chsum2['expr'], (
            "Inconsistent expression hashes found for cohorts in new "
            "experiments `{}` and `{}` !".format(mdl1, mdl2)
            )

        for both_lvl in ((chsum1.keys() - {'expr'})
                         & (chsum2.keys() - {'expr'})):
            assert chsum1[both_lvl] == chsum2[both_lvl], (
                "Inconsistent hashes at mutation level `{}` "
                "found for cohorts in new experiments `{}` and "
                "`{}` !".format(both_lvl, mdl1, mdl2)
                )

    # TODO: keep union of all found mutation trees
    # instead of just the first set
    if new_files:
        if cur_hash is not None:
            for new_mdl, new_chsum in new_chsums.items():
                assert new_chsum['expr'] == cur_hash['expr'], (
                    "Inconsistent expression hashes found for cohort in "
                    "new experiment `{}` !".format(new_mdl)
                    )

                for both_lvl in ((new_chsum.keys() - {'expr'})
                                 & (cur_hash.keys() - {'expr'})):
                    assert new_chsum[both_lvl] == cur_hash[both_lvl], (
                        "Inconsistent hashes at mutation "
                        "level `{}` found for cohort in new "
                        "experiment `{}` !".format(both_lvl, new_mdl)
                        )

            use_cdata = cur_cdata

        else:
            use_cdata = tuple(new_cdatas.values())[0]
            with open(cdata_file, 'wb') as f:
                pickle.dump(use_cdata, f, protocol=-1)

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

    # get list of output files from all parallelized jobs
    file_list = tuple(Path(os.path.join(args.use_dir, 'output')).glob(
        "out__cv-*_task*.p"))
    assert (len(file_list) % 40) == 0, "Missing output files detected!"

    task_count = len(file_list) // 40
    out_data = [[None for task_id in range(task_count)]
                for cv_id in range(40)]

    for out_fl in file_list:
        base_name = out_fl.stem.split('out__')[1]
        cv_id = int(base_name.split('cv-')[1].split('_')[0])
        task_id = int(base_name.split('task-')[1])
 
        with open(out_fl, 'rb') as f:
            out_data[cv_id][task_id] = pickle.load(f)

    use_clfs = set(out_dict['Clf'] for ols in out_data for out_dict in ols)
    assert len(use_clfs) == 1, ("Each experiment must be run with "
                                "exactly one classifier!")
    use_clf = tuple(use_clfs)[0]

    use_tune = set(out_dict['Clf'].tune_priors for ols in out_data
                   for out_dict in ols)
    assert len(use_tune) == 1, ("Each experiment must be run with "
                                "exactly one set of tuning priors!")

    with open(os.path.join(args.use_dir, '..', '..', "cohort-data.p"),
              'rb') as fl:
        cdata = pickle.load(fl)

    out_dfs = {k: {cis_lbl: [None for cv_id in range(40)]
                   for cis_lbl in cis_lbls}
               for k in ['Pred', 'Pars', 'Time', 'Acc']}

    for cv_id, ols in enumerate(out_data):
        for k in out_dfs:
            for cis_lbl in cis_lbls:
                out_dfs[k][cis_lbl][cv_id] = pd.concat([
                    pd.DataFrame.from_dict(
                        {mtype: out_vals[cis_lbl]
                         for mtype, out_vals in out_dict[k].items()},
                        orient='index'
                        )
                    for out_dict in ols
                    ])

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
        assert pars_df.shape[1] == (40 * len(use_clf.tune_priors)), (
            "Tuned parameter values missing for some CVs!")

    time_dfs = {cis_lbl: pd.concat(time_mats, axis=1)
                for cis_lbl, time_mats in out_dfs['Time'].items()}

    for cis_lbl, time_df in time_dfs.items():
        assert time_df.shape[1] == 80, (
            "Model fitting times missing for some CVs!")
        assert (time_df.applymap(len) == use_clf.test_count).values.all(), (
            "Model fitting times missing for some hyper-parameter values!")

    acc_dfs = {cis_lbl: pd.concat(acc_mats, axis=1)
               for cis_lbl, acc_mats in out_dfs['Acc'].items()}

    for cis_lbl, acc_df in acc_dfs.items():
        assert acc_df.shape[1] == 120, (
            "Algorithm tuning accuracies missing for some CVs!")
        assert (acc_df.applymap(len) == use_clf.test_count).values.all(), (
            "Algorithm tuning stats missing for some hyper-parameter values!")

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    for cis_lbl, pred_df in pred_dfs.items():
        assert compare_muts(pred_df.index, muts_list), (
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
            for mtype in muts_list
            }).transpose()
        for cis_lbl, pred_df in pred_dfs.items()
        }

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pred.p.gz"), 'w') as fl:
        pickle.dump(pred_dfs, fl, protocol=-1)
    with bz2.BZ2File(os.path.join(args.use_dir, "out-tune.p.gz"), 'w') as fl:
        pickle.dump([pars_df, time_df, acc_df, use_clf], fl, protocol=-1)

    cdata.update_split(test_prop=0)
    train_samps = np.array(cdata.get_train_samples())
    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in muts_list}

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pheno.p.gz"), 'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    auc_vals = pd.DataFrame({
        cis_lbl: pd.Series(dict(zip(muts_list, Parallel(
            backend='threading', n_jobs=12, pre_dispatch=12)(
                delayed(calculate_auc)(
                    pheno_dict[mtype],
                    np.vstack(pred_df.loc[
                        mtype, train_samps].values).mean(axis=1)
                    )
                for mtype in muts_list
                )
            )))
        for cis_lbl, pred_df in pred_dfs.items()
        })

    with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"), 'w') as fl:
        pickle.dump(auc_vals, fl, protocol=-1)

    random.seed(7609)
    sub_inds = [random.choices([False, True], k=len(cdata.get_samples()))
                for _ in range(500)]

    conf_dict = {
        cis_lbl: pd.DataFrame.from_records(
            tuple(zip(cycle(muts_list), Parallel(
                backend='threading', n_jobs=12, pre_dispatch=12)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype][sub_indx],
                        np.vstack(pred_df.loc[
                            mtype, train_samps[sub_indx]].values).mean(axis=1)
                        )
                    for sub_indx in sub_inds for mtype in muts_list
                    )
                ))
            ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0]
        for cis_lbl, pred_df in pred_dfs.items()
        }

    with bz2.BZ2File(os.path.join(args.use_dir, "out-conf.p.gz"), 'w') as fl:
        pickle.dump(conf_dict, fl, protocol=-1)


if __name__ == "__main__":
    main()

