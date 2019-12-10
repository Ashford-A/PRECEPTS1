
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

from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_tour.merge_tour import compare_muts
from HetMan.experiments.subvariant_infer.merge_infer import (
    get_cohort_subtypes)

from itertools import cycle
from itertools import combinations as combn
from functools import reduce
from operator import or_


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

    new_files = tuple(Path(out_dir).glob("cohort-data__*.p"))
    new_mdls = [new_file.stem.split("cohort-data__")[1].split(".p")[0]
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

        use_cdata.mtrees = dict(reduce(
            or_, [set(cdata.mtrees.items())
                  for cdata in [use_cdata] + list(new_cdatas.values())]
            ))

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


def calculate_auc(phn_vec, pred_mat):
    if phn_vec.all() or not phn_vec.any():
        auc_val = 0.5

    else:
        auc_val = np.greater.outer(pred_mat[phn_vec],
                                   pred_mat[~phn_vec]).mean()
        auc_val += 0.5 * np.equal.outer(pred_mat[phn_vec],
                                        pred_mat[~phn_vec]).mean()

    return auc_val


def transfer_signatures(trnsf_cdata, orig_cdata,
                        pred_df, mtype_list, subt_smps):
    use_muts = {mtype for mtype in mtype_list
                if not isinstance(mtype, RandomType)}

    base_lvls = 'Gene', 'Scale', 'Copy', 'Exon', 'Location', 'Protein'
    if base_lvls not in trnsf_cdata.mtrees:
        trnsf_cdata.add_mut_lvls(base_lvls)

    sub_stat = np.array([smp in orig_cdata.get_train_samples()
                         for smp in trnsf_cdata.get_train_samples()])
    if subt_smps:
        sub_stat |= np.array([smp not in subt_smps
                              for smp in trnsf_cdata.get_train_samples()])

    pheno_dict = {mtype: np.array(trnsf_cdata.train_pheno(mtype))[~sub_stat]
                  for mtype in use_muts}
    use_muts = {mtype for mtype in use_muts if pheno_dict[mtype].sum() >= 20}
    auc_dict = dict()

    if use_muts:
        auc_dict = {
            'all': pd.Series(dict(zip(use_muts, Parallel(
                backend='threading', n_jobs=12, pre_dispatch=12)(
                    delayed(calculate_auc)(pheno_dict[mtype],
                                           pred_df.loc[mtype].T[~sub_stat])
                    for mtype in use_muts
                    )
                ))),

            'CV': pd.DataFrame.from_records(
                tuple(zip(cycle(use_muts), Parallel(
                    backend='threading', n_jobs=12, pre_dispatch=12)(
                        delayed(calculate_auc)(
                            pheno_dict[mtype],
                            pred_df.loc[mtype].T[~sub_stat, cv_id]
                            )
                        for cv_id in range(40) for mtype in use_muts
                        )
                    ))
                ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0],

            'mean': pd.Series(dict(zip(use_muts, Parallel(
                backend='threading', n_jobs=12, pre_dispatch=12)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype],
                        pred_df.loc[mtype].T[~sub_stat].mean(axis=1)
                        )
                    for mtype in use_muts
                    )
                )))
            }

        auc_dict['CV'].name = None
        auc_dict['CV'].index.name = None

    return pheno_dict, auc_dict


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

    with open(os.path.join(args.use_dir, 'setup', "cohort-data.p"),
              'rb') as fl:
        cdata = pickle.load(fl)

    out_dfs = {k: [None for cv_id in range(40)]
               for k in ['Pred', 'Pars', 'Time', 'Acc', 'Transfer']}

    for cv_id, ols in enumerate(out_data):
        for k in out_dfs:
            out_dfs[k][cv_id] = pd.concat([
                pd.DataFrame.from_dict(out_dict[k], orient='index')
                for out_dict in ols
                ])

        cdata_samps = sorted(cdata.get_samples())
        random.seed((cv_id // 4) * 7712 + 13)
        random.shuffle(cdata_samps)

        cdata.update_split(9073 + 97 * cv_id,
                           test_samps=cdata_samps[(cv_id % 4)::4])
        test_samps = cdata.get_test_samples()
        out_dfs['Pred'][cv_id].columns = test_samps

    pred_df = pd.concat(out_dfs['Pred'], axis=1)
    assert all(smp in pred_df.columns for smp in cdata.get_samples()), (
        "Missing mutation scores for some samples in the cohort!")
    assert (pred_df.columns.value_counts() == 10).all(), (
        "Inconsistent number of CV scores across cohort samples!")

    pars_df = pd.concat(out_dfs['Pars'], axis=1)
    assert pars_df.shape[1] == (40 * len(use_clf.tune_priors)), (
        "Tuned parameter values missing for some CVs!")

    time_df = pd.concat(out_dfs['Time'], axis=1)
    assert time_df.shape[1] == 80, (
        "Algorithm fitting times missing for some CVs!")
    assert (time_df.applymap(len) == use_clf.test_count).values.all(), (
        "Algorithm fitting times missing for some hyper-parameter values!")

    acc_df = pd.concat(out_dfs['Acc'], axis=1)
    assert acc_df.shape[1] == 120, (
        "Algorithm tuning accuracies missing for some CVs!")
    assert (acc_df.applymap(len) == use_clf.test_count).values.all(), (
        "Algorithm tuning stats missing for some hyper-parameter values!")

    trnsf_df = pd.concat(out_dfs['Transfer'], axis=1)
    assert (trnsf_df.columns.value_counts() == 40).all(), (
        "Inconsistent number of predicted scores across transfer cohorts!")

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    for out_df in [pred_df, pars_df, time_df, acc_df, trnsf_df]:
        assert compare_muts(out_df.index, muts_list), (
            "Mutations for which predictions were made do not match the list "
            "of mutations enumerated during setup!"
            )

    pred_df = pd.DataFrame({
        mtype: pred_df.loc[mtype].groupby(level=0).apply(lambda x: x.values)
         for mtype in muts_list
        }).transpose()

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pred.p.gz"), 'w') as fl:
        pickle.dump(pred_df, fl, protocol=-1)
    with bz2.BZ2File(os.path.join(args.use_dir, "out-tune.p.gz"), 'w') as fl:
        pickle.dump([pars_df, time_df, acc_df, use_clf], fl, protocol=-1)

    cdata.update_split(test_prop=0)
    train_samps = np.array(cdata.get_train_samples())
    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in muts_list}

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pheno.p.gz"), 'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    # calculates AUCs for prediction tasks using scores from all
    # cross-validations concatenated together...
    auc_dict = {
        'all': pd.Series(dict(zip(muts_list, Parallel(
            backend='threading', n_jobs=12, pre_dispatch=12)(
                delayed(calculate_auc)(
                    pheno_dict[mtype],
                    np.vstack(pred_df.loc[mtype, train_samps].values)
                    )
                for mtype in muts_list
                )
            ))),

        # ...and for each cross-validation run considered separately...
        'CV': pd.DataFrame.from_records(
            tuple(zip(cycle(muts_list), Parallel(
                backend='threading', n_jobs=12, pre_dispatch=12)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype],
                        np.vstack(pred_df.loc[
                            mtype, train_samps].values)[:, cv_id]
                        )
                    for cv_id in range(10) for mtype in muts_list
                    )
                ))
            ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0],

        # ...and finally using the average of predicted scores for each
        # sample across CV runs
        'mean': pd.Series(dict(zip(muts_list, Parallel(
            backend='threading', n_jobs=12, pre_dispatch=12)(
                delayed(calculate_auc)(
                    pheno_dict[mtype],
                    np.vstack(pred_df.loc[
                        mtype, train_samps].values).mean(axis=1)
                    )
                for mtype in muts_list
                )
            )))
        }

    auc_dict['CV'].name = None
    auc_dict['CV'].index.name = None
    with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"), 'w') as fl:
        pickle.dump(auc_dict, fl, protocol=-1)

    random.seed(7609)
    sub_inds = [random.choices([False, True], k=len(cdata.get_samples()))
                for _ in range(500)]

    conf_list = {
        'all': pd.DataFrame.from_records(
            tuple(zip(cycle(muts_list), Parallel(
                backend='threading', n_jobs=12, pre_dispatch=12)(
                    delayed(calculate_auc)(
                        pheno_dict[mtype][sub_indx],
                        np.vstack(pred_df.loc[
                            mtype, train_samps[sub_indx]].values)
                        )
                    for sub_indx in sub_inds for mtype in muts_list
                    )
                ))
            ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0],

        'mean': pd.DataFrame.from_records(
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
        }

    conf_list['all'].name = None
    conf_list['mean'].name = None
    conf_list['all'].index.name = None
    conf_list['mean'].index.name = None

    with bz2.BZ2File(os.path.join(args.use_dir, "out-conf.p.gz"), 'w') as fl:
        pickle.dump(conf_list, fl, protocol=-1)

    trnsf_df = pd.DataFrame(
        {coh: {mtype: np.vstack(vals) for mtype, vals in trnsf_mat.iterrows()}
         for coh, trnsf_mat in trnsf_df.groupby(level=0, axis=1)}
        )

    coh_files = Path(os.path.join(args.use_dir, 'setup')).glob(
        "cohort-data__*.p")
    coh_dict = {coh_fl.stem.split('__')[1]: coh_fl for coh_fl in coh_files}
    subt_dict = dict()

    for coh in tuple(coh_dict):
        subt_dict[coh] = None

        if coh != cdata.cohort:
            coh_subtypes = get_cohort_subtypes(coh)
            for subt, sub_smps in coh_subtypes.items():
                coh_dict['_'.join([coh, subt])] = coh_dict[coh]
                subt_dict['_'.join([coh, subt])] = sub_smps

    trnsf_dict = {coh: {'Samps': None, 'AUC': None} for coh in coh_dict}
    for coh, coh_fl in coh_dict.items():
        with open(coh_fl, 'rb') as f:
            trnsf_cdata = pickle.load(f)

        trnsf_dict[coh]['Samps'] = trnsf_cdata.get_train_samples()
        if any(mtree != dict() for mtree in trnsf_cdata.mtrees.values()):
            if subt_dict[coh] is None:
                coh_k = coh
            else:
                coh_k = coh.split('_')[0]

            trnsf_dict[coh]['Pheno'], trnsf_dict[coh]['AUC'] = (
                transfer_signatures(trnsf_cdata, cdata, trnsf_df[coh_k],
                                    muts_list, subt_dict[coh])
                )

    trnsf_vals = {coh: trnsf_mat.apply(lambda vals: np.mean(vals, axis=0))
                  for coh, trnsf_mat in trnsf_df.items()}
    with bz2.BZ2File(os.path.join(args.use_dir, "trnsf-vals.p.gz"),
                     'w') as fl:
        pickle.dump(trnsf_vals, fl, protocol=-1)

    with bz2.BZ2File(os.path.join(args.use_dir, "out-trnsf.p.gz"), 'w') as fl:
        pickle.dump(trnsf_dict, fl, protocol=-1)


if __name__ == "__main__":
    main()

