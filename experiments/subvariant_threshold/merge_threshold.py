
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

from HetMan.experiments.subvariant_test.merge_test import (
    compare_muts, calculate_auc, transfer_signatures)
from HetMan.experiments.subvariant_infer.merge_infer import (
    get_cohort_subtypes)

from itertools import cycle
from itertools import combinations as combn
from functools import reduce
from operator import or_


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

    out_dfs = {k: [None for cv_id in range(40)] for k in ['Pred', 'Transfer']}
    with open(os.path.join(args.use_dir, 'setup', "cohort-data.p"),
              'rb') as fl:
        cdata = pickle.load(fl)

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

    trnsf_df = pd.concat(out_dfs['Transfer'], axis=1)
    assert (trnsf_df.columns.value_counts() == 40).all(), (
        "Inconsistent number of predicted scores across transfer cohorts!")

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    for out_df in [pred_df, trnsf_df]:
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

