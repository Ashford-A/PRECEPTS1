
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

import argparse
from glob import glob
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random

from HetMan.experiments.subvariant_tour import cis_lbls
from HetMan.experiments.subvariant_tour.utils import RandomType
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


def compare_muts(*muts_lists):
    """
    return len(set(tuple(sorted([mut for mut in muts_list
                                 if not isinstance(mut, RandomType)]))
                   for muts_list in muts_lists)) == 1
    """
    return len(set(tuple(sorted(muts_list))
                   for muts_list in muts_lists)) == 1


def calculate_auc(pheno_dict, infer_vals, mtype, sub_indx=None):
    if sub_indx is None:
        mut_phn = pheno_dict[mtype]
        wt_phn = ~pheno_dict[mtype]
    else:
        mut_phn = pheno_dict[mtype] & sub_indx
        wt_phn = ~pheno_dict[mtype] & sub_indx

    auc_val = np.greater.outer(
        np.concatenate(infer_vals.values[mut_phn]),
        np.concatenate(infer_vals.values[wt_phn])
        ).mean()
    
    auc_val += 0.5 * np.equal.outer(
        np.concatenate(infer_vals.values[mut_phn]),
        np.concatenate(infer_vals.values[wt_phn])
        ).mean()

    return auc_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('use_dir', type=str, default=base_dir)
    args = parser.parse_args()

    file_list = glob(os.path.join(args.use_dir, 'output', "out_task-*.p"))
    base_names = [os.path.basename(fl).split('out_')[1] for fl in file_list]
    task_ids = [int(nm.split('task-')[1].split('.p')[0]) for nm in base_names]

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    out_data = [None for _ in file_list]
    for i, fl in enumerate(file_list):
        with open(fl, 'rb') as f:
            out_data[i] = pickle.load(f)

    use_clfs = set(out_dict['Clf'] for out_dict in out_data)
    assert len(use_clfs) == 1, ("Each experiment must be run with "
                                "exactly one classifier!")

    use_tune = set(out_dict['Clf'].tune_priors for out_dict in out_data)
    assert len(use_tune) == 1, ("Each experiment must be run with "
                                "exactly one set of tuning priors!")


    out_dfs = {k: {
        cis_lbl: pd.concat([
            pd.DataFrame.from_dict({mtype: vals[cis_lbl]
                                    for mtype, vals in out_dict[k].items()},
                                   orient='index')
            for out_dict in out_data
            ])
        for cis_lbl in cis_lbls
        }
        for k in ['Infer', 'Pars', 'Time', 'Acc']}

    for cis_lbl in cis_lbls:
        assert compare_muts(out_dfs['Infer'][cis_lbl].index, muts_list), (
            "Mutations for which predictions were made do not match the list "
            "of mutations enumerated during setup!"
            )

    for cis_lbl1, cis_lbl2 in combn(cis_lbls, 2):
        assert compare_muts(
            out_dfs['Infer'][cis_lbl1].index,
            out_dfs['Infer'][cis_lbl2].index,
            out_dfs['Time'][cis_lbl1].index,
            out_dfs['Time'][cis_lbl2].index
            ), ("Mutations tested using cis-exclusion strategy {} do "
                "not match those tested using strategy {}!".format(
                    cis_lbl1, cis_lbl2))

    for cis_lbl1, cis_lbl2 in product(cis_lbls, repeat=2):
        assert compare_muts(
            out_dfs['Infer'][cis_lbl1].index, out_dfs['Pars'][cis_lbl2].index,
            out_dfs['Time'][cis_lbl1].index, out_dfs['Acc'][cis_lbl2].index
            ), ("Mutations with predicted scores do not match those for "
                "which tuned hyper-parameter values are available!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-data.p.gz"), 'w') as fl:
        pickle.dump({'Infer': out_dfs['Infer'],
                     'Pars': pd.concat(out_dfs['Pars'], axis=1),
                     'Time': pd.concat(out_dfs['Time'], axis=1),
                     'Acc': pd.concat(out_dfs['Acc'], axis=1),
                     'Clf': tuple(use_clfs)[0]}, fl, protocol=-1)

    with open(os.path.join(args.use_dir, '..', '..',
                           "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)

    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in tuple(out_dfs['Infer'].values())[0].index}
    with bz2.BZ2File(os.path.join(args.use_dir, "out-pheno.p.gz"), 'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    auc_df = pd.DataFrame({
        cis_lbl: dict(zip(infer_df.index, Parallel(
            backend='threading', n_jobs=12, pre_dispatch=12)(
                delayed(calculate_auc)(pheno_dict, infer_vals, mtype)
                for mtype, infer_vals in infer_df.iterrows()
                )
            ))
        for cis_lbl, infer_df in out_dfs['Infer'].items()
        })

    with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"), 'w') as fl:
        pickle.dump(auc_df, fl, protocol=-1)

    random.seed(7609)
    sub_inds = [random.choices([False, True], k=len(cdata.get_samples()))
                for _ in range(100)]

    conf_df = {
        cis_lbl: pd.DataFrame.from_records(
            tuple(zip(cycle(infer_df.index), Parallel(
                backend='threading', n_jobs=12, pre_dispatch=12)(
                    delayed(calculate_auc)(pheno_dict, infer_vals, mtype,
                                           sub_indx)
                    for sub_indx in sub_inds
                    for mtype, infer_vals in infer_df.iterrows()
                    )
                ))
            ).pivot_table(index=0, values=1, aggfunc=list)
        for cis_lbl, infer_df in out_dfs['Infer'].items()
        }

    with bz2.BZ2File(os.path.join(args.use_dir, "out-conf.p.gz"), 'w') as fl:
        pickle.dump(conf_df, fl, protocol=-1)


if __name__ == "__main__":
    main()

