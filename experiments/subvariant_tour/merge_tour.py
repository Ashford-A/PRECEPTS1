
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_tour import *
import argparse
import bz2
import dill as pickle

from glob import glob
import pandas as pd
from itertools import product
from itertools import combinations as combn


def merge_cohort_data(out_dir, mut_lvls, use_seed=None):
    cdata_file = os.path.join(out_dir, "cohort-data__{}.p".format(mut_lvls))

    if os.path.isfile(cdata_file):
        with open(cdata_file, 'rb') as fl:
            cur_cdata = pickle.load(fl)
            cur_hash = cur_cdata.data_hash()
            cur_hash = tuple(cur_hash[0]), cur_hash[1]

    else:
        cur_hash = None

    #TODO: handle case where input `mut_lvls` is malformed
    # eg. Domain_SMART instead of Domain_SMART__Form_base
    new_files = glob(os.path.join(out_dir,
                                  "cohort-data__{}__*.p".format(mut_lvls)))
    new_mdls = [
        new_file.split("cohort-data__{}__".format(mut_lvls))[1].split(".p")[0]
        for new_file in new_files
        ]

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

    file_list = glob(os.path.join(args.use_dir, 'output', "out_task-*.p"))
    base_names = [os.path.basename(fl).split('out_')[1] for fl in file_list]
    task_ids = [int(nm.split('task-')[1].split('.p')[0]) for nm in base_names]

    muts_list = pickle.load(open(os.path.join(
        args.use_dir, 'setup', "muts-list.p"), 'rb'))
    out_data = [pickle.load(open(fl, 'rb')) for fl in file_list]

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
        for k in ['Infer', 'Tune']}

    for cis_lbl in cis_lbls:
        assert sorted(out_dfs['Infer'][cis_lbl].index) == sorted(muts_list), (
            "Mutations for which predictions were made do not match the list "
            "of mutations enumerated during setup!"
            )

    for cis_lbl1, cis_lbl2 in combn(cis_lbls, 2):
        assert (sorted(out_dfs['Infer'][cis_lbl1].index)
                    == sorted(out_dfs['Infer'][cis_lbl2].index)), (
            "Mutations tested using cis-exclusion strategy {} do not match "
            "those tested using strategy {}!".format(cis_lbl1, cis_lbl2)
            )

    for cis_lbl1, cis_lbl2 in product(cis_lbls, repeat=2):
        assert (sorted(out_dfs['Infer'][cis_lbl1].index)
                    == sorted(out_dfs['Tune'][cis_lbl2].index)), (
            "Mutations with predicted scores do not match those for which "
            "tuned hyper-parameter values are available!"
            )

    with bz2.BZ2File(os.path.join(args.use_dir, "out-data.p.gz"), 'w') as fl:
        pickle.dump({'Infer': out_dfs['Infer'],
                     'Tune': pd.concat(out_dfs['Tune'], axis=1),
                     'Clf': tuple(use_clfs)[0]}, fl, protocol=-1)


if __name__ == "__main__":
    main()

