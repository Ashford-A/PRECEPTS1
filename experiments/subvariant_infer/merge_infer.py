
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.variant_baseline.merge_tests import MergeError
from HetMan.experiments.subvariant_infer.setup_infer import Mcomb, ExMcomb
from HetMan.experiments.subvariant_infer.utils import compare_scores

import argparse
import pandas as pd
import bz2
import dill as pickle
from glob import glob


def cdict_hash(cdict):
    return hash((lvls, cdata.data_hash()) for lvls, cdata in cdict.items())


def merge_cohort_dict(out_dir, use_seed=None):
    cdict_file = os.path.join(out_dir, "cohort-dict.p")

    cur_hash = None
    if os.path.isfile(cdict_file):
        with open(cdict_file, 'rb') as fl:
            cur_cdict = pickle.load(fl)
            cur_hash = cdict_hash(cur_cdict)

    new_files = glob(os.path.join(out_dir, "cohort-dict__*__*.p"))
    new_mdls = [new_file.split("cohort-dict__")[1].split(".p")[0]
                for new_file in new_files]

    new_cdicts = {new_mdl: pickle.load(open(new_file, 'rb'))
                  for new_mdl, new_file in zip(new_mdls, new_files)}
    new_chsums = {mdl: cdict_hash(cdict) for mdl, cdict in new_cdicts.items()}

    for mdl, new_cdict in new_cdicts.items():
        for lvls, cdata in new_cdict.items():
            if cdata.get_seed() != use_seed:
                raise MergeError("Cohort for levels {} in model {} does not "
                                 "have the correct cross-validation "
                                 "seed!".format(lvls, mdl))

            if cdata.get_test_samples():
                raise MergeError("Cohort for levels {} in model {} does not "
                                 "have an empty testing sample "
                                 "set!".format(lvls, mdl))

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
            use_cdict = cur_cdict

        else:
            use_cdict = tuple(new_cdicts.values())[0]
            with open(cdict_file, 'wb') as f:
                pickle.dump(use_cdict, f)

        for new_file in new_files:
            os.remove(new_file)

    else:
        if cur_hash is None:
            raise ValueError("No cohort datasets found in {}, has an "
                             "experiment with these parameters been run to "
                             "completion yet?".format(out_dir))

        else:
            use_cdict = cur_cdict

    return use_cdict


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

    test = {mut: vals['All'] for mut, vals in out_data[0]['Tune'].items()}

    out_dfs = {k: {
        smps: pd.concat([
            pd.DataFrame.from_records({mut: vals[smps]
                                       for mut, vals in out_dict[k].items()})
            for out_dict in out_data
            ], axis=1, sort=True).transpose()
        for smps in ['All', 'Iso']
        }
        for k in ['Infer', 'Tune']}

    assert (set(out_dfs['Infer']['All'].index)
            == set(out_dfs['Infer']['Iso'].index)), (
                "Mutations with inferred scores in naive mode do not match "
                "the mutations with scores in isolation mode!"
                )

    assert (set(out_dfs['Tune']['All'].index)
            == set(out_dfs['Tune']['Iso'].index)), (
                "Mutations with tuned hyper-parameters in naive mode do not "
                "match the mutations tuned in isolation mode!"
                )

    assert (set(out_dfs['Infer']['All'].index)
            == set(out_dfs['Tune']['Iso'].index)), (
                "Mutations with tuned hyper-parameters do not match the "
                "mutations with inferred scores in isolation mode!"
                )

    assert out_dfs['Infer']['All'].shape[0] == len(muts_list), (
        "Inferred scores missing for some tested mutations!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-data.p.gz"), 'w') as fl:
        pickle.dump({'Infer': out_dfs['Infer'],
                     'Tune': pd.concat(out_dfs['Tune'], axis=1),
                     'Clf': tuple(use_clfs)[0]}, fl, protocol=-1)

    with open(os.path.join(args.use_dir,
                           'setup', "cohort-dict.p"), 'rb') as fl:
        cdata_dict = pickle.load(fl)

    with bz2.BZ2File(os.path.join(args.use_dir, "out-simil.p.gz"), 'w') as fl:
        pickle.dump(compare_scores(
            out_dfs['Infer']['Iso'],
            sorted(cdata_dict['Exon__Location__Protein'].get_train_samples()),
            {lvls: cdata.mtree for lvls, cdata in cdata_dict.items()}
            ), fl, protocol=-1)


if __name__ == "__main__":
    main()

