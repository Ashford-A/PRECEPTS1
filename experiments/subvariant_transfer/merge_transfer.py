
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.variant_baseline.merge_tests import MergeError
from dryadic.features.mutations.trees import MuTree

import argparse
import pandas as pd
from re import sub as gsub
import dill as pickle
from glob import glob


def get_mtree_newick(mtree):
    """Get the Newick tree format representation of this MuTree."""
    newick_str = ''
 
    for nm, mut in sorted(mtree, key=lambda x: x[0]):
        if isinstance(mut, MuTree):
            newick_str += '(' + gsub(',$', '', get_mtree_newick(mut)) + ')'
 
        if nm == ".":
            newick_str += '{*none*},'
        else:
            newick_str += '{' + nm + '},'

    if mtree.depth == 0:
        newick_str = gsub(',$', '', newick_str) + ';'

    return newick_str


def cdata_hash(cdata):
    expr_hash = tuple(dict(cdata.omic_data.sum().round(5)).items())
    mut_str = get_mtree_newick(cdata.train_mut)

    samp_hash = tuple(sorted((cohort, mtype)
                             for cohort, mtypes in cdata.cohort_samps.items()
                             for mtype in mtypes))

    return (expr_hash, tuple(mut_str.count(k) for k in sorted(set(mut_str))),
            samp_hash)


def merge_cohort_data(out_dir, use_lvl='Location__Protein'):
    new_files = glob(os.path.join(
        out_dir, "cohort-data__{}_*.p".format(use_lvl)))

    cur_files = glob(os.path.join(out_dir, "cohort-data_v*.p"))
    cur_lvls = [cur_file.split("cohort-data_v")[1].split(".p")[0]
                for cur_file in cur_files]

    if not new_files and use_lvl not in cur_lvls:
        raise ValueError(
            "No cohort datasets found for levels {} in {}, has an experiment "
            "using this combination been run to completion yet?".format(
                use_lvl, out_dir)
            )

    new_mdls = [
        new_file.split("cohort-data__{}_".format(use_lvl))[1].split(".p")[0]
        for new_file in new_files
        ]

    cur_cdatas = {cur_lvl: pickle.load(open(cur_file, 'rb'))
                  for cur_lvl, cur_file in zip(cur_lvls, cur_files)}
    cur_chsums = {lvl: cdata_hash(cdata) for lvl, cdata in cur_cdatas.items()}

    assert len(
        set((expr_hsh, samp_hsh)
            for expr_hsh, _, samp_hsh in cur_chsums.values())
        ) <= 1, ("Inconsistent cohort hashes found for cached experiments "
                 "in {} !".format(out_dir))

    new_cdatas = {new_mdl: pickle.load(open(new_file, 'rb'))
                  for new_mdl, new_file in zip(new_mdls, new_files)}
    new_chsums = {mdl: cdata_hash(cdata) for mdl, cdata in new_cdatas.items()}

    for mdl, cdata in new_cdatas.items():
        if cdata.cv_seed != 9078:
            raise MergeError("Cohort for model {} does not have the correct "
                             "cross-validation seed!".format(mdl))

        if cdata.test_samps is not None:
            raise MergeError("Cohort for model {} does not have an empty "
                             "testing sample set!".format(mdl))

    assert len(set(new_chsums.values())) <= 1, (
        "Inconsistent cohort hashes found for new experiments using levels "
        "{} in {} !".format(use_lvl, out_dir)
        )

    if use_lvl in cur_cdatas:
        use_cdata = cur_cdatas[use_lvl]

        if new_files:
            assert tuple(new_chsums.values())[0] == cur_chsums[use_lvl], (
                "Cohort hash for new experiment using levels {} in {} does "
                "not match hash for cached cohorts!".format(use_lvl, out_dir)
                )

    else:
        use_cdata = tuple(new_cdatas.values())[0]

        out_file = os.path.join(out_dir, "cohort-data_v{}.p".format(use_lvl))
        with open(out_file, 'wb') as fl:
            pickle.dump(use_cdata, fl)

    for new_file in new_files:
        os.remove(new_file)

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

    use_clfs = set(out_dict['Clf'].__class__ for out_dict in out_data)
    assert len(use_clfs) == 1, ("Each experiment must be run with "
                                "exactly one classifier!")

    use_tune = set(out_dict['Clf'].tune_priors for out_dict in out_data)
    assert len(use_tune) == 1, ("Each experiment must be run with "
                                "exactly one set of tuning priors!")

    out_dfs = {k: {
        smps: pd.concat([
            pd.DataFrame.from_dict({mtype: vals[smps]
                                    for mtype, vals in out_dict[k].items()},
                                   orient='index')
            for out_dict in out_data
            ])
        for smps in ['All', 'Iso']
        }
        for k in ['Infer', 'Tune']}

    assert (set(out_dfs['Infer']['All'].index)
            == set(out_dfs['Infer']['Iso'].index)), (
                "Mutations tested in naive mode do not match the mutations "
                "tested in isolation mode!"
                )

    assert out_dfs['Infer']['All'].shape[0] == len(muts_list), (
        "Inferred scores missing for some tested mutations!")

    pickle.dump({'Infer': out_dfs['Infer'],
                 'Tune': pd.concat(out_dfs['Tune'], axis=1),
                 'Clf': tuple(use_clfs)[0]},
                open(os.path.join(args.use_dir, "out-data.p"), 'wb'))


if __name__ == "__main__":
    main()

