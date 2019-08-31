
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.variant_baseline.merge_tests import MergeError
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_infer.utils import Mcomb, ExMcomb
from dryadic.features.mutations import MuType

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

import argparse
from glob import glob
import bz2
import dill as pickle


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


def calculate_simls(pheno_dict, all_vals, iso_vals, cur_mtype):
    print(cur_mtype)
    wt_all_vals = np.concatenate(all_vals.iloc[
        0, ~pheno_dict[cur_mtype]].values)
    cur_all_vals = np.concatenate(all_vals.iloc[
        0, pheno_dict[cur_mtype]].values)

    none_vals = np.concatenate(iso_vals.iloc[
        0, pheno_dict['Wild-Type']].values)
    cur_iso_vals = np.concatenate(iso_vals.iloc[
        0, pheno_dict[cur_mtype]].values)

    if (not isinstance(cur_mtype, RandomType)
            or isinstance(cur_mtype.size_dist, int)):
        assert (((len(wt_all_vals), len(cur_all_vals))
                 / np.bincount(pheno_dict[cur_mtype]))
                == np.array([20, 20])).all()

        assert (((len(none_vals), len(cur_iso_vals))
                 / np.array([np.sum(pheno_dict['Wild-Type']),
                             np.sum(pheno_dict[cur_mtype])]))
                == np.array([20, 20])).all()

    all_auc = np.greater.outer(cur_all_vals, wt_all_vals).mean()
    all_auc += np.equal.outer(cur_all_vals, wt_all_vals).mean() / 2
    iso_auc = np.greater.outer(cur_iso_vals, none_vals).mean()
    iso_auc += np.equal.outer(cur_iso_vals, none_vals).mean() / 2

    siml_dict = {cur_mtype: 1}
    cur_diff = np.subtract.outer(cur_iso_vals, none_vals).mean()
    other_mtypes = [
        mtype for mtype in set(pheno_dict) - {cur_mtype, 'Wild-Type'}
        if not isinstance(mtype, RandomType)
        ]

    if cur_diff != 0 and not isinstance(cur_mtype, RandomType):
        for other_mtype in other_mtypes:
            other_vals = np.concatenate(iso_vals.iloc[
                0, pheno_dict[other_mtype]].values)

            siml_dict[other_mtype] = np.subtract.outer(
                other_vals, none_vals).mean() / cur_diff

    else:
        siml_dict.update({other_mtype: 0 for other_mtype in other_mtypes})

    return all_auc, iso_auc, siml_dict


def compare_scores(infer_df, samps, muts_dict):
    use_gene = {mtype.subtype_list()[0][0]
                for _, mtype in infer_df['All'].index
                if not isinstance(mtype, (ExMcomb, Mcomb, RandomType))}

    assert len(use_gene) == 1, ("Mutations to merge are not all associated "
                                "with the same gene!")
    use_gene = tuple(use_gene)[0]
    base_muts = tuple(muts_dict.values())[0]
    all_mtype = MuType({('Gene', use_gene): base_muts[use_gene].allkey()})

    pheno_dict = {
        mtype: np.array(muts_dict[lvls].status(samps, mtype))
        if lvls in muts_dict else np.array(base_muts.status(samps, mtype))
        for lvls, mtype in infer_df['All'].index
        }
    pheno_dict['Wild-Type'] = ~np.array(base_muts.status(samps, all_mtype))

    siml_vals = dict(zip(infer_df['All'].index, Parallel(
        backend='threading', n_jobs=12, pre_dispatch=12)(
            delayed(calculate_simls)(
                pheno_dict, infer_df['All'].loc[[(lvls, cur_mtype)]],
                infer_df['Iso'].loc[[(lvls, cur_mtype)]], cur_mtype
                )
            for lvls, cur_mtype in infer_df['All'].index
            )
        ))

    auc_df = pd.DataFrame(
        {mut: [all_auc, iso_auc]
         for mut, (all_auc, iso_auc, _) in siml_vals.items()},
        index=['All', 'Iso']
        ).transpose()

    simil_df = pd.DataFrame(
        {mut: siml_dict
         for mut, (_, _, siml_dict) in siml_vals.items()}
        ).transpose()

    return pheno_dict, auc_df, simil_df


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
        smps: pd.concat([
            pd.DataFrame.from_records({mut: vals[smps]
                                       for mut, vals in out_dict[k].items()})
            for out_dict in out_data
            ], axis=1, sort=True).transpose()
        for smps in ['All', 'Iso']
        }
        for k in ['Infer', 'Tune']}
    print('assert!')

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

    out_list = compare_scores(
        out_dfs['Infer'],
        sorted(cdata_dict['Exon__Location__Protein'].get_train_samples()),
        {lvls: cdata.mtree for lvls, cdata in cdata_dict.items()}
        )

    with bz2.BZ2File(os.path.join(args.use_dir, "out-simil.p.gz"), 'w') as fl:
        pickle.dump(out_list, fl, protocol=-1)


if __name__ == "__main__":
    main()

