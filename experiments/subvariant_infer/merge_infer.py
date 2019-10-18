
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.subvariant_infer import *
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_infer.utils import Mcomb, ExMcomb
from HetMan.experiments.subvariant_tour.merge_tour import compare_muts
from HetMan.experiments.subvariant_infer import copy_mtype
from dryadic.features.mutations import MuType

from HetMan.features.cohorts.metabric import load_metabric_samps
from HetMan.features.cohorts.metabric import (
    choose_subtypes as choose_metabric_subtypes)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import argparse
from pathlib import Path
import bz2
import dill as pickle


def get_cohort_subtypes(coh):
    if coh == 'METABRIC':
        metabric_samps = load_metabric_samps(metabric_dir)
        subt_dict = {subt: choose_metabric_subtypes(metabric_samps, subt)
                     for subt in ('nonbasal', 'luminal')}

    else:
        subt_dict = dict()

    return subt_dict


def cdict_hash(cdict):
    return hash((lvls, cdata.data_hash()) for lvls, cdata in cdict.items())


def merge_cohort_dict(out_dir, use_seed=None):
    cdict_file = os.path.join(out_dir, "cohort-dict.p")

    cur_hash = None
    if os.path.isfile(cdict_file):
        with open(cdict_file, 'rb') as fl:
            cur_cdict = pickle.load(fl)
            cur_hash = cdict_hash(cur_cdict)

    new_files = Path(out_dir).glob("cohort-dict__*__*.p")
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


def calculate_simls(pheno_dict, cur_mtype, all_vals, iso_vals, mut_list=None):
    if mut_list is None:
        mut_list = tuple(pheno_dict)

    wt_all_vals = np.concatenate(all_vals[~pheno_dict[cur_mtype]])
    cur_all_vals = np.concatenate(all_vals[pheno_dict[cur_mtype]])
    none_vals = np.concatenate(iso_vals[pheno_dict['Wild-Type']])
    cur_iso_vals = np.concatenate(iso_vals[pheno_dict[cur_mtype]])

    if not isinstance(cur_mtype, (RandomType, Mcomb, ExMcomb)):
        assert (len(wt_all_vals)
                / (~pheno_dict[cur_mtype]).sum()) in [20., 1.], cur_mtype
        assert (len(cur_all_vals)
                / pheno_dict[cur_mtype].sum()) in [20., 1.], cur_mtype

        """
        assert (len(none_vals)
                / (~pheno_dict['Wild-Type']).sum()) in [20., 1.], cur_mtype
        assert (len(cur_iso_vals)
                / pheno_dict[cur_mtype].sum()) in [20., 1.], cur_mtype
        """

    all_auc = np.greater.outer(cur_all_vals, wt_all_vals).mean()
    all_auc += np.equal.outer(cur_all_vals, wt_all_vals).mean() / 2
    iso_auc = np.greater.outer(cur_iso_vals, none_vals).mean()
    iso_auc += np.equal.outer(cur_iso_vals, none_vals).mean() / 2

    siml_dict = {cur_mtype: 1}
    cur_diff = np.subtract.outer(cur_iso_vals, none_vals).mean()
    other_mtypes = [
        mtype for mtype in set(mut_list) - {cur_mtype, 'Wild-Type'}
        if not isinstance(mtype, RandomType)
        ]

    if cur_diff != 0 and not isinstance(cur_mtype, RandomType):
        for other_mtype in other_mtypes:
            other_vals = np.concatenate(iso_vals[pheno_dict[other_mtype]])
            siml_dict[other_mtype] = np.subtract.outer(
                other_vals, none_vals).mean() / cur_diff

    else:
        siml_dict.update({other_mtype: 0 for other_mtype in other_mtypes})

    return all_auc, iso_auc, siml_dict


def compare_scores(infer_df, cdata):
    use_gene = {mtype.get_labels()[0]
                for mtype in infer_df['All'].index
                if not isinstance(mtype, (ExMcomb, Mcomb, RandomType))}

    assert len(use_gene) == 1, ("Mutations to merge are not all associated "
                                "with the same gene!")
    use_gene = tuple(use_gene)[0]

    base_lvls = 'Gene', 'Scale', 'Copy', 'Exon', 'Location', 'Protein'
    if base_lvls not in cdata.mtrees:
        cdata.add_mut_lvls(base_lvls)

    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in infer_df['All'].index}
    pheno_dict['Wild-Type'] = ~np.array(cdata.train_pheno(
        MuType(cdata.mtrees[base_lvls].allkey())))

    siml_vals = dict(zip(infer_df['All'].index, Parallel(
        backend='threading', n_jobs=12, pre_dispatch=12)(
            delayed(calculate_simls)(
                pheno_dict, cur_mtype, infer_df['All'].loc[cur_mtype].values,
                infer_df['Iso'].loc[cur_mtype].values
                )
            for cur_mtype in infer_df['All'].index
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


def transfer_signatures(trnsf_cdata, orig_cdata, mtype_list, subt_smps,
                        all_df, iso_df):
    use_muts = {mtype for mtype in mtype_list
                if not isinstance(mtype, RandomType)}

    use_gene = {mtype.get_labels()[0] for mtype in use_muts
                if not isinstance(mtype, (ExMcomb, Mcomb))}
    assert len(use_gene) == 1, ("Mutations to merge are not all associated "
                                "with the same gene!")
    use_gene = tuple(use_gene)[0]

    if 'Copy' in trnsf_cdata.muts:
        base_lvls = 'Gene', 'Scale', 'Copy', 'Exon', 'Location', 'Protein'

    else:
        base_lvls = 'Gene', 'Exon', 'Location', 'Protein'

        use_muts = {
            mut for mut in use_muts
            if ((isinstance(mut, Mcomb)
                 and not any(not (copy_mtype
                                  & mtype.subtype_list()[0][1]).is_empty()
                             for mtype in mut.mtypes))
                or (isinstance(mut, ExMcomb)
                    and (copy_mtype
                         & mut.all_mtype.subtype_list()[0][1]).is_empty())
                or (not isinstance(mut, (Mcomb, ExMcomb))
                    and (copy_mtype & mut.subtype_list()[0][1]).is_empty()))
            }

    if base_lvls not in trnsf_cdata.mtrees:
        trnsf_cdata.add_mut_lvls(base_lvls)

    sub_stat = np.array([smp in orig_cdata.get_train_samples()
                         for smp in trnsf_cdata.get_train_samples()])
    if subt_smps:
        sub_stat |= np.array([smp not in subt_smps
                              for smp in trnsf_cdata.get_train_samples()])

    pheno_dict = {mtype: np.array(trnsf_cdata.train_pheno(mtype))[~sub_stat]
                  for mtype in use_muts}

    pheno_dict['Wild-Type'] = ~np.array(trnsf_cdata.train_pheno(
        MuType(trnsf_cdata.mtrees[base_lvls].allkey())))[~sub_stat]
    use_muts = {mtype for mtype in use_muts if pheno_dict[mtype].sum() >= 20}

    siml_vals = dict(zip(use_muts, Parallel(
        backend='threading', n_jobs=12, pre_dispatch=12)(
            delayed(calculate_simls)(
                pheno_dict, cur_mtype,
                all_df.loc[cur_mtype][~sub_stat].reshape(-1, 1),
                iso_df.loc[cur_mtype][~sub_stat].reshape(-1, 1),
                mut_list=use_muts
                )
            for cur_mtype in use_muts
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

    file_list = tuple(Path(os.path.join(args.use_dir, 'output')).glob(
        "out_task-*.p"))
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
        smps: pd.concat([
            pd.DataFrame.from_dict({mtype: vals[smps]
                                    for mtype, vals in out_dict[k].items()},
                                   orient='index')
            for out_dict in out_data
            ])
        for smps in ['All', 'Iso']
        }
        for k in ['Infer', 'Pars', 'Time', 'Acc', 'Transfer']}

    for smps in ['All', 'Iso']:
        assert compare_muts(out_dfs['Infer'][smps].index, muts_list), (
            "Mutations for which predictions were made do not match the list "
            "of mutation enumerated during setup!"
            )

    for out_k in ['Infer', 'Pars', 'Time', 'Acc']:
        assert compare_muts(
            out_dfs[out_k]['All'].index, out_dfs[out_k]['Iso'].index), (
                "Mutations with inferred `{}` in naive mode do not match the "
                "mutations with scores in isolation mode!".format(out_k)
                )

    assert compare_muts(
        *[out_dfs[out_k]['All'].index
          for out_k in ['Infer', 'Pars', 'Time', 'Acc']]
        ), "Mutations across different output types do not match!"

    with bz2.BZ2File(os.path.join(args.use_dir, "out-data.p.gz"), 'w') as fl:
        pickle.dump(
            {'Infer': out_dfs['Infer'],
             'Pars': pd.concat(out_dfs['Pars'], axis=1),
             'Time': pd.concat(out_dfs['Time'], axis=1),
             'Acc': pd.concat(out_dfs['Acc'], axis=1),
             'Transfer': out_dfs['Transfer'], 'Clf': tuple(use_clfs)[0]},
            fl, protocol=-1
            )

    with open(os.path.join(args.use_dir,
                           'setup', "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)

    out_list = compare_scores(out_dfs['Infer'], cdata)
    with bz2.BZ2File(os.path.join(args.use_dir, "out-simil.p.gz"), 'w') as fl:
        pickle.dump(out_list, fl, protocol=-1)

    coh_files = Path(os.path.join(args.use_dir, 'setup')).glob(
        "*__cohort-data.p")
    coh_dict = {coh_fl.stem.split('__')[0]: coh_fl for coh_fl in coh_files}
    subt_dict = dict()

    for coh in tuple(coh_dict):
        subt_dict[coh] = None

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

            trnsf_dict[coh]['AUC'] = transfer_signatures(
                trnsf_cdata, cdata, muts_list, subt_dict[coh],
                out_dfs['Transfer']['All'][coh_k],
                out_dfs['Transfer']['Iso'][coh_k]
                )

    with bz2.BZ2File(os.path.join(args.use_dir, "out-trnsf.p.gz"), 'w') as fl:
        pickle.dump(trnsf_dict, fl, protocol=-1)


if __name__ == "__main__":
    main()

