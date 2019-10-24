
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_tour.merge_tour import (
    compare_muts, calculate_auc)
from HetMan.experiments.subvariant_infer.merge_infer import (
    get_cohort_subtypes)

import argparse
from pathlib import Path
import bz2
import dill as pickle

import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed


def transfer_signatures(trnsf_cdata, orig_cdata,
                        mtype_list, subt_smps, infer_df):
    sub_stat = np.array([smp in orig_cdata.get_train_samples()
                         for smp in trnsf_cdata.get_train_samples()])

    if subt_smps:
        sub_stat |= np.array([smp not in subt_smps
                              for smp in trnsf_cdata.get_train_samples()])

    pheno_dict = {mtype: np.array(trnsf_cdata.train_pheno(mtype))[~sub_stat]
                  for mtype in mtype_list}
    use_muts = {mtype for mtype in mtype_list
                if pheno_dict[mtype].sum() >= 20}

    return dict(zip(use_muts, Parallel(
        backend='threading', n_jobs=12, pre_dispatch=12)(
            delayed(calculate_auc)(
                pheno_dict,
                pd.DataFrame(infer_df.loc[mtype][~sub_stat].reshape(-1, 1)),
                mtype
                )
            for mtype in use_muts
            )
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('use_dir', type=str, default=base_dir)
    args = parser.parse_args()

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    file_list = tuple(Path(
        os.path.join(args.use_dir, 'output')).glob("out_task-*.p"))
    base_names = [fl.stem.split('out_')[1] for fl in file_list]
    task_ids = [int(nm.split('task-')[1].split('.p')[0]) for nm in base_names]

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

    out_dfs = {k: pd.concat([pd.DataFrame.from_dict(out_dict[k],
                                                    orient='index')
                             for out_dict in out_data], sort=True)
               for k in ['Infer', 'Transfer']}

    assert compare_muts(out_dfs['Infer'].index, out_dfs['Transfer'].index,
                        muts_list), ("Mutations for which predictions were "
                                     "made do not match the list of "
                                     "mutations enumerated during setup!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-data.p.gz"), 'w') as fl:
        pickle.dump(
            {'Infer': out_dfs['Infer'], 'Transfer': out_dfs['Transfer'],
             'Clf': tuple(use_clfs)[0]},
            fl, protocol=-1
            )

    with open(os.path.join(args.use_dir, 'setup',
                           "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)

    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in out_dfs['Infer'].index}

    for (mtype1, phn1), (mtype2, phn2) in product(pheno_dict.items(),
                                                  repeat=2):
        if (mtype1.base_mtype == mtype2.base_mtype
                and mtype1.annot == mtype2.annot):
            phn_intx = phn1 & phn2
            assert (phn_intx == phn1).all() or (phn_intx == phn2).all()

            if mtype1.min_val > mtype2.min_val:
                assert phn1.sum() < phn2.sum()

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pheno.p.gz"), 'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    auc_df = dict(zip(out_dfs['Infer'].index, Parallel(
        backend='threading', n_jobs=12, pre_dispatch=12)(
            delayed(calculate_auc)(pheno_dict, infer_vals, mtype)
            for mtype, infer_vals in out_dfs['Infer'].iterrows()
            )
        ))

    with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"), 'w') as fl:
        pickle.dump(auc_df, fl, protocol=-1)

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
        if trnsf_cdata.mtrees['Gene', 'Scale'] != dict():
            if subt_dict[coh] is None:
                coh_k = coh
            else:
                coh_k = coh.split('_')[0]

            trnsf_dict[coh]['AUC'] = transfer_signatures(
                trnsf_cdata, cdata, muts_list, subt_dict[coh],
                out_dfs['Transfer'][coh_k],
                )

    with bz2.BZ2File(os.path.join(args.use_dir, "out-trnsf.p.gz"), 'w') as fl:
        pickle.dump(trnsf_dict, fl, protocol=-1)


if __name__ == "__main__":
    main()

