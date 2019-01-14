
from dryadic.features.mutations import MuType
from .setup_isolate import ExMcomb

import numpy as np
import pandas as pd
from functools import reduce
from operator import or_, and_


def compare_scores(iso_df, cdata, get_similarities=True,
                   muts=None, all_mtype=None):
    simil_df = pd.DataFrame(
        0.0, index=iso_df.index, columns=iso_df.index, dtype=np.float)

    pheno_dict = {mtypes: None for mtypes in iso_df.index}
    auc_list = pd.Series(index=iso_df.index, dtype=np.float)

    if muts is None:
        muts = cdata.train_mut
    if all_mtype is None:
        all_mtype = MuType(muts.allkey())

    samps = sorted(cdata.train_samps)
    all_pheno = np.array(muts.status(samps, all_mtype))

    pheno_dict = {mcomb: np.array(muts.status(samps, mcomb))
                  for mcomb in iso_df.index}
    pheno_dict['Wild-Type'] = ~all_pheno

    for cur_mcomb, iso_vals in iso_df.iterrows():
        simil_df.loc[cur_mcomb, cur_mcomb] = 1.0

        none_vals = np.concatenate(iso_vals[~all_pheno].values)
        cur_vals = np.concatenate(iso_vals[pheno_dict[cur_mcomb]].values)
        auc_list[cur_mcomb] = np.greater.outer(cur_vals, none_vals).mean()
        auc_list[cur_mcomb] += np.equal.outer(cur_vals, none_vals).mean() / 2

        if get_similarities:
            cur_diff = np.subtract.outer(cur_vals, none_vals).mean()

            if cur_diff != 0:
                for other_mcomb in set(iso_df.index) - {cur_mcomb}:

                    other_vals = np.concatenate(
                        iso_vals[pheno_dict[other_mcomb]].values)
                    other_diff = np.subtract.outer(
                        other_vals, none_vals).mean()

                    simil_df.loc[cur_mcomb, other_mcomb] = other_diff
                    simil_df.loc[cur_mcomb, other_mcomb] /= cur_diff

    return pheno_dict, auc_list, simil_df

