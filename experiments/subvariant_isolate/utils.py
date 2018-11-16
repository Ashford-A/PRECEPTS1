
from dryadic.features.mutations import MuType
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
    samps = sorted(cdata.train_samps)

    if all_mtype is None:
        all_mtype = MuType(muts.allkey())
    all_pheno = np.array(muts.status(samps, all_mtype))

    for mtypes in iso_df.index:
        rest_pheno = np.array(muts.status(samps,
                                          all_mtype - reduce(or_, mtypes)))
        cur_phenos = [np.array(muts.status(samps, mtype)) for mtype in mtypes]
        and_pheno = reduce(and_, cur_phenos)

        pheno_dict[mtypes] = and_pheno
        pheno_dict[mtypes] &= ~(rest_pheno
                                | (reduce(or_, cur_phenos) & ~and_pheno))

    for cur_mtypes, iso_vals in iso_df.iterrows():
        simil_df.loc[cur_mtypes, cur_mtypes] = 1.0

        none_vals = np.concatenate(iso_vals[~all_pheno].values)
        cur_vals = np.concatenate(iso_vals[pheno_dict[cur_mtypes]].values)

        auc_list[cur_mtypes] = np.greater.outer(cur_vals, none_vals).mean()
        auc_list[cur_mtypes] += np.equal.outer(cur_vals, none_vals).mean() / 2

        if get_similarities:
            cur_diff = np.subtract.outer(cur_vals, none_vals).mean()

            if cur_diff != 0:
                for other_mtypes in set(iso_df.index) - {cur_mtypes}:

                    other_vals = np.concatenate(
                        iso_vals[pheno_dict[other_mtypes]].values)
                    other_diff = np.subtract.outer(
                        other_vals, none_vals).mean()

                    simil_df.loc[[cur_mtypes], [other_mtypes]] = other_diff
                    simil_df.loc[[cur_mtypes], [other_mtypes]] /= cur_diff

    return simil_df, auc_list, pheno_dict

