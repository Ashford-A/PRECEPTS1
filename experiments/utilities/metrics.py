
from dryadic.features.mutations import MuType
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_mean_siml(wt_vals, mut_vals, other_vals,
                        wt_mean=None, mut_mean=None, other_mean=None):
    if wt_mean is None:
        wt_mean = wt_vals.mean()

    if mut_mean is None:
        mut_mean = mut_vals.mean()

    if other_mean is None:
        other_mean = other_vals.mean()

    return (other_mean - wt_mean) / (mut_mean - wt_mean)


def calculate_ks_siml(wt_vals, mut_vals, other_vals,
                      base_dist=None, wt_dist=None, mut_dist=None):
    if base_dist is None:
        base_dist = ks_2samp(wt_vals, mut_vals,
                             alternative='greater').statistic
        base_dist -= ks_2samp(wt_vals, mut_vals, alternative='less').statistic

    if wt_dist is None:
        wt_dist = ks_2samp(wt_vals, other_vals,
                           alternative='greater').statistic
        wt_dist -= ks_2samp(wt_vals, other_vals, alternative='less').statistic

    if mut_dist is None:
        mut_dist = ks_2samp(mut_vals, other_vals,
                            alternative='greater').statistic
        mut_dist -= ks_2samp(mut_vals, other_vals,
                             alternative='less').statistic

    return (base_dist + wt_dist + mut_dist) / (2 * base_dist)


def compare_scores(iso_df, samps, muts_dict,
                   get_similarities=True, all_mtype=None):
    base_muts = tuple(muts_dict.values())[0]

    if all_mtype is None:
        all_mtype = MuType(base_muts.allkey())

    pheno_dict = {mtype: np.array(muts_dict[lvls].status(samps, mtype))
                  if lvls in muts_dict
                  else np.array(base_muts.status(samps, mtype))
                  for lvls, mtype in iso_df.index}

    simil_df = pd.DataFrame(0.0, index=pheno_dict.keys(),
                            columns=pheno_dict.keys(), dtype=np.float)
    auc_df = pd.DataFrame(index=pheno_dict.keys(), columns=['All', 'Iso'],
                          dtype=np.float)

    all_pheno = np.array(base_muts.status(samps, all_mtype))
    pheno_dict['Wild-Type'] = ~all_pheno

    for (_, cur_mtype), iso_vals in iso_df.iterrows():
        simil_df.loc[cur_mtype, cur_mtype] = 1.0

        none_vals = np.concatenate(iso_vals[~all_pheno].values)
        wt_vals = np.concatenate(iso_vals[~pheno_dict[cur_mtype]].values)
        cur_vals = np.concatenate(iso_vals[pheno_dict[cur_mtype]].values)

        auc_df.loc[cur_mtype, 'All'] = np.greater.outer(
            cur_vals, wt_vals).mean()
        auc_df.loc[cur_mtype, 'All'] += np.equal.outer(
            cur_vals, wt_vals).mean() / 2

        auc_df.loc[cur_mtype, 'Iso'] = np.greater.outer(
            cur_vals, none_vals).mean()
        auc_df.loc[cur_mtype, 'Iso'] += np.equal.outer(
            cur_vals, none_vals).mean() / 2

        if get_similarities:
            cur_diff = np.subtract.outer(cur_vals, none_vals).mean()

            if cur_diff != 0:
                for other_mtype in set(simil_df.index) - {cur_mtype}:

                    other_vals = np.concatenate(
                        iso_vals[pheno_dict[other_mtype]].values)
                    other_diff = np.subtract.outer(
                        other_vals, none_vals).mean()

                    simil_df.loc[cur_mtype, other_mtype] = other_diff
                    simil_df.loc[cur_mtype, other_mtype] /= cur_diff

    return pheno_dict, auc_df, simil_df


def calc_auc(vals, stat):
    if stat.all() or not stat.any():
        auc_val = 0.5

    else:
        auc_val = np.greater.outer(vals[stat], vals[~stat]).mean()
        auc_val += 0.5 * np.equal.outer(vals[stat], vals[~stat]).mean()

    return auc_val


def calc_conf(auc_vals1, auc_vals2):
    return (np.greater.outer(auc_vals1, auc_vals2).mean()
            + 0.5 * np.equal.outer(auc_vals1, auc_vals2).mean())

