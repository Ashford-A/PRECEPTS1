
from dryadic.features.mutations import MuType
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm


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
    """Calculates the area under the ROC curve

    Args:
        vals (np.array): A vector of continuous predicted labels.
        stat (np.array): The ground truth binary class labels.

    Returns:
        auc_val (float)

    """
    ##### Added by Andrew to see what variables passed to this function contain #####
    #print(vals)
    #print(stat)
    #print(vals[stat])
    #print(vals[~stat])
    #print(vals[~stat].mean)

    if stat.all() or not stat.any():
        print("########## default_AUC ##########")
        auc_val = 0.5

        ##### Added by Andrew to see what variables passed to this function contain #####
        print(vals)
        print(stat)

    else:
        auc_val = np.greater.outer(vals[stat], vals[~stat]).mean()
        auc_val += 0.5 * np.equal.outer(vals[stat], vals[~stat]).mean()
        
        ##### Added by Andrew to see what variables passed to this function contain #####
        print(vals)
        print(stat)
        #print(vals[stat])
        #print(vals[~stat])
        #print(vals[~stat].mean)

    return auc_val


def calc_conf(auc_vals1, auc_vals2):
    return (np.greater.outer(auc_vals1, auc_vals2).mean()
            + 0.5 * np.equal.outer(auc_vals1, auc_vals2).mean())


def calc_delong(preds1, preds2, stat, auc1=None, auc2=None):
    """Calculates the one-sided version of DeLong's test statistic.

    Args:
        preds1, preds2 (np.array)
            Vectors of continuous predicted labels. This function tests to
            what extent we can reject the hypothesis that `preds1` does not
            better predict the ground truth labels than `preds2`.

        stat (np.array): The ground truth binary class labels.
        auc1, auc2 (float, optional)
            Pre-computed AUCs can be given if possible which will save time.

    Returns:
        delong_val (float)

    """
    strc1 = np.greater.outer(preds1[stat], preds1[~stat]).astype(float)
    strc1 += 0.5 * np.equal.outer(preds1[stat], preds1[~stat]).astype(float)
    strc2 = np.greater.outer(preds2[stat], preds2[~stat]).astype(float)
    strc2 += 0.5 * np.equal.outer(preds2[stat], preds2[~stat]).astype(float)

    if auc1 is None:
        auc1 = strc1.mean()
    if auc2 is None:
        auc2 = strc2.mean()

    mut_n, wt_n = strc1.shape
    vvecs1 = strc1.mean(axis=1), strc1.mean(axis=0)
    vvecs2 = strc2.mean(axis=1), strc2.mean(axis=0)

    smat1 = [[((vv_i[0] - auc_i) * (vv_j[0] - auc_j)).sum() / (mut_n - 1)
              for vv_j, auc_j in zip([vvecs1, vvecs2], [auc1, auc2])]
             for vv_i, auc_i in zip([vvecs1, vvecs2], [auc1, auc2])]
    smat2 = [[((vv_i[1] - auc_i) * (vv_j[1] - auc_j)).sum() / (wt_n - 1)
              for vv_j, auc_j in zip([vvecs1, vvecs2], [auc1, auc2])]
             for vv_i, auc_i in zip([vvecs1, vvecs2], [auc1, auc2])]

    smat = np.array(smat1) / strc1.shape[0] + np.array(smat2) / strc1.shape[1]
    z_scr = (auc1 - auc2) / np.sqrt(smat[0, 0] + smat[1, 1] - 2 * smat[1, 0])

    return norm.sf(z_scr)

