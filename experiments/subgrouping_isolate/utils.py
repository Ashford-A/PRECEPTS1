
from dryadic.features.cohorts.mut import BaseMutationCohort
from dryadic.features.mutations import MuTree
import numpy as np


def search_siml_pair(siml_dicts, mut, other_mut):
    simls = dict()

    for mut_lvls, siml_dfs in siml_dicts.items():
        for siml_df in siml_dfs:
            if mut in siml_df.columns and other_mut in siml_df.index:
                simls[mut_lvls] = siml_df.loc[other_mut, mut]
                break

    return simls

# TODO: clean this up / find another use for it
def calculate_pair_siml(mcomb1, mcomb2, siml_dicts, all_phn=None,
                        pheno_dict=None, pred_vals=None, mean_dict=None):

    run_tests = False
    pair_simls = []

    if len(pair_simls) == 0 or (run_tests and len(pair_simls) == 1):
        if mean_dict is None:
            none_mean = np.concatenate(pred_vals[~all_phn].values).mean()
            base_mean = np.concatenate(
                pred_vals[pheno_dict[mcomb1]].values).mean()
        else:
            none_mean, base_mean = mean_dict['none'], mean_dict['base']

        pair_siml = np.concatenate(
            pred_vals[pheno_dict[mcomb2]].values).mean() - none_mean
        pair_siml /= (base_mean - none_mean)

        if run_tests and len(pair_simls) == 1:
            test_list = mcomb1, mcomb2

            assert (
                pair_siml == tuple(pair_simls.values())[0]), (
                    "Similarity values are internally inconsistent!")

    elif len(pair_simls) == 1:
        pair_siml = tuple(pair_simls.values())[0]

    else:
        raise ValueError("Multiple similarity values found!")

    if run_tests:
        return pair_siml, test_list
    else:
        return pair_siml


def remove_pheno_dups(muts, pheno_dict):
    mut_phns = set()
    mut_list = set()

    for mut in muts:
        mut_phn = tuple(pheno_dict[mut].tolist())

        if mut_phn not in mut_phns:
            mut_phns |= {mut_phn}
            mut_list |= {mut}

    return mut_list

