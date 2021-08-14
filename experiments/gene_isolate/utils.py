
import numpy as np
from ..utilities.mutations import ExMcomb
from itertools import product
from functools import reduce
from operator import and_


def check_disjoint(pheno_dict, *mcombs):
    return (all(reduce(and_, mtypes).is_empty()
                for mtypes in product(*[mcomb.mtypes for mcomb in mcombs]))
            or not any(all(phns) for phns in zip(*[pheno_dict[mcomb]
                                                   for mcomb in mcombs])))


def calculate_auc(phn_vec, pred_vals, cv_indx=None, use_mean=False):
    test_stat = [len(vals) == 10 for vals in pred_vals]
    use_phn = phn_vec[test_stat]

    if use_phn.all() or not use_phn.any():
        auc_val = 0.5

    else:
        pred_mat = np.vstack(pred_vals.values[test_stat])

        if cv_indx is None:
            cv_indx = list(range(10))
        elif isinstance(cv_indx, int):
            cv_indx = [cv_indx]

        elif not isinstance(cv_indx, list):
            raise TypeError("`cv_indx` must be a list, an integer value, "
                            "or left as None to use all CV iterations!")

        mut_vals = pred_mat[use_phn].T[cv_indx]
        wt_vals = pred_mat[~use_phn].T[cv_indx]

        assert (mut_vals.shape[0] == wt_vals.shape[0]
                == len(cv_indx)), ("Wrong number of CV iterations in "
                                   "classifier output, must be {}!".format(
                                       len(cv_indx)))

        assert mut_vals.shape[1] == use_phn.sum(), (
            "Wrong number of mutated samples in classifier output, must "
            "be {} instead of {}!".format(use_phn.sum(), mut_vals.shape[1])
            )
        assert wt_vals.shape[1] == (~use_phn).sum(), (
            "Wrong number of wild-type samples in classifier output, must "
            "be {} instead of {}!".format((~use_phn).sum(), wt_vals.shape[1])
            )

        if use_mean:
            mut_vals, wt_vals = mut_vals.mean(axis=0), wt_vals.mean(axis=0)

        auc_val = np.greater.outer(mut_vals, wt_vals).mean()
        auc_val += 0.5 * np.equal.outer(mut_vals, wt_vals).mean()

    return auc_val


def calculate_siml(base_mtype, phn_dict, ex_k, pred_vals):
    siml_vec = {mtype: 1.0 for mtype in phn_dict}

    none_mean = np.concatenate(pred_vals[~phn_dict[ex_k]].values).mean()
    base_mean = np.concatenate(pred_vals[phn_dict[base_mtype]].values).mean()
    cur_diff = base_mean - none_mean

    return {othr_mtype: (np.concatenate(pred_vals[phn].values).mean()
                         - none_mean) / cur_diff
            for othr_mtype, phn in phn_dict.items()
            if isinstance(othr_mtype, ExMcomb)}

