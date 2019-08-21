
import os
from glob import glob
import dill as pickle

from dryadic.features.mutations import MuType, MutComb
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
import random

from functools import reduce
from operator import or_


class Mcomb(MutComb):

    def __new__(cls, *mtypes):
        return super().__new__(cls, *mtypes)

    def __hash__(self):
        value = 0x230199 ^ len(self.mtypes)
        value += hash(self.mtypes)

        if value == -1:
            value = -2

        return value

    def __getnewargs__(self):
        return tuple(self.mtypes)

    def __str__(self):
        return ' & '.join(str(mtype) for mtype in sorted(self.mtypes))

    def __repr__(self):
        return 'BOTH {}'.format(
            ' AND '.join(repr(mtype) for mtype in sorted(self.mtypes)))

    def __eq__(self, other):
        if not isinstance(other, Mcomb):
            eq = False
        else:
            eq = self.mtypes == other.mtypes

        return eq

    def __lt__(self, other):
        if isinstance(other, Mcomb):
            lt = self.mtypes < other.mtypes

        elif isinstance(other, MuType):
            lt = False
        elif isinstance(other, ExMcomb):
            lt = True
        else:
            lt = NotImplemented

        return lt


class ExMcomb(MutComb):

    def __new__(cls, mtree, *mtypes):
        if isinstance(mtree, MuType):
            all_mtype = mtree
        else:
            all_mtype = MuType(mtree.allkey())

        obj = super().__new__(cls, *mtypes,
                              not_mtype=all_mtype - reduce(or_, mtypes))
        obj.all_mtype = all_mtype
        obj.cur_level = all_mtype.cur_level

        return obj

    def __hash__(self):
        value = 0x981324 ^ (len(self.mtypes) * hash(self.all_mtype))
        value += hash(self.mtypes)

        if value == -1:
            value = -2

        return value

    def __getnewargs__(self):
        return (self.all_mtype,) + tuple(self.mtypes)

    def __str__(self):
        return ' & '.join(str(mtype) for mtype in sorted(self.mtypes))

    def __repr__(self):
        return 'ONLY {}'.format(
            ' AND '.join(repr(mtype) for mtype in sorted(self.mtypes)))

    def __eq__(self, other):
        if not isinstance(other, ExMcomb):
            eq = False

        else:
            eq = self.all_mtype == other.all_mtype
            eq &= self.mtypes == other.mtypes

        return eq

    def __lt__(self, other):
        if isinstance(other, ExMcomb):
            if self.all_mtype != other.all_mtype:
                lt = self.all_mtype < other.all_mtype
            else:
                lt = self.mtypes < other.mtypes

        elif isinstance(other, (MuType, Mcomb)):
            lt = False
        else:
            lt = NotImplemented

        return lt

    def get_sorted_levels(self):
        return self.all_mtype.get_sorted_levels()


class RandomType(MuType):

    def __init__(self, base_mtype=None, size_dist=None, seed=None):
        self.base_mtype = base_mtype

        if size_dist is None and base_mtype is None:
            raise ValueError("Must provide either a mutation type to sample "
                             "over or a discrete probability distribution!")

        elif isinstance(size_dist, int):
            self.size_dist = size_dist

            self.size_rv = rv_discrete(a=size_dist, b=size_dist,
                                       values=([size_dist], [1]), seed=seed)

        elif len(size_dist) == 2 and isinstance(size_dist[0], int):
            self.size_dist = tuple(size_dist)

            self.size_rv = rv_discrete(
                a=size_dist[0], b=size_dist[1],
                values=([x for x in range(size_dist[0], size_dist[1] + 1)],
                        [(size_dist[1] + 1 - size_dist[0]) ** -1
                         for _ in range(size_dist[0], size_dist[1] + 1)]),
                seed=seed
                )

        elif isinstance(size_dist, set):
            self.size_dist = tuple(size_dist)

            self.size_rv = rv_discrete(
                a=min(size_dist), b=max(size_dist),
                values=(sorted(size_dist),
                        [len(size_dist) ** -1 for _ in size_dist]),
                seed=seed
                )

        else:
            raise ValueError("Unrecognized size distribution!")

        super().__init__([])

    def __hash__(self):
        value = 0x302378 ^ (hash(self.base_mtype) * hash(self.size_dist)
                            * hash(self.size_rv.random_state))

        if value == -1:
            value = -2

        return value

    def __str__(self):
        use_str = str(self.size_dist)

        if self.base_mtype:
            use_str = "{} ({})".format(use_str, str(self.base_mtype))

        return use_str

    def __repr__(self):
        use_str = "random samples using {} with seed {}".format(
            self.size_rv, self.size_rv.random_state)

        if self.base_mtype:
            use_str = "({}) {}".format(str(self.base_mtype), use_str)

        return use_str

    def __eq__(self, other):
        if not isinstance(other, MuType):
            return NotImplemented

        elif not isinstance(other, RandomType):
            return False

        elif self.base_mtype != other.base_mtype:
            return False
        elif self.size_dist != other.size_dist:
            return False
        elif self.size_rv.random_state != other.size_rv.random_state:
            return False

        else:
            return True

    def __lt__(self, other):
        if isinstance(other, RandomType):
            if self.base_mtype is None and other.base_mtype is not None:
                return False
            elif self.base_mtype is not None and other.base_mtype is None:
                return True
            elif self.base_mtype is not None and other.base_mtype is not None:
                return self.base_mtype < other.base_mtype

            elif isinstance(self.size_dist, int):
                if isinstance(other.size_dist, tuple):
                    return True

            elif isinstance(self.size_dist, tuple):
                if isinstance(other.size_dist, int):
                    return False

            elif self.size_dist < other.size_dist:
                return True
            elif self.size_rv.random_state < other.size_rv.random_state:
                return True

            else:
                return False

        elif isinstance(other, (MuType, MutComb)):
            return True

        else:
            return NotImplemented

    def get_samples(self, mtree):
        random.seed(self.size_rv.random_state)
        use_size = self.size_rv.rvs()

        if self.base_mtype:
            use_samps = self.base_mtype.get_samples(mtree)
        else:
            use_samps = mtree.get_samples()

        return set(random.sample(sorted(use_samps), k=use_size))


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
    return (np.greater.outer(vals[stat], vals[~stat]).mean()
            + 0.5 * np.equal.outer(vals[stat], vals[~stat]).mean())

