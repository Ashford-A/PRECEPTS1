"""
Novel mutation types introduced across various experiments.
"""

from dryadic.features.mutations import MuType, MutComb
from functools import reduce
from operator import or_, and_
from scipy.stats import rv_discrete
import random


pnt_mtype = MuType({('Scale', 'Point'): None})
copy_mtype = MuType({('Scale', 'Copy'): None})

shal_mtype = MuType({('Scale', 'Copy'): {(
    'Copy', ('ShalGain', 'ShalDel')): None}})
deep_mtype = MuType({('Scale', 'Copy'): {(
    'Copy', ('DeepGain', 'DeepDel')): None}})

dup_mtype = MuType({('Scale', 'Copy'): {('Copy', 'DeepGain'): None}})
loss_mtype = MuType({('Scale', 'Copy'): {('Copy', 'DeepDel'): None}})

gains_mtype = MuType({('Scale', 'Copy'): {(
    'Copy', ('ShalGain', 'DeepGain')): None}})
dels_mtype = MuType({('Scale', 'Copy'): {(
    'Copy', ('ShalDel', 'DeepDel')): None}})


class Mcomb(MutComb):

    def __new__(cls, *mtypes):
        obj = super().__new__(cls, *mtypes)

        if isinstance(obj, MutComb):
            cur_levels = {mtype.cur_level for mtype in mtypes}

            if len(cur_levels) > 1:
                raise ValueError("Cannot combine mutations "
                                 "of diferent levels!")

            obj.cur_level = tuple(cur_levels)[0]

        return obj

    def __hash__(self):
        value = 0x230199 ^ len(self.mtypes)
        value += hash(tuple(sorted(self.mtypes)))

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
            eq = sorted(self.mtypes) == sorted(other.mtypes)

        return eq

    def __lt__(self, other):
        if isinstance(other, Mcomb):
            lt = sorted(self.mtypes) < sorted(other.mtypes)

        elif isinstance(other, MuType):
            lt = False
        elif isinstance(other, ExMcomb):
            lt = True
        else:
            lt = NotImplemented

        return lt

    def label_iter(self):
        return iter(reduce(or_, [set(mtype.label_iter())
                                 for mtype in self.mtypes]))


class ExMcomb(MutComb):

    def __new__(cls, all_mtype, *mtypes):
        obj = super().__new__(cls, *mtypes,
                              not_mtype=all_mtype - reduce(or_, mtypes))

        obj.all_mtype = all_mtype
        obj.cur_level = all_mtype.cur_level

        return obj

    def __hash__(self):
        value = 0x981324 ^ (len(self.mtypes) * hash(self.all_mtype))
        value += hash(tuple(sorted(self.mtypes)))

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
            eq &= sorted(self.mtypes) == sorted(other.mtypes)

        return eq

    def __lt__(self, other):
        if isinstance(other, ExMcomb):
            if self.all_mtype != other.all_mtype:
                lt = self.all_mtype < other.all_mtype
            else:
                lt = sorted(self.mtypes) < sorted(other.mtypes)

        elif isinstance(other, (MuType, Mcomb)):
            lt = False
        else:
            lt = NotImplemented

        return lt

    def get_samples(self, *mtrees):
        samps = self.mtype_apply(
            lambda mtype: mtype.get_samples(*mtrees), and_)

        if self.not_mtype is not None:
            if self.cur_level == 'Gene':
                for gn, sub_type in self.not_mtype.subtype_iter():
                    samps -= MuType({
                        ('Gene', gn): sub_type}).get_samples(*mtrees)

            else:
                samps -= self.not_mtype.get_samples(*mtrees)

        return samps

    def get_sorted_levels(self):
        return self.all_mtype.get_sorted_levels()

    def label_iter(self):
        return iter(set(self.all_mtype.label_iter())
                    | reduce(or_, [set(mtype.label_iter())
                                   for mtype in self.mtypes]))


class RandomType(MuType):

    def __init__(self, size_dist, base_mtype=None, seed=None):
        self.size_dist = size_dist
        self.base_mtype = base_mtype
        self.seed = seed

        if isinstance(size_dist, int):
            self.size_rv = rv_discrete(a=size_dist, b=size_dist,
                                       values=([size_dist], [1]), seed=seed)

        elif len(size_dist) == 2 and isinstance(size_dist[0], int):
            self.size_rv = rv_discrete(
                a=size_dist[0], b=size_dist[1],
                values=([x for x in range(size_dist[0], size_dist[1] + 1)],
                        [(size_dist[1] + 1 - size_dist[0]) ** -1
                         for _ in range(size_dist[0], size_dist[1] + 1)]),
                seed=seed
                )

        elif isinstance(size_dist, set):
            self.size_rv = rv_discrete(
                a=min(size_dist), b=max(size_dist),
                values=(sorted(size_dist),
                        [len(size_dist) ** -1 for _ in size_dist]),
                seed=seed
                )

        else:
            raise ValueError("Unrecognized size distribution "
                             "`{}` !".format(size_dist))

        super().__init__([])

    def __getstate__(self):
        return self.size_dist, self.base_mtype, self.seed

    def __setstate__(self, state):
        self.__init__(state[0], base_mtype=state[1], seed=state[2])

    def __hash__(self):
        value = 0x302378 ^ (hash(self.base_mtype) * hash(self.size_dist)
                            * hash(self.seed))

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
            str(self.size_dist), self.seed)

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
        elif self.seed != other.seed:
            return False

        else:
            return True

    def __lt__(self, other):
        if isinstance(other, RandomType):
            if self.base_mtype is None and other.base_mtype is not None:
                return False
            elif self.base_mtype is not None and other.base_mtype is None:
                return True
            elif (self.base_mtype is not None and other.base_mtype is not None
                    and self.base_mtype != other.base_mtype):
                return self.base_mtype < other.base_mtype

            elif (isinstance(self.size_dist, int)
                    and isinstance(other.size_dist, tuple)):
                return True

            elif (isinstance(self.size_dist, tuple)
                    and isinstance(other.size_dist, int)):
                return False

            elif self.size_dist != other.size_dist:
                return self.size_dist < other.size_dist

            else:
                return self.seed < other.seed

        elif isinstance(other, (MuType, MutComb)):
            return True

        else:
            return NotImplemented

    def get_samples(self, *mtrees):
        #TODO: having two random seed setups is not great
        random.seed(self.seed)
        use_size = self.size_rv.rvs(random_state=self.seed)

        if self.base_mtype:
            use_samps = self.base_mtype.get_samples(*mtrees)
        else:
            use_samps = mtrees[0].get_samples()

        return set(random.sample(sorted(use_samps), k=use_size))

    def get_sorted_levels(self):
        if self.base_mtype is None:
            lvls = None
        else:
            lvls = self.base_mtype.get_sorted_levels()

        return lvls

    def label_iter(self):
        if self.base_mtype is None:
            mut_lbls = None
        else:
            mut_lbls = self.base_mtype.label_iter()

        return mut_lbls

