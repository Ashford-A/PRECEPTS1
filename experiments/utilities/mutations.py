
from dryadic.features.mutations import MuType, MutComb
from functools import reduce
from operator import or_


class Mcomb(MutComb):

    def __new__(cls, *mtypes):
        return super().__new__(cls, *mtypes)

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

    def get_sorted_levels(self):
        return self.all_mtype.get_sorted_levels()

    def get_labels(self):
        return list(reduce(or_, [set(mtype.get_labels())
                                 for mtype in self.mtypes]))


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

    def get_sorted_levels(self):
        return self.all_mtype.get_sorted_levels()

    def get_labels(self):
        return list(set(self.all_mtype.get_labels())
                    | reduce(or_, [set(mtype.get_labels())
                                   for mtype in self.mtypes]))

