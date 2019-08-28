
from dryadic.features.mutations import MuType, MutComb
from scipy.stats import rv_discrete
import random


class RandomType(MuType):

    def __init__(self, base_mtype=None, size_dist=None, seed=None):
        self.base_mtype = base_mtype
        self.seed = seed

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

    def get_samples(self, mtree):
        random.seed(self.seed)
        use_size = self.size_rv.rvs()

        if self.base_mtype:
            use_samps = self.base_mtype.get_samples(mtree)
        else:
            use_samps = mtree.get_samples()

        return set(random.sample(sorted(use_samps), k=use_size))

