
from dryadic.features.mutations import MuType, MutComb
import re
from scipy.stats import rv_discrete
import random


def get_fancy_label(mtype):
    sub_keys = [skey['Gene', mtype.get_labels()[0]]
                for skey in mtype.subkeys()]

    if len(sub_keys) <= 2:
        use_lbls = []

        for lbl_key in sub_keys:
            sub_mtype = MuType(lbl_key)
            sub_lvls = sub_mtype.get_sorted_levels()

            if (sub_lvls[-1] == 'Protein' and not any('Form' in lvl
                                                      for lvl in sub_lvls)):
                sub_lbls = [str(sub_mtype).split(':')[-1].split('p.')[-1]]

            else:
                sub_lbls = str(sub_mtype).split(':')

            dom_lvls = [slvl for slvl in sub_lvls if "Domain_" in slvl]
            if len(dom_lvls) == 1:
                dom_lbl = dom_lvls[0].split('_')[1]

            for i in range(len(sub_lbls)):
                if 'none' in sub_lbls[i]:
                    sub_lbls[i] = sub_lbls[i].replace(
                        "none", "no {} domain".format(dom_lbl))

                sub_lbls[i] = sub_lbls[i].replace("_Mutation", "")
                sub_lbls[i] = sub_lbls[i].replace("_", "")
                sub_lbls[i] = sub_lbls[i].replace("|", " or ")

                for dom_prfx in ['SM', 'PF']:
                    dom_mtch = re.search("(^[:]|^){}[0-9]".format(dom_prfx),
                                         sub_lbls[i])

                    while dom_mtch:
                        sub_lbls[i] = "{}Domain:{}-{}".format(
                            sub_lbls[i][:dom_mtch.span()[0]], dom_lbl,
                            sub_lbls[i][(dom_mtch.span()[1] - 1):]
                            )

                        dom_mtch = re.search(
                            "(^[:]|^){}[0-9]".format(dom_prfx), sub_lbls[i])

                exn_mtch = re.search("([0-9]+)/([0-9]+)", sub_lbls[i])
                while exn_mtch:
                    sub_lbls[i] = "Exon:{}{}".format(
                        exn_mtch.groups()[0],
                        sub_lbls[i][exn_mtch.span()[1]:]
                        )
                    exn_mtch = re.search("([0-9]+)/([0-9]+)", sub_lbls[i])

            use_lbls += ['\nwith '.join(sub_lbls)]
        use_lbl = '\nor '.join(use_lbls)

    else:
        mtype_lvls = mtype.get_sorted_levels()[1:]

        if len(mtype_lvls) == 1:
            use_lbl = "grouping of 3+ {}s".format(mtype_lvls[0].lower())
        else:
            use_lbl = "grouping of\n3+ mutation types"

    return use_lbl


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

    def get_sorted_levels(self):
        return None

    def get_samples(self, mtree):
        random.seed(self.seed)
        use_size = self.size_rv.rvs()

        if self.base_mtype:
            use_samps = self.base_mtype.get_samples(mtree)
        else:
            use_samps = mtree.get_samples()

        return set(random.sample(sorted(use_samps), k=use_size))

