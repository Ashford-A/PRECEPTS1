
from dryadic.features.mutations import MuType
from ..utilities.mutations import RandomType

import numpy as np
import pandas as pd
from functools import reduce
from operator import or_


class MutThresh(MuType):

    def __init__(self, annot, min_val, base_mtype):
        self.annot = annot
        self.min_val = min_val
        self.base_mtype = base_mtype

        super().__init__([])

    def __getstate__(self):
        return self.annot, self.min_val, self.base_mtype

    def __setstate__(self, state):
        self.__init__(*state)

    def __hash__(self):
        value = 0x311891 ^ (hash(self.annot) * hash(self.min_val)
                            * hash(self.base_mtype))
        if value == -1:
            value = -2

        return value

    def __str__(self):
        return "{}, {} >= {}".format(self.base_mtype,
                                     self.annot, self.min_val)

    def __repr__(self):
        return "{} with {} value of at least {}".format(
            self.base_mtype, self.annot, self.min_val)

    def __eq__(self, other):
        if not isinstance(other, MuType):
            return NotImplemented

        elif not isinstance(other, MutThresh):
            return False

        elif self.base_mtype != other.base_mtype:
            return False
        elif self.annot != other.annot:
            return False
        elif self.min_val != other.min_val:
            return False

        else:
            return True

    def __lt__(self, other):
        if not isinstance(other, MuType):
            return NotImplemented

        elif isinstance(other, RandomType):
            return True
        elif not isinstance(other, MutThresh):
            return False

        elif self.base_mtype != other.base_mtype:
            return self.base_mtype < other.base_mtype
        elif self.annot != other.annot:
            return self.annot < other.annot

        else:
            return self.min_val < other.min_val

    def get_samples(self, mtree):
        if self.annot == 'VAF':
            lf_annt = self.base_mtype.get_leaf_annot(
                mtree, ['ref_count', 'alt_count'])

        else:
            lf_annt = self.base_mtype.get_leaf_annot(mtree, [self.annot])

        if self.annot == 'VAF':
            samps = {samp for samp, vals in lf_annt.items()
                     if (all(np.isnan(alt_cnt)
                             for alt_cnt in vals['alt_count'])
                         or (max(alt_cnt / (alt_cnt + ref_cnt)
                                 for alt_cnt, ref_cnt in zip(
                                     vals['alt_count'], vals['ref_count']))
                             >= self.min_val))}

        else:
            samps = {samp for samp, vals in lf_annt.items()
                     if (np.isnan(vals[self.annot]).all()
                         or (max(vals[self.annot]) >= self.min_val))}

        return samps

    def get_sorted_levels(self):
        return self.base_mtype.get_sorted_levels()

