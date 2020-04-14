
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_isolate.utils import ExMcomb
from dryadic.features.mutations import MuType, MutComb

from functools import reduce
from operator import or_


def get_mtype_gene(mtype):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is not None:
            mtype_genes = mtype.base_mtype.get_labels()

        else:
            mtype_genes = [None]

    elif isinstance(mtype, ExMcomb):
        mtype_genes = mtype.all_mtype.get_labels()

    elif isinstance(mtype, MutComb):
        mtype_genes = list(reduce(or_, [set(mtp.get_labels())
                                        for mtp in mtype.mtypes]))

    elif isinstance(mtype, MuType):
        mtype_genes = mtype.get_labels()

    else:
        raise ValueError("Cannot retrieve gene for something that is "
                         "not a mutation!")

    assert isinstance(mtype_genes, list)

    if len(mtype_genes) > 1:
        raise ValueError("Cannot retrieve gene for a mutation associated "
                         "with multiple genes!")

    if len(mtype_genes) == 0:
        raise ValueError("Cannot retrieve gene for a mutation associated "
                         "with no genes!")

    return mtype_genes[0]

