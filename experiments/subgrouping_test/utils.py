
from ..utilities.misc import get_label, choose_label_colour
from ..utilities.pcawg_colours import cohort_clrs
from ..utilities.mutations import RandomType


def choose_mtype_colour(mtype):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is None:
            plt_clr = '0.59'
        else:
            plt_clr = choose_label_colour(tuple(
                mtype.base_mtype.label_iter())[0])

    else:
        plt_clr = choose_label_colour(tuple(mtype.label_iter())[0])

    return plt_clr


def choose_cohort_colour(cohort):
    coh_base = cohort.split('_')[0]

    # if using a non-TCGA cohort, match to a TCGA cohort of the same
    # disease type, using white for pan-cancer cohorts
    if coh_base == 'METABRIC':
        use_clr = cohort_clrs['BRCA']
    elif coh_base == 'beatAML':
        use_clr = cohort_clrs['LAML']
    elif coh_base == 'CCLE':
        use_clr = '#000000'

    # otherwise, choose the colour according to the PCAWG scheme
    else:
        use_clr = cohort_clrs[coh_base]

    # convert the hex colour to a [0-1] RGB tuple
    return tuple(int(use_clr.lstrip('#')[i:(i + 2)], 16) / 256
                 for i in range(0, 6, 2))


def filter_mtype(mtype, gene):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is None:
            filter_stat = False
        else:
            filter_stat = get_label(mtype.base_mtype) == gene

    else:
        filter_stat = get_label(mtype) == gene

    return filter_stat
