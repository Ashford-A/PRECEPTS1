
from ..utilities.labels import ordinal_label
from ..utilities.misc import choose_label_colour
from ..utilities.mutations import RandomType
from dryadic.features.mutations import MuType
import re


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


def label_subtype(sub_type):
    sub_lvls = sub_type.get_sorted_levels()

    if (sub_lvls[-1] == 'Protein' and not any('Form' in lvl
                                              for lvl in sub_lvls)):
        sub_lbls = [str(sub_type).split(':')[-1].split('p.')[-1]]

    else:
        sub_lbls = str(sub_type).split(':')

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

        if sub_lvls[i] == 'Exon':
            sub_lbls[i] = sub_lbls[i].replace(".", "no exon")

        for dom_prfx in ['SM', 'PF']:
            dom_mtch = re.search("(^[:]|^){}[0-9]".format(dom_prfx),
                                 sub_lbls[i])

            while dom_mtch:
                sub_lbls[i] = "{}Domain={}-{}".format(
                    sub_lbls[i][:dom_mtch.span()[0]], dom_lbl,
                    sub_lbls[i][(dom_mtch.span()[1] - 1):]
                    )

                dom_mtch = re.search(
                    "(^[:]|^){}[0-9]".format(dom_prfx), sub_lbls[i])

        exn_mtch = re.search("([0-9]+)/([0-9]+)", sub_lbls[i])
        while exn_mtch:
            sub_lbls[i] = "{} exon{}".format(
                ordinal_label(int(exn_mtch.groups()[0])),
                sub_lbls[i][exn_mtch.span()[1]:]
                )
            exn_mtch = re.search("([0-9]+)/([0-9]+)", sub_lbls[i])

    return '\nwith '.join(sub_lbls)


def get_fancy_label(mtype):
    sub_dict = dict(mtype.subtype_list()[0][1].subtype_list())

    if 'Copy' in sub_dict:
        use_lbls = sub_dict['Copy'].get_labels()
    else:
        use_lbls = []

    if 'Point' in sub_dict:
        if sub_dict['Point'] is None:
            use_lbls += ["any point mutation"]

        else:
            use_lbls += [label_subtype(MuType(sub_key))
                         for sub_key in sub_dict['Point'].subkeys()]

    return "{} mutations that are\n{}".format(
        mtype.get_labels()[0], '\nor '.join(use_lbls))

