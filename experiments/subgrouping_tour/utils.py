
from dryadic.features.mutations import MuType
import re


def get_fancy_label(mtype, max_subs=2):
    sub_keys = [skey['Gene', mtype.get_labels()[0]]
                for skey in mtype.subkeys()]

    if len(sub_keys) <= max_subs:
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
        use_lbl = "grouping of {}\n".format(len(sub_keys))

        if len(mtype_lvls) == 1:
            use_lbl = ''.join([use_lbl, "{}s".format(mtype_lvls[0].lower())])
        else:
            use_lbl = ''.join([use_lbl, "mutation types"])

    return use_lbl
