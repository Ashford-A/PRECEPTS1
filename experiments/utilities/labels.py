
from .mutations import (shal_mtype, deep_mtype,
                        dup_mtype, loss_mtype, gains_mtype, dels_mtype)
from dryadic.features.mutations import MuType
from Bio.SeqUtils import seq1
import re


copy_lbls = {shal_mtype: 'any shallow loss/gain',
             deep_mtype: 'any deep loss/gain',
             dup_mtype: 'deep gains', loss_mtype: 'deep losses',
             gains_mtype: 'any gain', dels_mtype: 'any loss'}


def get_cohort_label(coh):
    if '_' in coh:
        coh_lbl = "{}({})".format(*coh.split('_'))
        coh_lbl = coh_lbl.replace("IDHmut-non-codel", "IDHmut-nc")
        coh_lbl = coh_lbl.replace("SquamousCarcinoma", "SqmsCarc")

    else:
        coh_lbl = str(coh)

    return coh_lbl


def ordinal_label(n):
    return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20
                                                        else n % 10, "th"))


def parse_hgvs(hgvs_lbl):
    return re.sub('[A-Z][a-z][a-z]', lambda mtch: seq1(mtch.group()),
                  hgvs_lbl)
 

def nest_label(mtype, sub_link=' or ', phrase_link=' '):
    sub_lbls = []

    for lbls, tp in mtype.child_iter():
        if (tp is not None and len(lbls) == 1
                and tp.get_sorted_levels()[-1][:4] == 'HGVS'):
            hgvs_lbl = str(tp).split(':')[-1].split('.')[-1]

            if hgvs_lbl == '-':
                sub_lbls += ["(no location)"]
            else:
                sub_lbls += [parse_hgvs(hgvs_lbl)]

        else:
            if mtype.cur_level[:4] == 'HGVS':
                hgvs_lbls = [str(lbl).split(':')[-1].split('.')[-1]
                             for lbl in lbls]

                sub_lbls += [' or '.join(["(no location)" if hgvs_lbl == '-'
                                          else parse_hgvs(hgvs_lbl)
                                          for hgvs_lbl in hgvs_lbls])]

            elif mtype.cur_level == 'Exon':
                if len(lbls) == 1:
                    if tuple(lbls)[0] != '-':
                        sub_lbls += ["on {} exon".format(
                            ordinal_label(int(tuple(lbls)[0].split('/')[0])))]
                    else:
                        sub_lbls += ["not on any exon"]

                else:
                    exn_lbls = [ordinal_label(int(lbl.split('/')[0]))
                                if lbl != '-' else 'no exon'
                                for lbl in sorted(lbls)]
                    exn_lbls[-1] = 'or {}'.format(exn_lbls[-1])

                    if len(exn_lbls) > 2:
                        exn_jn = ', '
                    else:
                        exn_jn = ' '

                    sub_lbls += ["on {} exons".format(exn_jn.join(exn_lbls))]

            elif mtype.cur_level == 'Position':
                if len(lbls) == 1:
                    if tuple(lbls)[0] != '-':
                        sub_lbls += ["at codon {}".format(tuple(lbls)[0])]
                    else:
                        sub_lbls += ["(no codon)"]

                else:
                    cdn_lbls = sorted(lbls)
                    cdn_lbls[-1] = 'or {}'.format(cdn_lbls[-1])

                    if len(cdn_lbls) > 2:
                        cdn_jn = ', '
                    else:
                        cdn_jn = ' '

                    sub_lbls += ["at codons {}".format(cdn_jn.join(cdn_lbls))]

            elif mtype.cur_level == 'Consequence':
                sub_lbls += [' or '.join([
                    lbl.replace("_variant", "").replace('_', ' ')
                    for lbl in sorted(lbls)
                    ])]

            elif '-domain' in mtype.cur_level:
                dtb_lbl = mtype.cur_level.split('-domain')[0]

                if len(lbls) == 1:
                    if tuple(lbls)[0] != 'none':
                        sub_lbls += ["on domain {}".format(tuple(lbls)[0])]
                    else:
                        sub_lbls += ["not on any {} domain".format(dtb_lbl)]

                else:
                    dmn_lbls = sorted(lbls)
                    dmn_lbls[-1] = 'or {}'.format(dmn_lbls[-1])

                    if len(dmn_lbls) > 2:
                        dmn_link = ', '
                    else:
                        dmn_link = ' '

                    sub_lbls += ["on domains {}".format(
                        dmn_link.join(dmn_lbls))]

            elif mtype.cur_level == 'Impact':
                lbl_list = [lbl.lower() for lbl in lbls]

                if len(lbls) == 1:
                    sub_lbls += ["{} impact".format(lbl_list[0])]
                else:
                    sub_lbls += ["{} or {} impact".format(
                        ', '.join(lbl_list[:-1]), lbl_list[1])]

                    if len(lbls) == 2:
                        sub_lbls[-1] = sub_lbls[-1].replace(',', '')

            elif mtype.cur_level == 'Class':
                if lbls - {'SNV', 'insertion', 'deletion'}:
                    raise ValueError(
                        "Unrecognized `Class` attribute labels {} !".format(
                            lbls - {'SNV', 'insertion', 'deletion'})
                        )

                lbl_list = ["{}s".format(lbl) for lbl in lbls]
                if lbl_list == ["SNVs"] and tp is None:
                    sub_lbls += ["any SNV"]

                elif lbl_list != ["SNVs"] and len(lbl_list) == 1:
                    sub_lbls += [lbl_list[0]]

                elif len(lbl_list) > 1:
                    sub_lbls += ["{} or {}".format(
                        ', '.join(lbl_list[:-1]), lbl_list[1])]

                    if len(lbls) == 2:
                        sub_lbls[-1] = sub_lbls[-1].replace(',', '')

            else:
                raise ValueError("Unrecognized type of mutation "
                                 "level `{}`!".format(mtype.cur_level))

            if tp is not None:
                if mtype.cur_level == 'Class' and lbl_list == ["SNVs"]:
                    sub_lbls += [nest_label(tp)]

                elif mtype.cur_level == 'Class' and lbl_list != ["SNVs"]:
                    sub_words = nest_label(tp).split(' ')

                    if len(sub_words) > 1:
                        if sub_words[1][:3] in {'ins', 'del'}:
                            sub_words = [sub_words[0]] + sub_words[2:]

                        if len(sub_words) > 1:
                            sub_lbls[-1] = phrase_link.join([
                                ' '.join([sub_words[0], sub_lbls[-1]]),
                                ' '.join(sub_words[1:])
                                ])

                        else:
                            sub_lbls[-1] = ' '.join([sub_words[0],
                                                     sub_lbls[-1]])

                    else:
                        sub_lbls[-1] = phrase_link.join([sub_words[0],
                                                         sub_lbls[-1]])

                elif tp.cur_level == 'Consequence':
                    sub_lbls[-1] = phrase_link.join([
                        nest_label(tp), sub_lbls[-1]])

                else:
                    sub_lbls[-1] = phrase_link.join([
                        sub_lbls[-1], nest_label(tp)])

    return sub_link.join(sub_lbls)


def get_fancy_label(mtype, scale_link=None, pnt_link=None, phrase_link=None):
    if scale_link is None:
        scale_link = ' or '

    if pnt_link is None:
        pnt_link = ' or '
    if phrase_link is None:
        phrase_link = ' '

    if mtype.cur_level == 'Scale':
        sub_dict = dict(mtype.subtype_iter())
        use_lbls = []

        if 'Copy' in sub_dict:
            use_lbls += [copy_lbls[
                MuType({('Scale', 'Copy'): sub_dict['Copy']})]]

        if 'Point' in sub_dict:
            if sub_dict['Point'] is None:
                use_lbls += ["any point mutation"]

            else:
                use_lbls += [nest_label(sub_dict['Point'],
                                        pnt_link, phrase_link)]

    else:
        use_lbls = [nest_label(mtype, pnt_link, phrase_link)]

    return scale_link.join(use_lbls)

