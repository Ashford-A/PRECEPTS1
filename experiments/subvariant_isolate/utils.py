
from HetMan.experiments.utilities.misc import ordinal_label
from dryadic.features.mutations import MuType

import numpy as np
import pandas as pd
import re


def nest_label(mtype, sub_link=' or ', phrase_link=' '):
    sub_lbls = []

    for lbls, tp in mtype.child_iter():
        if (tp is not None and len(lbls) == 1
                and tp.get_sorted_levels()[-1][:4] == 'HGVS'):
            hgvs_lbl = re.sub('[a-z]', '',
                              str(tp).split(':')[-1].split('.')[-1])

            if hgvs_lbl == '-':
                sub_lbls += ["(no location)"]
            else:
                sub_lbls += [hgvs_lbl]

        else:
            if mtype.cur_level == 'Exon':
                if len(lbls) == 1:
                    if tuple(lbls)[0] != '-':
                        sub_lbls += ["on {} exon".format(
                            ordinal_label(int(tuple(lbls)[0].split('/')[0])))]
                    else:
                        sub_lbls += ["not on any exon"]

                else:
                    exn_lbls = [ordinal_label(int(lbl.split('/')[0]))
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
                    lbl.replace("_variant", "").replace(',', '/')
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

            else:
                raise ValueError("Unrecognized type of mutation "
                                 "level `{}`!".format(mtype.cur_level))

            if tp is not None:
                if tp.cur_level == 'Consequence':
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
        pnt_link = scale_link
    if phrase_link is None:
        phrase_link = scale_link

    sub_dict = dict(mtype.subtype_list())

    if 'Copy' in sub_dict:
        if sub_dict['Copy'] == MuType({
                ('Copy', ('ShalDel', 'DeepDel')): None}):
            use_lbls = ["any loss"]

        elif sub_dict['Copy'] == MuType({
                ('Copy', ('ShalGain', 'DeepGain')): None}):
            use_lbls = ["any gain"]

        elif sub_dict['Copy'] == MuType({
                ('Copy', ('DeepDel', 'DeepGain')): None}):
            use_lbls = ["any deep loss/gain"]

        elif sub_dict['Copy'] == MuType({
                ('Copy', ('ShalDel', 'ShalGain')): None}):
            use_lbls = ["any shallow loss/gain"]

        elif sub_dict['Copy'] == MuType({('Copy', 'DeepDel'): None}):
            use_lbls = ["deep loss"]
        elif sub_dict['Copy'] == MuType({('Copy', 'DeepGain'): None}):
            use_lbls = ["deep gain"]

        elif sub_dict['Copy'] == MuType({('Copy', 'ShalDel'): None}):
            use_lbls = ["shallow loss"]
        elif sub_dict['Copy'] == MuType({('Copy', 'ShalGain'): None}):
            use_lbls = ["shallow gain"]

        else:
            raise ValueError("Unrecognized alteration `{}`!".format(
                repr(sub_dict['Copy'])))

    else:
        use_lbls = []

    if 'Point' in sub_dict:
        if sub_dict['Point'] is None:
            use_lbls += ["any point mutation"]

        else:
            use_lbls += [nest_label(sub_dict['Point'], pnt_link, phrase_link)]

    return scale_link.join(use_lbls)


def compare_scores(iso_df, samps, muts_dict,
                   get_similarities=True, all_mtype=None):
    base_muts = tuple(muts_dict.values())[0]

    if all_mtype is None:
        all_mtype = MuType(base_muts.allkey())

    pheno_dict = {mtype: np.array(muts_dict[lvls].status(samps, mtype))
                  if lvls in muts_dict
                  else np.array(base_muts.status(samps, mtype))
                  for lvls, mtype in iso_df.index}

    simil_df = pd.DataFrame(0.0, index=pheno_dict.keys(),
                            columns=pheno_dict.keys(), dtype=np.float)
    auc_df = pd.DataFrame(index=pheno_dict.keys(), columns=['All', 'Iso'],
                          dtype=np.float)

    all_pheno = np.array(base_muts.status(samps, all_mtype))
    pheno_dict['Wild-Type'] = ~all_pheno

    for (_, cur_mtype), iso_vals in iso_df.iterrows():
        simil_df.loc[cur_mtype, cur_mtype] = 1.0

        none_vals = np.concatenate(iso_vals[~all_pheno].values)
        wt_vals = np.concatenate(iso_vals[~pheno_dict[cur_mtype]].values)
        cur_vals = np.concatenate(iso_vals[pheno_dict[cur_mtype]].values)

        auc_df.loc[cur_mtype, 'All'] = np.greater.outer(
            cur_vals, wt_vals).mean()
        auc_df.loc[cur_mtype, 'All'] += np.equal.outer(
            cur_vals, wt_vals).mean() / 2

        auc_df.loc[cur_mtype, 'Iso'] = np.greater.outer(
            cur_vals, none_vals).mean()
        auc_df.loc[cur_mtype, 'Iso'] += np.equal.outer(
            cur_vals, none_vals).mean() / 2

        if get_similarities:
            cur_diff = np.subtract.outer(cur_vals, none_vals).mean()

            if cur_diff != 0:
                for other_mtype in set(simil_df.index) - {cur_mtype}:

                    other_vals = np.concatenate(
                        iso_vals[pheno_dict[other_mtype]].values)
                    other_diff = np.subtract.outer(
                        other_vals, none_vals).mean()

                    simil_df.loc[cur_mtype, other_mtype] = other_diff
                    simil_df.loc[cur_mtype, other_mtype] /= cur_diff

    return pheno_dict, auc_df, simil_df


def calc_auc(vals, stat):
    return (np.greater.outer(vals[stat], vals[~stat]).mean()
            + 0.5 * np.equal.outer(vals[stat], vals[~stat]).mean())

