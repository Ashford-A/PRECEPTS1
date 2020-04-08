
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.subvariant_test.utils import ordinal_label
from dryadic.features.mutations import MuType, MutComb

import numpy as np
import pandas as pd

from functools import reduce
from operator import or_
import re


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


def nest_label(mtype, sub_link=' or '):
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
                        dmn_jn = ', '
                    else:
                        dmn_jn = ' '

                    sub_lbls += ["on domains {}".format(
                        dmn_jn.join(dmn_lbls))]

            else:
                raise ValueError("Unrecognized type of mutation "
                                 "level `{}`!".format(mtype.cur_level))

            if tp is not None:
                if tp.cur_level == 'Consequence':
                    sub_lbls[-1] = ' '.join([nest_label(tp), sub_lbls[-1]])
                else:
                    sub_lbls[-1] = ' '.join([sub_lbls[-1], nest_label(tp)])

    return sub_link.join(sub_lbls)


def get_fancy_label(mtype, scale_link=' or ', pnt_link=' or '):
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
            use_lbls += [nest_label(sub_dict['Point'], pnt_link)]

    return scale_link.join(use_lbls)


def get_mtype_gene(mtype):
    mtype_genes = None

    if isinstance(mtype, RandomType):
        if mtype.base_mtype is not None:
            mtype_genes = mtype.base_mtype.get_labels()

    elif isinstance(mtype, ExMcomb):
        mtype_genes = mtype.all_mtype.get_labels()

    elif isinstance(mtype, Mcomb):
        mtype_genes = {mtp.get_labels() for mtp in mtype.mtypes}

    elif isinstance(mtype, MuType):
        mtype_genes = mtype.get_labels()

    else:
        raise ValueError("Cannot retrieve gene for something that is "
                         "not a mutation!")

    if mtype_genes is None:
        mtype_genes = [None]

    if len(mtype_genes) > 1:
        raise ValueError("Cannot retrieve gene for a mutation associated "
                         "with multiple genes!")

    return mtype_genes[0]


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

