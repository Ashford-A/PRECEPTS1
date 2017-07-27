
import os
base_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd

import glob
import argparse
import pickle
import re

from itertools import chain, permutations

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
from pylab import rcParams

from HetMan.features.variants import MuType
from sklearn.decomposition import PCA


def load_output(cohort, gene):

    mut_list = [
        [pickle.load(open(fl, 'rb')) for fl in
         glob.glob(os.path.join(
             base_dir, "output", cohort, gene,
             "results/out__cv-{}_task-*").format(cv_id))
         if 'cna.p' not in fl]
        for cv_id in range(5)
        ]

    mut_data = [
        {fld: dict(chain(*map(dict.items,
                              [x[fld] for x in mut_list[cv_id]])))
         for fld in mut_list[cv_id][0].keys()}
        for cv_id in range(5)
        ]

    cna_data = [
        pickle.load(open(os.path.join(
            base_dir, "output", cohort, gene,
            "results/out__cv-{}_task-cna.p".format(cv_id)), 'rb'))
        for cv_id in range(5)
        ]

    return mut_data, cna_data


def get_median_coefs(out_data):
    coef_df = pd.DataFrame(
        {mtype: pd.DataFrame(
            [out_data[cv_id]['Coef'][mtype]
             for cv_id in range(5)]
            ).fillna(0).quantile(q=0.5)
            for mtype in out_data[0]['Acc']}).fillna(0)
    coef_df = coef_df.loc[(coef_df != 0).any(axis=1), :].transpose()

    return coef_df


def choose_point_scheme(pnt_scheme, **scheme_args):
    """Choose the plotting characteristics of an individual point."""

    if pnt_scheme is None:
        plt_clr = 'black'
        plt_size = 50
        plt_alpha = 0.7
        plt_mark = 'o'
        plt_lbl = None

    elif pnt_scheme[0] == 'sub_mtype':
        sub_mtype = pnt_scheme[1]
        gn = [k for k,v in sub_mtype][0]

        if scheme_args['mtype'] == MuType({('Gene', gn): None}):
            plt_clr = '#804B15'
            plt_size = 200
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == sub_mtype:
            plt_clr = '#0E553C'
            plt_size = 190
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'].is_supertype(sub_mtype):
            plt_clr = '#803815'
            plt_size = 27
            plt_alpha = 0.4
            plt_mark = 'o'

        elif sub_mtype.is_supertype(scheme_args['mtype']):
            plt_clr = '#0F414F'
            plt_size = 21
            plt_alpha = 0.4
            plt_mark = 'o'

        else:
            plt_clr = 'black'
            plt_size = 9
            plt_alpha = 0.2
            plt_mark = 'o'

    elif pnt_scheme == 'main_subs':
        gn = [k for k,v in scheme_args['mtype']][0]

        if scheme_args['mtype'] == MuType({('Gene', gn): None}):
            plt_clr = 'black'
            plt_size = 200
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Missense_Mutation'): None}}):
            plt_clr = '#31224A'
            plt_size = 190
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Nonsense_Mutation'): None}}):
            plt_clr = '#466228'
            plt_size = 190
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Frame_Shift'): None}}):
            plt_clr = '#6C4E2C'
            plt_size = 190
            plt_alpha = 0.8
            plt_mark = '*'

        elif MuType({('Gene', gn): {('Form_base', 'Missense_Mutation'): None}})\
                .is_supertype(scheme_args['mtype']):
            plt_clr = '#31224A'
            plt_size = 25
            plt_alpha = 0.4
            plt_mark = 'o'

        elif MuType({('Gene', gn): {('Form_base', 'Nonsense_Mutation'): None}})\
                .is_supertype(scheme_args['mtype']):
            plt_clr = '#466228'
            plt_size = 25
            plt_alpha = 0.4
            plt_mark = 'o'

        elif MuType({('Gene', gn): {('Form_base', 'Frame_Shift'): None}})\
                .is_supertype(scheme_args['mtype']):
            plt_clr = '#6C4E2C'
            plt_size = 25
            plt_alpha = 0.4
            plt_mark = 'o'

        else:
            plt_clr = 'black'
            plt_size = 9
            plt_alpha = 0.2
            plt_mark = 'o'

    return plt_clr, plt_size, plt_alpha, plt_mark


def get_legend_label(pnt_scheme, **scheme_args):

    if pnt_scheme is None:
        plt_lbl = None

    elif pnt_scheme[0] == 'sub_mtype':
        sub_mtype = pnt_scheme[1]
        gn = [k for k,v in sub_mtype][0]

        if scheme_args['mtype'] in [sub_mtype, MuType({('Gene', gn): None})]:
            plt_lbl = str(scheme_args['mtype'])

        elif scheme_args['mtype'].is_supertype(sub_mtype):
            plt_lbl = "Super-types of {0!s}".format(sub_mtype)

        elif sub_mtype.is_supertype(scheme_args['mtype']):
            plt_lbl = "Sub-types of {0!s}".format(sub_mtype)

        else:
            plt_lbl = "Other"

    elif pnt_scheme == 'main_subs':
        gn = [k for k,v in scheme_args['mtype']][0]

        if scheme_args['mtype'] == MuType({('Gene', gn): None}):
            plt_lbl = str(scheme_args['mtype'])

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Missense_Mutation'): None}}):
            plt_lbl = str(scheme_args['mtype'])

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Nonsense_Mutation'): None}}):
            plt_lbl = str(scheme_args['mtype'])

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Frame_Shift'): None}}):
            plt_lbl = str(scheme_args['mtype'])

        elif MuType({('Gene', gn): {('Form_base', 'Missense_Mutation'): None}})\
                .is_supertype(scheme_args['mtype']):
            plt_lbl = "Sub-types of Missense Mutations"

        elif MuType({('Gene', gn): {('Form_base', 'Nonsense_Mutation'): None}})\
                .is_supertype(scheme_args['mtype']):
            plt_lbl = "Sub-types of Nonsense Mutations"

        elif MuType({('Gene', gn): {('Form_base', 'Frame_Shift'): None}})\
                .is_supertype(scheme_args['mtype']):
            plt_lbl = "Sub-types of Frameshift Mutations"

        else:
            plt_lbl = 'Other'

    return plt_lbl


def get_scheme_label(pnt_schemes):
    if pnt_schemes is None:
        clr_lbl = 'none'

    elif pnt_schemes[0][0] == 'sub_mtype':
        clr_lbl = 'sub-{}'.format(str(pnt_schemes[0][1]))

    elif pnt_schemes[0] == 'main_subs':
        clr_lbl = 'main_subs'

    if pnt_schemes[1][0] == 'sub_mtype':
        clr_lbl += '_sub-{}'.format(str(pnt_schemes[1][1]))

    return clr_lbl


def print_coefs(coef_dict, top_n=20):
    coef_str = ""
    for gn, coef in sorted(
            coef_dict.items(),
            key=lambda x: abs(x[1]), reverse=True)[:top_n]:
        coef_str += "{:8}{:+.3f}\n".format(gn, coef)

    return re.sub("\n$", "", coef_str)


def plot_signature_pca(args, out_data, pnt_schemes=None):
    rcParams['figure.figsize'] = 18, 18

    coef_df = get_median_coefs(out_data)
    pca = PCA()
    coef_pca = pd.DataFrame(pca.fit_transform(coef_df), index=coef_df.index)

    f, axarr = plt.subplots(3, 3, sharex='col', sharey='row')

    for pc_a, pc_b in permutations(range(3), 2):
        plt_x = []
        plt_y = []
        plt_lbls = set()

        for mtype in out_data[0]['Acc']:
            acc = np.mean([out_data[cv_id]['Acc'][mtype]
                           for cv_id in range(5)])

            plt_x += [coef_pca.ix[mtype, pc_b]]
            plt_y += [coef_pca.ix[mtype, pc_a]]

            if pc_a > pc_b:
                plt_clr, plt_size, plt_alpha, plt_mark = choose_point_scheme(
                    pnt_schemes[0], mtype=mtype)
                plt_lbl = get_legend_label(pnt_schemes[0], mtype=mtype)

            else:
                plt_clr, plt_size, plt_alpha, plt_mark = choose_point_scheme(
                    pnt_schemes[1], mtype=mtype)
                plt_lbl = get_legend_label(pnt_schemes[1], mtype=mtype)

            if plt_lbl in plt_lbls:
                plt_lbl = None
            else:
                plt_lbls |= {plt_lbl}

            axarr[pc_a, pc_b].scatter(
                plt_x[-1], plt_y[-1], alpha=plt_alpha, s=plt_size,
                c=plt_clr, marker=plt_mark, label=plt_lbl
                )

        if abs(pc_a - pc_b) == 2:
            leg_adj = np.sign(pc_a - pc_b)
            handles, labels = axarr[pc_a, pc_b].get_legend_handles_labels()

            sort_hand, sort_lbl = zip(*sorted(
                zip(handles, labels),
                key=lambda x: x[0].get_sizes()
                )[::-1])

            axarr[pc_a, pc_b].legend(
                sort_hand, sort_lbl,
                bbox_to_anchor=(0.5 - 0.5 * leg_adj,
                                0.5 - 0.55 * leg_adj),
                loc=(3 - leg_adj), borderaxespad=0., markerscale=1.8, ncol=3,
                labelspacing=1.4, borderpad=1.2, fontsize=16
                )

    for pc in range(3):

        axarr[pc, pc].text(
            0.12, 0.93, "PC {}".format(str(pc + 1)),
            fontsize=30, ha='center', va='center',
            transform=axarr[pc, pc].transAxes
            )

        axarr[pc, pc].text(
            0.65, 0.93,
            "Variance: {:.1%}".format(pca.explained_variance_ratio_[pc]),
            fontsize=30, ha='center', va='center',
            transform=axarr[pc, pc].transAxes
            )

        axarr[pc, pc].text(
            0.06, 0.83,
            print_coefs(dict(zip(coef_df.columns, pca.components_[pc, :]))),
            fontsize=14, transform=axarr[pc, pc].transAxes,
            ha='left', va='top'
            )

    f.tight_layout()
    plt.savefig(
        os.path.join(
            base_dir, 'plots',
            '{}-{}_signature-pca_{}.png'.format(
                args.cohort, args.gene, get_scheme_label(pnt_schemes))
            ),
        dpi=500, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('-c', '--cohort')
    parser.add_argument('-g', '--gene')
    args = parser.parse_args()

    out_data = load_output(args.cohort)
    plot_signature_pca(
        args, out_data,
        pnt_schemes=(('sub_mtype', MuType({('Gene', 'TP53'): {
                         ('Form', 'Missense_Mutation'): None}})),
                     ('sub_mtype', MuType({('Gene', 'TP53'): {
                         ('Form', 'Nonsense_Mutation'): None}}))
                    )
        )

if __name__ == '__main__':
    main()

