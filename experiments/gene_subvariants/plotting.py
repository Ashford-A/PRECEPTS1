
import os
base_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

import numpy as np
import pandas as pd

import glob
import argparse
import pickle
import re
import colorsys

from itertools import chain, permutations

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
from pylab import rcParams

from HetMan.features.variants import MuType
from sklearn.decomposition import PCA


def load_output(cohort, gene, classif):

    mut_list = [
        [pickle.load(open(fl, 'rb')) for fl in
         glob.glob(os.path.join(
             base_dir, "output", cohort, gene, classif,
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
            base_dir, "output", cohort, gene, classif,
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


def get_overlap_stat(out_data, mtype1, mtype2):
    stat1 = [set(dt['Stat'][mtype1]) for dt in out_data]
    stat2 = [set(dt['Stat'][mtype2]) for dt in out_data]
    stat_ov = [len(s1 & s2) for s1, s2 in zip(stat1, stat2)]

    sub_indx = np.mean([ov / len(s1) for ov, s1 in zip(stat_ov, stat1)])
    sup_indx = np.mean([ov / len(s2) for ov, s2 in zip(stat_ov, stat2)])

    return sub_indx, sup_indx


def choose_point_scheme(pnt_scheme, **scheme_args):
    """Choose the plotting characteristics of an individual point."""

    if pnt_scheme is None:
        plt_clr = 'black'
        plt_size = 20
        plt_alpha = 0.4
        plt_mark = 'o'
        plt_lbl = None
        plt_edge = 'none'

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
            plt_size = 8
            plt_alpha = 0.16
            plt_mark = 'o'

    elif pnt_scheme[0] == 'ovlp_mtype':
        ovlp_mtype = pnt_scheme[1]
        plt_edge = 'black'


        if ovlp_mtype == scheme_args['mtype']:
            plt_size = 260
            plt_mark = '*'
            plt_clr = 'black'
            plt_alpha = 0.7

        else:
            plt_mark = 'o'
            sub_indx, sup_indx = get_overlap_stat(
                scheme_args['out_data'], ovlp_mtype, scheme_args['mtype'])

            plt_size = 7 + 15 * ((sub_indx + sup_indx) ** 2)
            clr_h = 300 + (sup_indx - sub_indx) * 100
            clr_s = abs(sub_indx - sup_indx) ** (1 - (sub_indx + sup_indx) / 2)
            clr_v = 1 - ((sub_indx + sup_indx) / 2) ** 2
            plt_alpha = (sub_indx + sup_indx) / 4 + 0.16
            plt_clr = colorsys.hsv_to_rgb(clr_h / 360, clr_s, clr_v)

    elif pnt_scheme == 'main_subs':
        gn = [k for k,v in scheme_args['mtype']][0]
        plt_edge = 'none'
        plt_size = 260

        if scheme_args['mtype'] == MuType({('Gene', gn): None}):
            plt_clr = 'black'
            plt_alpha = 0.8
            plt_mark = 'h'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Missense_Mutation'): None}}):
            plt_clr = '#2F35A9'
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Nonsense_Mutation'): None}}):
            plt_clr = '#F35B27'
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Frame_Shift'): None}}):
            plt_clr = '#7D7203'
            plt_alpha = 0.8
            plt_mark = '*'

        elif scheme_args['mtype'] == MuType(
                {('Gene', gn): {('Form_base', 'Splice_Site'): None}}):
            plt_clr = '#156711'
            plt_alpha = 0.8
            plt_mark = '*'

#        elif MuType({('Gene', gn): {('Form_base', 'Missense_Mutation'): None}})\
#                .is_supertype(scheme_args['mtype']):
#            plt_clr = '#31224A'
#            plt_size = 25
#            plt_alpha = 0.4
#            plt_mark = 'o'

#        elif MuType({('Gene', gn): {('Form_base', 'Nonsense_Mutation'): None}})\
#                .is_supertype(scheme_args['mtype']):
#            plt_clr = '#466228'
#            plt_size = 25
#            plt_alpha = 0.4
#            plt_mark = 'o'

#        elif MuType({('Gene', gn): {('Form_base', 'Frame_Shift'): None}})\
#                .is_supertype(scheme_args['mtype']):
#            plt_clr = '#6C4E2C'
#            plt_size = 25
#            plt_alpha = 0.4
#            plt_mark = 'o'

        else:
            plt_clr = 'black'
            plt_size = 12
            plt_alpha = 0.2
            plt_mark = 'o'

    return plt_clr, plt_size, plt_alpha, plt_mark, plt_edge


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

        if scheme_args['mtype'] in [
                MuType({('Gene', gn): None}),
                MuType({('Gene', gn): {
                    ('Form_base', 'Missense_Mutation'): None}}),
                MuType({('Gene', gn): {
                    ('Form_base', 'Nonsense_Mutation'): None}}),
                MuType({('Gene', gn): {
                    ('Form_base', 'Frame_Shift'): None}}),
                MuType({('Gene', gn): {
                    ('Form_base', 'Splice_Site'): None}}),
                ]:
            plt_lbl = str(scheme_args['mtype'])

        else:
            plt_lbl = 'Other'

    elif pnt_scheme[0] == 'ovlp_mtype':
        if scheme_args['mtype'] == pnt_scheme[1]:
            plt_lbl = str(scheme_args['mtype'])
        else:
            plt_lbl = None

    return plt_lbl


def get_scheme_label(pnt_scheme):

    if pnt_scheme is None:
        clr_lbl = 'none'

    elif pnt_scheme == 'main_subs':
        clr_lbl = 'main-subs'

    elif pnt_scheme[0] == 'sub_mtype':
        clr_lbl = 'sub-{}'.format(str(pnt_scheme[1]))

    elif pnt_scheme[0] == 'ovlp_mtype':
        clr_lbl = 'ovlp-{}'.format(pnt_scheme[1])

    return clr_lbl.replace("/", "_")


def print_coefs(coef_dict, top_n=20):
    coef_str = ""
    for gn, coef in sorted(
            coef_dict.items(),
            key=lambda x: abs(x[1]), reverse=True)[:top_n]:
        coef_str += "{:10}{:+.3f}\n".format(gn, coef)

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
                use_scheme = pnt_schemes[0]
            else:
                use_scheme = pnt_schemes[1]
                
            (plt_clr, plt_size, plt_alpha,
             plt_mark, plt_edge) = choose_point_scheme(
                 use_scheme, mtype=mtype, out_data=out_data)
            plt_lbl = get_legend_label(use_scheme, mtype=mtype)

            if plt_lbl in plt_lbls:
                plt_lbl = None
            else:
                plt_lbls |= {plt_lbl}

            axarr[pc_a, pc_b].scatter(
                plt_x[-1], plt_y[-1], alpha=plt_alpha, s=plt_size,
                c=plt_clr, marker=plt_mark, label=plt_lbl, edgecolor=plt_edge
                )

        if abs(pc_a - pc_b) == 2:
            leg_adj = np.sign(pc_a - pc_b)

            leg_obj = axarr[pc_a, pc_b].get_legend_handles_labels()
            if len(leg_obj[0]) > 0:
                handles, labels = leg_obj

                sort_hand, sort_lbl = zip(*sorted(
                    zip(handles, labels),
                    key=lambda x: x[0].get_sizes()
                    )[::-1])

                axarr[pc_a, pc_b].legend(
                    sort_hand, sort_lbl,
                    bbox_to_anchor=(0.5 - 0.5 * leg_adj,
                                    0.5 - 0.55 * leg_adj),
                    loc=(3 - leg_adj), borderaxespad=0., markerscale=1.8,
                    ncol=2, labelspacing=1.4, borderpad=1.2, fontsize=16
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
            '{}-{}-{}__signature-pca__{}_{}.png'.format(
                args.cohort, args.gene, args.classif,
                get_scheme_label(pnt_schemes[0]),
                get_scheme_label(pnt_schemes[1])
                )
            ),
        dpi=500, bbox_inches='tight'
        )

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('-c', '--cohort')
    parser.add_argument('-g', '--gene')
    parser.add_argument('-p', '--classif')
    args = parser.parse_args()

    mut_data, cna_data = load_output(args.cohort, args.gene, args.classif)
    plot_signature_pca(
        args, mut_data,
        pnt_schemes=(('sub_mtype', MuType({('Gene', 'TP53'): {
                         ('Form_base', 'Missense_Mutation'): None}})),
                     ('sub_mtype', MuType({('Gene', 'TP53'): {
                         ('Form_base', 'Nonsense_Mutation'): None}}))
                    )
        )
    plot_signature_pca(
        args, mut_data,
        pnt_schemes=('main_subs',
                     ('sub_mtype', MuType({('Gene', 'TP53'): {
                         ('Form_base', 'Frame_Shift'): None}}))
                    )
        )


if __name__ == '__main__':
    main()

