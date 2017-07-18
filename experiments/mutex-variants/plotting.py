
import os
base_dir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from itertools import chain
import glob
import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
from pylab import rcParams

import numpy as np
from math import log10

import matplotlib as mpl
import matplotlib.cm as cm


def load_output(cohort):
    out_list = [
        pickle.load(open(fl, 'rb')) for fl in
        glob.glob(os.path.join(base_dir, "output", cohort, "results/ex*"))
        ]

    out_data = {fld: dict(chain(*map(dict.items, [x[fld] for x in out_list])))
                for fld in out_list[0].keys()}

    return out_data


def choose_colour_scheme(clr_scheme, **clr_args):

    if clr_scheme == 'Dist':
        norm = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
        cmap = pltcol.LinearSegmentedColormap.from_list(
                'new_map', [(1, 0, 0), (0, 0, 1)]
                )
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        plt_clr = m.to_rgba(list(
            clr_args['out_data']['Dist'][clr_args['mtype1'],
                                         clr_args['mtype2']]
            )[0])
    
    elif clr_scheme[0] == 'Gene':
        if clr_args['gn1'] == clr_scheme[1] or clr_args['gn2'] == clr_scheme[1]:
            plt_clr = '#172457'
        else:
            plt_clr = '#806015'

    return plt_clr


def get_colour_label(clr_scheme):
    if clr_scheme == 'Dist':
        clr_lbl = 'dist'
    else:
        clr_lbl = 'gene'

    return clr_lbl


def plot_predicted_means(cohort, mutex_dict, clr_scheme='Dist'):
    rcParams['figure.figsize'] = 15, 15

    cmap = pltcol.LinearSegmentedColormap.from_list(
            'new_map', [(1, 0, 0), (0, 0, 1)]
            )

    plt_x = []
    plt_y = []
    plt_mtypes = []

    for mtype1, mtype2 in out_data['Acc']:
        acc = min(out_data['Acc'][mtype1, mtype2])

        if acc > 0.7:
            plt_mtypes += [(mtype1, mtype2)]
            gn1 = [k for k,v in mtype1][0]
            gn2 = [k for k,v in mtype2][0]

            plt_x += [out_data['Stat'][mtype1, mtype2][0][0]]
            plt_y += [out_data['Stat'][mtype1, mtype2][0][2]]
            plt_alpha = acc ** 4.0
            plt_clr = choose_colour_scheme(clr_scheme)

            if gn1 == gn2:
                plt_mark = 'o'
            else:
                plt_mark = 's'

            plt.scatter(plt_x[-1], plt_y[-1],
                        alpha=plt_alpha, s=65, c=plt_col, marker=plt_mark)

    annot_enum = enumerate(zip(plt_x, plt_y, plt_mtypes))
    for i, (x, y, (mtype1, mtype2)) in annot_enum:

        if all(x < (xs - 1) or x > xs or y < (ys - 0.1) or y > (ys + 0.1)
               for xs, ys in zip(
                   plt_x[:i] + plt_x[i+1:], plt_y[:i] + plt_y[i+1:]
                   )):
            plt.annotate('{}|{}'.format(str(mtype1), str(mtype2)),
                         (x, y), (x, y + 0.06),
                         size="small", stretch="condensed")

        elif (x > 1
              and all(x < xs or x > (xs + 5.0/7)
                      or y < (ys - 0.1) or y > (ys + 0.1)
                      for xs, ys in zip(plt_x[:i] + plt_x[i+1:],
                                        plt_y[:i] + plt_y[i+1:]))
                ):
            plt.annotate('{}|{}'.format(str(mtype1), str(mtype2)),
                         (x, y), (x-0.5, y + 0.06),
                         size="small", stretch="condensed")

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0,1], linewidth=3, color = 'black', ls='--', alpha=0.5)
    
    plt.savefig(os.path.join(base_dir, 'plots',
                             cohort + '_mutex-preds_' + clr_lbl + '.png'),
                dpi=500, bbox_inches='tight')
    plt.close()


def plot_mutex_signatures(args, out_data, mutex_dict, clr_scheme='Dist'):
    rcParams['figure.figsize'] = 20, 10


    plt_x = []
    plt_y = []
    plt_mtypes = []

    for mtype1, mtype2 in out_data['Acc']:
        acc = min(out_data['Acc'][mtype1, mtype2])

        if acc > 0.7:
            print('{}  +  {}'.format(mtype1, mtype2))
            print(np.round(out_data['Stat'][mtype1, mtype2], 3))
            print(mutex_dict[mtype1, mtype2])

            sim1 = ((out_data['Stat'][mtype1, mtype2][0][1]
                     - out_data['Stat'][mtype1, mtype2][0][0])
                    / (out_data['Stat'][mtype1, mtype2][0][2]
                       - out_data['Stat'][mtype1, mtype2][0][0]))
            sim2 = ((out_data['Stat'][mtype1, mtype2][1][1]
                     - out_data['Stat'][mtype1, mtype2][1][0])
                    / (out_data['Stat'][mtype1, mtype2][1][2]
                       - out_data['Stat'][mtype1, mtype2][1][0]))

            if (-5 < sim1 < 5) and (-5 < sim2 < 5):
                plt_mtypes += [(mtype1, mtype2)]
                gn1 = [k for k,v in mtype1][0]
                gn2 = [k for k,v in mtype2][0]

                plt_x += [-log10(mutex_dict[mtype1, mtype2])]
                plt_y += [(sim1 + sim2) / 2.0]

                plt_alpha = acc ** 4.0
                plt_clr = choose_colour_scheme(
                    clr_scheme, out_data=out_data, gn1=gn1, gn2=gn2,
                    mtype1=mtype1, mtype2=mtype2
                    )

                if gn1 == gn2:
                    plt_mark = 'o'
                else:
                    plt_mark = 's'

                plt.scatter(plt_x[-1], plt_y[-1],
                            alpha=plt_alpha, s=65, c=plt_clr, marker=plt_mark)

            else:
                print('Anomalous pair!')

    annot_enum = enumerate(zip(plt_x, plt_y, plt_mtypes))
    for i, (x, y, (mtype1, mtype2)) in annot_enum:

        if all(x < (xs - 1) or x > xs or y < (ys - 0.1) or y > (ys + 0.1)
               for xs, ys in zip(
                   plt_x[:i] + plt_x[i+1:], plt_y[:i] + plt_y[i+1:]
                   )):
            plt.annotate('{}|{}'.format(str(mtype1), str(mtype2)),
                         (x, y), (x, y + 0.06),
                         size="small", stretch="condensed")

        elif (x > 1
              and all(x < xs or x > (xs + 5.0/7)
                      or y < (ys - 0.1) or y > (ys + 0.1)
                      for xs, ys in zip(plt_x[:i] + plt_x[i+1:],
                                        plt_y[:i] + plt_y[i+1:]))
                ):
            plt.annotate('{}|{}'.format(str(mtype1), str(mtype2)),
                         (x, y), (x-0.5, y + 0.06),
                         size="small", stretch="condensed")

    plt.axhline(y=0, xmin=0, xmax=100,
                linewidth=3, color = 'black', ls='--', alpha=0.5)
    plt.axhline(y=1, xmin=0, xmax=100,
                linewidth=3, color = 'black', ls='--', alpha=0.5)

    clr_lbl = get_colour_label(clr_scheme)
    plt.savefig(
        os.path.join(base_dir, 'plots',
                     args.cohort + '_mutex-sigs_' + clr_lbl + '.png'),
        dpi=500, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process plotting options.')
    parser.add_argument('-c', '--cohort')
    args = parser.parse_args()

    out_data = load_output(args.cohort)
    mutex_dict = dict(pickle.load(
        open(os.path.join(base_dir,
                          'output', args.cohort, 'tmp/mutex_dict.p'),
             'rb')
        ))

    plot_mutex_signatures(args, out_data, mutex_dict)


if __name__ == '__main__':
    main()

