
import sys
sys.path.extend(['/home/exacloud/lustre1/CompBio/mgrzad/bergamot/'])

from itertools import chain
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import numpy as np
from math import log10

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


def main(argv):

    print("... loading experiment data ...")
    out_list = [
        pickle.load(open(fl, 'rb')) for fl in
        glob.glob("HetMan/experiments/mutex-variants/output/BRCA/ex*")
        ]
    print("... data has been loaded ...")

    out_acc = dict(chain(*map(dict.items, [x['Acc'] for x in out_list])))
    out_stat = dict(chain(*map(dict.items, [x['Stat'] for x in out_list])))
    out_dist = dict(chain(*map(dict.items, [x['Dist'] for x in out_list])))
    out_coef = dict(chain(*map(dict.items, [x['Coef'] for x in out_list])))

    mutex_dict = dict(pickle.load(
        open('HetMan/experiments/mutex-variants/output/BRCA/mutex_dict.p',
             'rb')
        ))

    cmap = pltcol.LinearSegmentedColormap.from_list(
            'new_map', [(1, 0, 0), (0, 0, 1)]
            )
    plt_alpha = []
    plt_col = []

    plt_x = []
    plt_y = []
    plt_mtypes = []

    for mtype1, mtype2 in out_acc:
        acc = min(out_acc[(mtype1, mtype2)])

        if acc > 0.7:
            print('{}  +  {}'.format(mtype1, mtype2))
            print(np.round(out_stat[(mtype1, mtype2)], 3))
            print(mutex_dict[(mtype1, mtype2)])

            sim1 = ((out_stat[(mtype1, mtype2)][0][1]
                     - out_stat[(mtype1, mtype2)][0][0])
                    / (out_stat[(mtype1, mtype2)][0][2]
                       - out_stat[(mtype1, mtype2)][0][0]))
            sim2 = ((out_stat[(mtype1, mtype2)][1][1]
                     - out_stat[(mtype1, mtype2)][1][0])
                    / (out_stat[(mtype1, mtype2)][1][2]
                       - out_stat[(mtype1, mtype2)][1][0]))

            if (-5 < sim1 < 5) and (-5 < sim2 < 5):
                plt_x += [-log10(mutex_dict[(mtype1, mtype2)])]
                plt_y += [(sim1 + sim2) / 2.0]
                plt_alpha += [acc ** 4.0]
                plt_col += [list(out_dist[(mtype1, mtype2)])[0]]
                plt_mtypes += [(mtype1, mtype2)]

            else:
                print('Anomalous pair!')

    plt.scatter(plt_x, plt_y,
                alpha=0.4, s=65,
                c=plt_col, cmap=cmap, vmin=-0.5, vmax=0.5)

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
    
    plt.savefig('test.png', dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1:])

