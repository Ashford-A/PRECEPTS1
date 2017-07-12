
import sys
sys.path.extend(['/home/exacloud/lustre1/CompBio/mgrzad/bergamot/'])

from itertools import chain
import glob
import dill as pickle

import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
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

    mutex_dict = pickle.load(open(
        'HetMan/experiments/mutex-variants/output/BRCA/mutex_dict.p',
        'rb'
        ))

    cmap = pltcol.LinearSegmentedColormap.from_list(
            'new_map', [(1, 0, 0), (0, 0, 1)]
            )
    plt_alpha = []
    plt_col = []

    plt_x = []
    plt_y = []

    for mtype1, mtype2 in out_acc.keys():
        acc = min(out_acc[(mtype1, mtype2)])

        if acc > 0.70:
            print('{}  +  {}'.format(mtype1, mtype2))

            sim1 = ((out_stat[(mtype1, mtype2)][0][1]
                     - out_stat[(mtype1, mtype2)][0][0])
                    / (out_stat[(mtype1, mtype2)][0][2]
                       - out_stat[(mtype1, mtype2)][0][0]))
            sim2 = ((out_stat[(mtype1, mtype2)][1][1]
                     - out_stat[(mtype1, mtype2)][1][0])
                    / (out_stat[(mtype1, mtype2)][1][2]
                       - out_stat[(mtype1, mtype2)][1][0]))

            plt_x += [-log10(mutex_dict[(mtype1, mtype2)])]
            plt_y += [(sim1 + sim2) / 2.0]
            plt_alpha += [acc ** 2.0]
            plt_col += [list(out_dist[(mtype1, mtype2)])[0]]
            #plt.annotate(str(mtype1) + str(mtype2), plt_x + 0.02, plt_y + 0.02)

    plt.scatter(plt_x, plt_y,
                alpha=0.5, s=40, c=plt_col, cmap=cmap, vmin=-0.5,
                vmax=0.5)
    plt.xlim([-2.0,2.0])
    plt.savefig('test.png', dpi=2000, bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1:])

