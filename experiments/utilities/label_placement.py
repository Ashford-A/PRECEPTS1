
import random
import numpy as np


def place_scatterpie_labels(plot_dict, clr_dict, fig, ax, plt_lims=None,
                            lbl_dens=1., font_adj=1., seed=None):
    if plt_lims:
        ax.set_xlim(plt_lims[0])
        ax.set_ylim(plt_lims[1])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if seed is not None:
        random.seed(seed)

    dist_trans = lambda x: (
        ax.transData.inverted().transform(fig.transFigure.transform(x)))
    adj_trans = lambda x: ax.transData.inverted().transform(x)

    #TODO: does density only apply to freely-placed labels?
    xdist, ydist = 0.31 * (dist_trans([1, 1]) - dist_trans([0, 0]))
    xadj, yadj = 9.7 * (adj_trans([1, 1]) - adj_trans([0, 0]))

    # initialize objects storing where each label will be positioned, and how
    # much space needs to be left around already placed points and labels
    lbl_pos = {pnt: None for pnt, (_, lbls) in plot_dict.items() if lbls[0]}
    pnt_gaps = {pnt: (7 * xadj * (sz + 0.053), 7 * yadj * sz)
                for pnt, (sz, _) in plot_dict.items()}
    pnt_boxs = {pnt: [[xgap * 1.07, xgap * 1.07], [ygap * 1.07, ygap * 1.07]]
                for pnt, (xgap, ygap) in pnt_gaps.items()}

    # calculate how much space each label to plot will occupy once placed
    lbl_wdths = {
        pnt: (font_adj * xadj * max(len(ln) for ln in lbls[1].split('\n'))
              if lbls[1] else font_adj * xadj * len(lbls[0]) * 2.1)
        for pnt, (_, lbls) in plot_dict.items()
        }

    lbl_hghts = {
        pnt: (font_adj * yadj * (4 + lbls[1].count('\n'))
              if lbls[1] else (font_adj * yadj) / 1.3)
        for pnt, (_, lbls) in plot_dict.items()
        }

    # for each point, check if there is enough space to plot its label
    # to the left of it...
    for pnt in sorted(set(lbl_pos)):
        #TODO: clean this up
        if (pnt[0] > (xmin + lbl_wdths[pnt])
            and pnt[1] < (ymax - lbl_hghts[pnt])
            and not any((((pnt[0] - pnt_boxs[pnt][0][0] - lbl_wdths[pnt])
                          < (pnt2[0] - pnt_boxs[pnt2][0][0])
                          < (pnt[0] + pnt_boxs[pnt][0][1]))
                         or ((pnt[0] - pnt_boxs[pnt][0][0] - lbl_wdths[pnt])
                             < (pnt2[0] + pnt_boxs[pnt2][0][1])
                             < (pnt[0] + pnt_boxs[pnt][0][1]))
                         or ((pnt2[0] - pnt_boxs[pnt2][0][0])
                             < pnt[0] < (pnt2[0] + pnt_boxs[pnt2][0][1])))

                        and (((pnt[1] - pnt_boxs[pnt][1][0]
                               - lbl_hghts[pnt] / 1.9)
                              < (pnt2[1] - pnt_boxs[pnt2][1][0])
                              < (pnt[1] + pnt_boxs[pnt][1][1]
                                 + lbl_hghts[pnt] / 2.1))
                             or ((pnt[1] - pnt_boxs[pnt][1][0]
                                  - lbl_hghts[pnt] / 1.9)
                                 < (pnt2[1] + pnt_boxs[pnt2][1][1])
                                 < (pnt[1] + pnt_boxs[pnt][1][1]))
                             or ((pnt2[1] - pnt_boxs[pnt2][1][0])
                                 < pnt[1] < (pnt2[1] + pnt_boxs[pnt2][1][1])))
                        for pnt2 in plot_dict if pnt2 != pnt)):
 
            lbl_pos[pnt] = (pnt[0] - pnt_gaps[pnt][0], pnt[1]), 'right'
            pnt_boxs[pnt][0][0] = pnt_boxs[pnt][0][0] + lbl_wdths[pnt]

            pnt_boxs[pnt][1][0] = max(pnt_boxs[pnt][1][0], lbl_hghts[pnt])
            pnt_boxs[pnt][1][1] = max(pnt_boxs[pnt][1][1],
                                      (font_adj * yadj) / 407)

        # ...if there isn't, check if there is enough space to plot its
        # label to the right of it
        elif (pnt[0] < (xmax - lbl_wdths[pnt])
              and pnt[1] < (ymax - lbl_hghts[pnt])
              and not any((((pnt[0] - pnt_boxs[pnt][0][0])
                            < (pnt2[0] - pnt_boxs[pnt2][0][0])
                            < (pnt[0] + pnt_boxs[pnt][0][1] + lbl_wdths[pnt]))
                           or ((pnt[0] - pnt_boxs[pnt][0][0])
                               < (pnt2[0] + pnt_boxs[pnt2][0][1])
                               < (pnt[0] + pnt_boxs[pnt][0][1]
                                  + lbl_wdths[pnt]))
                           or ((pnt2[0] - pnt_boxs[pnt2][0][0])
                               < pnt[0] < (pnt2[0] + pnt_boxs[pnt2][0][1])))

                          and (((pnt[1] - pnt_boxs[pnt][1][0]
                                 - lbl_hghts[pnt] / 1.9)
                                < (pnt2[1] - pnt_boxs[pnt2][1][0])
                                < (pnt[1] + pnt_boxs[pnt][1][1]
                                   + lbl_hghts[pnt] / 2.1))
                               or ((pnt[1] - pnt_boxs[pnt][1][0])
                                   < (pnt2[1] + pnt_boxs[pnt2][1][1])
                                   < (pnt[1] + pnt_boxs[pnt][1][1]
                                      + lbl_hghts[pnt] / 2.1))
                               or ((pnt2[1] - pnt_boxs[pnt2][1][0])
                                   < pnt[1]
                                   < (pnt2[1] + pnt_boxs[pnt2][1][1])))
                          for pnt2 in plot_dict if pnt2 != pnt)):

            lbl_pos[pnt] = (pnt[0] + pnt_gaps[pnt][0], pnt[1]), 'left'
            pnt_boxs[pnt][0][1] = pnt_boxs[pnt][0][1] + lbl_wdths[pnt]

            pnt_boxs[pnt][1][0] = max(pnt_boxs[pnt][1][0], lbl_hghts[pnt])
            pnt_boxs[pnt][1][1] = max(pnt_boxs[pnt][1][1],
                                      (font_adj * yadj) / 407)

    # for labels that couldn't be placed right beside their points, look for
    # empty space in the vicinity
    i = 0
    while i < 50000 and any(lbl is None for lbl in lbl_pos.values()):
        i += 1

        for pnt in tuple(plot_dict):
            if (pnt in lbl_pos and lbl_pos[pnt] is None
                    and lbl_wdths[pnt] < (0.91 * (xmax - xmin))
                    and lbl_hghts[pnt] < (0.47 * (ymax - ymin))):
                pos_rands = [random.expovariate((i * dist / 50000) ** -1)
                             for dist in [xdist, ydist]]

                new_pos = [
                    px + (pnt_gap + pos_rand) * random.choice([-1, 1])
                    for px, pos_rand, pnt_gap in zip(pnt, pos_rands,
                                                     pnt_gaps[pnt])
                    ]

                # exclude areas too close to the edge of the plot from the
                # vicinity to search over for the label
                new_pos[0] = np.clip(new_pos[0],
                                     xmin + lbl_wdths[pnt] * 0.53,
                                     xmax - lbl_wdths[pnt] * 0.53)
                new_pos[1] = np.clip(new_pos[1],
                                     ymin + lbl_hghts[pnt] * 1.61,
                                     ymax - lbl_hghts[pnt] * 2.61)

                # exclude areas too far away from the original point
                new_pos[0] = np.clip(new_pos[0],
                                     pnt[0] - xdist, pnt[0] + xdist)
                new_pos[1] = np.clip(new_pos[1],
                                     pnt[1] - ydist, pnt[1] + ydist)

                if not (any((((new_pos[0] - lbl_wdths[pnt] / 1.6)
                              < (pnt2[0] - pnt_boxs[pnt2][0][0])
                              < (new_pos[0] + lbl_wdths[pnt] / 1.6))
                             or ((new_pos[0] - lbl_wdths[pnt] / 1.6)
                                 < (pnt2[0] + pnt_boxs[pnt2][0][1])
                                 < (new_pos[0] + lbl_wdths[pnt] / 1.6))
                             or ((pnt2[0] - pnt_boxs[pnt2][0][0])
                                 < new_pos[0]
                                 < (pnt2[0] + pnt_boxs[pnt2][0][1])))

                            and (((new_pos[1] - lbl_hghts[pnt] / 1.4)
                                  < (pnt2[1] - pnt_boxs[pnt2][1][0])
                                  < (new_pos[1] + lbl_hghts[pnt] / 1.4))
                                 or ((new_pos[1] - lbl_hghts[pnt] / 1.4)
                                      < (pnt2[1] + pnt_boxs[pnt2][1][1])
                                      < (new_pos[1] + lbl_hghts[pnt] / 1.4))
                                 or ((pnt2[1] - pnt_boxs[pnt2][1][0])
                                     < new_pos[1]
                                     < (pnt2[1] + pnt_boxs[pnt2][1][1])))
                            for pnt2 in plot_dict)

                        or any(((new_pos[0] - pos2[0][0]) ** 2
                                + (new_pos[1] - pos2[0][1]) ** 2) ** 0.5
                               < (1.9 * (lbl_dens ** -0.5)
                                  * (xadj * yadj) ** 0.5)
                               for pos2 in lbl_pos.values()
                               if pos2 is not None)

                        or any(((abs(new_pos[0] - pos2[0][0])
                                 < (lbl_wdths[pnt] + lbl_wdths[pnt2]) / 1.9))
                                and (abs(new_pos[1] - pos2[0][1])
                                     < (lbl_hghts[pnt] + lbl_hghts[pnt2]))
                               for pnt2, pos2 in lbl_pos.items()
                               if pos2 is not None and pos2[1] == 'center')):

                    lbl_pos[pnt] = (new_pos[0], new_pos[1]), 'center'

    for (pnt_x, pnt_y), pos in lbl_pos.items():
        ax.text(pos[0][0], pos[0][1] + 700 ** -1,
                plot_dict[pnt_x, pnt_y][1][0],
                size=11, ha=pos[1], va='bottom')

        x_delta = pnt_x - pos[0][0]
        y_delta = pnt_y - pos[0][1]

        if abs(x_delta) > xdist / 8.53 or abs(y_delta) > ydist / 8.53:
            end_x = pos[0][0] + np.sign(x_delta) * xdist / 79
            end_y = pos[0][1] + np.heaviside(y_delta, 0) * ydist / 10.1

            ln_x, ln_y = (pnt_x - end_x) / xdist, (pnt_y - end_y) / ydist
            ln_mag = (ln_x ** 2 + ln_y ** 2) ** 0.5
            ln_cos, ln_sin = ln_x / ln_mag, ln_y / ln_mag

            ax.plot([pnt_x - ln_cos * xdist / 4.1
                     * plot_dict[pnt_x, pnt_y][0], end_x],
                    [pnt_y - ln_sin * ydist / 4.1
                     * plot_dict[pnt_x, pnt_y][0], end_y],
                    c=clr_dict[pnt_x, pnt_y], linewidth=1.7, alpha=0.31)

    return {pos: lbl for pos, lbl in lbl_pos.items() if lbl}

