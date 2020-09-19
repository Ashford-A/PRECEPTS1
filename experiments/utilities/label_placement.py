
import random
import numpy as np


def check_overlap(bx1, bx2):
    return not (bx1[0, 0] >= bx2[1, 0] or bx2[0, 0] >= bx1[1, 0]
                or bx1[0, 1] <= bx2[1, 1] or bx2[0, 1] <= bx1[1, 1])


def place_scatter_labels(plot_dict, clr_dict, fig, ax, plt_type='pie',
                         plt_lims=None, plc_lims=None,
                         font_size=13, seed=None):
    if plt_lims:
        ax.set_xlim(plt_lims[0])
        ax.set_ylim(plt_lims[1])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not plc_lims:
        plc_lims = (xmin, xmax), (ymin, ymax)

    if seed is not None:
        random.seed(seed)

    adj_trans = lambda x: ax.transData.inverted().transform(x)
    xadj, yadj = adj_trans([1, 1]) - adj_trans([0, 0])
    xgap, ygap = (xmax - xmin) / 173, (ymax - ymin) / 173
    font_adj = font_size / 13
    lbl_pos = {pnt: None for pnt, (_, lbls) in plot_dict.items() if lbls[0]}

    # initialize objects storing where each label will be positioned, and how
    # much space needs to be left around already placed points and labels
    if plt_type == 'pie':
        pnt_gaps = {pnt: (sz / 2 + xgap, sz / 2 + ygap)
                    for pnt, (sz, _) in plot_dict.items()}

    elif plt_type == 'scatter':
        pass
    else:
        raise ValueError("Unrecognized plot type `{}`!".format(plt_type))

    pnt_bxs = {pnt: [np.array([(pnt[0] - xgap, pnt[1] + ygap),
                               (pnt[0] + xgap, pnt[1] - ygap)])]
               for pnt, (xgap, ygap) in pnt_gaps.items()}

    # calculate how much space each label to plot will occupy once placed
    lbl_wdths = {
        pnt: (5.83 * font_adj * xadj * max(len(ln)
                                           for ln in lbls[1].split('\n'))
              if lbls[1] else 11 * font_adj * xadj * len(lbls[0]))
        for pnt, (_, lbls) in plot_dict.items() if lbls[0]
        }

    lbl_hghts = {pnt: (17 * font_adj * yadj
                       * (2.1 + 11 / 19 * lbls[1].count('\n'))
                       if lbls[1] else 17 * font_adj * yadj)
                 for pnt, (_, lbls) in plot_dict.items() if lbls[0]}

    # for each point, check if there is enough space to plot its label
    # to the left of it...
    for pnt in sorted(set(lbl_pos)):
        if ((plc_lims[1][0] + ygap / 2.3) < pnt[1]
                < (plc_lims[1][1] - ygap / 2.3)):

            if (pnt[0] > (plc_lims[0][0] + lbl_wdths[pnt] + pnt_gaps[pnt][0])
                    and not any(check_overlap(
                        pnt_bxs[pnt][0] - np.array([[lbl_wdths[pnt], 0],
                                                    [0, 0]]),
                        pnt_bxs[pnt2][0]
                        ) for pnt2 in plot_dict if pnt2 != pnt)):

                lbl_pos[pnt] = (pnt[0] - pnt_gaps[pnt][0], pnt[1]), 'right'
                pnt_bxs[pnt][0][0, 0] -= lbl_wdths[pnt] + xgap

                pnt_bxs[pnt][0][0, 1] = pnt[1] + max(
                    pnt_gaps[pnt][1], lbl_hghts[pnt] / 1.9)
                pnt_bxs[pnt][0][1, 1] = pnt[1] - max(
                    pnt_gaps[pnt][1], lbl_hghts[pnt] / 1.9)

            # ...if there isn't, check if there is enough space to plot its
            # label to the right of it
            elif (pnt[0] < (plc_lims[0][1]
                            - lbl_wdths[pnt] - pnt_gaps[pnt][1])
                    and not any(check_overlap(
                        pnt_bxs[pnt][0] + np.array([[0, 0],
                                                    [lbl_wdths[pnt], 0]]),
                        pnt_bxs[pnt2][0]
                        ) for pnt2 in plot_dict if pnt2 != pnt)):

                lbl_pos[pnt] = (pnt[0] + pnt_gaps[pnt][0], pnt[1]), 'left'
                pnt_bxs[pnt][0][1, 0] += lbl_wdths[pnt] + xgap

                pnt_bxs[pnt][0][0, 1] = pnt[1] + max(
                    pnt_gaps[pnt][1], lbl_hghts[pnt] / 1.9)
                pnt_bxs[pnt][0][1, 1] = pnt[1] - max(
                    pnt_gaps[pnt][1], lbl_hghts[pnt] / 1.9)

    # for labels that couldn't be placed right beside their points, look for
    # empty space in the vicinity
    i = 0
    while i < 50000 and any(lbl is None for lbl in lbl_pos.values()):
        i += 1

        for pnt, (_, (_, bot_lbl)) in plot_dict.items():
            if (pnt in lbl_pos and lbl_pos[pnt] is None
                    and lbl_wdths[pnt] < (0.91 * (xmax - xmin))
                    and lbl_hghts[pnt] < (0.47 * (ymax - ymin))):
                pos_rands = [random.expovariate((i * adj / 2317) ** -1)
                             for adj in [xadj, yadj]]

                new_pos = [
                    px + (1.9 * pnt_gap + pos_rand) * random.choice([-1, 1])
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
                                     pnt[0] - 331 * xadj, pnt[0] + 331 * xadj)
                new_pos[1] = np.clip(new_pos[1],
                                     pnt[1] - 331 * yadj, pnt[1] + 331 * yadj)

                if bot_lbl:
                    top_prop = (4 / 3 + bot_lbl.count('\n')) ** -1
                else:
                    top_prop = 0.5

                new_bx = np.array(
                    [[new_pos[0] - lbl_wdths[pnt] / 1.9 - xgap,
                      new_pos[1] + lbl_hghts[pnt] * top_prop + ygap],
                     [new_pos[0] + lbl_wdths[pnt] / 1.9 + xgap,
                      new_pos[1] - lbl_hghts[pnt] * (1 - top_prop) - ygap]]
                    )

                if not any(check_overlap(new_bx, pnt2_bx)
                           for pnt2_bxs in pnt_bxs.values()
                           for pnt2_bx in pnt2_bxs):
                    lbl_pos[pnt] = (new_pos[0], new_pos[1]), 'center'
                    pnt_bxs[pnt] += [new_bx]

    lbl_pos = {pnt: lbl for pnt, lbl in lbl_pos.items() if lbl}
    for (pnt_x, pnt_y), ((lbl_x, lbl_y), lbl_ha) in lbl_pos.items():
        if lbl_ha == 'center' and not plot_dict[pnt_x, pnt_y][1][1]:
            ax.text(lbl_x, lbl_y, plot_dict[pnt_x, pnt_y][1][0],
                    size=font_size, ha=lbl_ha, va='center')
        else:
            ax.text(lbl_x, lbl_y + yadj * 2.3, plot_dict[pnt_x, pnt_y][1][0],
                    size=font_size, ha=lbl_ha, va='bottom')

        if plot_dict[pnt_x, pnt_y][1][1]:
            ax.text(lbl_x, lbl_y - yadj * 2.3, plot_dict[pnt_x, pnt_y][1][1],
                    size=font_size / 1.61, ha=lbl_ha, va='top')

        lbl_bx = pnt_bxs[pnt_x, pnt_y][-1]
        txt_y = np.clip(pnt_y, *(lbl_bx[::-1, 1]
                                 + np.array([ygap, -ygap]) / 2.1))

        if pnt_x <= lbl_bx[0, 0]:
            txt_x = lbl_bx[0, 0] + 0.19 * np.diff(lbl_bx[:, 0])[0]
        elif pnt_x >= lbl_bx[1, 0]:
            txt_x = lbl_bx[1, 0] - 0.19 * np.diff(lbl_bx[:, 0])[0]
        else:
            txt_x = np.mean(lbl_bx[:, 0])

        x_delta, y_delta = (pnt_x - txt_x), (pnt_y - txt_y)
        ln_x, ln_y = x_delta / xadj, y_delta / yadj
        ln_mag = (ln_x ** 2 + ln_y ** 2) ** 0.5
        ln_cos, ln_sin = ln_x / ln_mag, ln_y / ln_mag

        crc_x = pnt_x - (pnt_gaps[pnt_x, pnt_y][0] * ln_cos)
        crc_y = pnt_y - (pnt_gaps[pnt_x, pnt_y][1] * ln_sin)
        crc_bx = np.array([[crc_x - xgap / 2.9, crc_y + ygap / 2.9],
                           [crc_x + xgap / 2.9, crc_y - ygap / 2.9]])

        if not check_overlap(lbl_bx, crc_bx):
            ax.plot([crc_x, txt_x], [crc_y, txt_y],
                    c=clr_dict[pnt_x, pnt_y], linewidth=1.7, alpha=0.31)

    return {pos: lbl for pos, lbl in lbl_pos.items() if lbl}

