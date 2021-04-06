"""Algorithms for placing plot labels in an aesthetically pleasing manner."""

import random
import numpy as np


def check_overlap(bx1, bx2):
    """Determines if two bounding boxes overlap.

    Args:
        bx1, bx2 (numpy arrays of shape (2, 2))
            These boxes take the form [[xmin, ymin],
                                       [xmax, ymax]]

    """

    return (bx1[0, 0] < bx2[1, 0] and bx2[0, 0] < bx1[1, 0]
            and bx1[0, 1] < bx2[1, 1] and bx2[0, 1] < bx1[1, 1])


# TODO: consolidate and clean up how these keyword arguments are implemented
def place_scatter_labels(plot_dict, ax, plt_lims=None,
                         plc_lims=None, font_size=13, seed=None,
                         font_dict=None, line_dict=None, **line_args):
    """Places two-part labels on a scatter-like plot without collisions.

    Arguments:
        plot_dict (dict)
            The data points and the labels, where:
                - keys are (x, y) data coordinate tuples
                - values are a list where the first element is the plot size
                  in data units and the second element is a list of length
                  two containing the labels, the second of which can be ''

    """

    # fixes plot limits if they are given
    if plt_lims:
        ax.set_xlim(plt_lims[0])
        ax.set_ylim(plt_lims[1])

    # gets plotting limits, sets limits on where to place labels if not given
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not plc_lims:
        plc_lims = (xmin, xmax), (ymin, ymax)

    if seed is not None:
        random.seed(seed)

    # sets default plotting parameters for connecting lines if not given
    if not line_args:
        line_args = dict(c='black', linewidth=1.61, alpha=0.31)

    # gets constants governing how distances between labels and other plotted
    # objects are translated into the plot coordinates
    adj_trans = lambda x: ax.transData.inverted().transform(x)
    xadj, yadj = adj_trans([1, 1]) - adj_trans([0, 0])
    xgap, ygap = xadj * 6.1, yadj * 6.1
    font_adj = font_size / 13

    # initialize object storing how much space needs to be left around already
    # placed points and labels
    pnt_gaps = {pnt: (sz / 2 + xgap, sz / 2 + ygap)
                for pnt, (sz, _) in plot_dict.items()}

    # initialize objects storing where each label will be positioned,
    lbl_pos = {pnt: None for pnt, (_, lbls) in plot_dict.items() if lbls[0]}
    pnt_bxs = {pnt: [np.array([(pnt[0] - xg, pnt[1] - yg),
                               (pnt[0] + xg, pnt[1] + yg)])]
               for pnt, (xg, yg) in pnt_gaps.items()}

    # calculate how much space each label to plot will occupy once placed
    lbl_wdths = {
        pnt: (font_adj * xadj
              * max(5.83 * max(len(ln) for ln in lbls[1].split('\n')),
                    11 * max(len(ln) for ln in lbls[0].split('\n'))))
        for pnt, (_, lbls) in plot_dict.items() if lbls[0]
        }

    lbl_hghts = {pnt: (19 * font_adj * yadj
                       * (2.3 + 11 / 19 * lbls[1].count('\n'))
                       if lbls[1] else (17 * font_adj * yadj
                                        * (1.1 + lbls[0].count('\n'))))
                 for pnt, (_, lbls) in plot_dict.items() if lbls[0]}

    # for each point, check if there is enough space to plot its label
    # to the left of it...
    for pnt in sorted(set(lbl_pos)):
        if ((plc_lims[1][0] + ygap / 2.3) < pnt[1]
                < (plc_lims[1][1] - ygap / 2.3)):
            placed = True

            if (pnt[0] > (plc_lims[0][0] + lbl_wdths[pnt] + pnt_gaps[pnt][0])
                    and not any(check_overlap(
                        pnt_bxs[pnt][0] - np.array([[lbl_wdths[pnt], 0],
                                                    [0, 0]]),
                        pnt_bxs[pnt2][0]
                        ) for pnt2 in plot_dict if pnt2 != pnt)):

                # if there is space, create a label location entry and update
                # the amount of space needed to be left empty around the point
                lbl_pos[pnt] = (pnt[0] - pnt_gaps[pnt][0], pnt[1]), 'right'
                pnt_bxs[pnt][0][0, 0] -= lbl_wdths[pnt] + xgap

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

            else:
                placed = False

            if placed:
                pnt_bxs[pnt][0][0, 1] = pnt[1] - max(pnt_gaps[pnt][1],
                                                     lbl_hghts[pnt] / 1.9)
                pnt_bxs[pnt][0][1, 1] = pnt[1] + max(pnt_gaps[pnt][1],
                                                     lbl_hghts[pnt] / 1.9)

    # for labels that couldn't be placed right beside their points, look for
    # empty space in a vicinity determined using simulated annealing
    i = 0
    while i < 65000 and any(lbl is None for lbl in lbl_pos.values()):
        i += 1

        for pnt, (_, (_, bot_lbl)) in plot_dict.items():
            if (pnt in lbl_pos and lbl_pos[pnt] is None
                    and lbl_wdths[pnt] < (0.91 * (xmax - xmin))
                    and lbl_hghts[pnt] < (0.47 * (ymax - ymin))):
                pos_rands = [random.expovariate((i * adj / 1703) ** -1)
                             for adj in [xadj, yadj]]

                # adds padding to the randomly chosen distances and randomly
                # picks a quadrant relative to the orig point for the label
                new_pos = [
                    px + (2.3 * pnt_gap + pos_rand) * random.choice([-1, 1])
                    for px, pos_rand, pnt_gap in zip(pnt, pos_rands,
                                                     pnt_gaps[pnt])
                    ]

                # exclude areas too close to the edge of the plot from the
                # vicinity to search over for the label
                new_pos[0] = np.clip(new_pos[0],
                                     xmin + lbl_wdths[pnt] * 0.53,
                                     xmax - lbl_wdths[pnt] * 0.53)
                new_pos[1] = np.clip(new_pos[1],
                                     ymin + lbl_hghts[pnt] * 0.71,
                                     ymax - lbl_hghts[pnt] * 0.71)

                # exclude areas too far away from the original point
                new_pos[0] = np.clip(new_pos[0],
                                     pnt[0] - 331 * xadj, pnt[0] + 331 * xadj)
                new_pos[1] = np.clip(new_pos[1],
                                     pnt[1] - 331 * yadj, pnt[1] + 331 * yadj)

                # if the label has a bottom text component, account for it
                # when determining the label's vertical alignment
                if bot_lbl:
                    top_prop = (4 / 3 + bot_lbl.count('\n')) ** -1
                else:
                    top_prop = 0.5

                # create a bounding box for the putative label location
                new_bx = np.array(
                    [[new_pos[0] - lbl_wdths[pnt] / 1.9 - xgap,
                      new_pos[1] - lbl_hghts[pnt] * top_prop - ygap],
                     [new_pos[0] + lbl_wdths[pnt] / 1.9 + xgap,
                      new_pos[1] + lbl_hghts[pnt] * (1 - top_prop) + ygap]]
                    )

                # if the putative bounding box does not overlap with any
                # existing plot elements, choose this as the label's location
                if not any(check_overlap(new_bx, pnt2_bx)
                           for pnt2_bxs in pnt_bxs.values()
                           for pnt2_bx in pnt2_bxs):
                    lbl_pos[pnt] = (new_pos[0], new_pos[1]), 'center'
                    pnt_bxs[pnt] += [new_bx]

    # for each point where labels were successfully placed, draw the label and
    # a line connecting the label and the point if necessary
    lbl_pos = {pnt: lbl for pnt, lbl in lbl_pos.items() if lbl}
    for (pnt_x, pnt_y), ((lbl_x, lbl_y), lbl_ha) in lbl_pos.items():
        text_props = dict()

        if font_dict and (pnt_x, pnt_y) in font_dict:
            text_props.update(font_dict[pnt_x, pnt_y])

        # create the main (top) part of the label
        if lbl_ha == 'center' and not plot_dict[pnt_x, pnt_y][1][1]:
            ax.text(lbl_x, lbl_y, plot_dict[pnt_x, pnt_y][1][0],
                    size=font_size, ha=lbl_ha, va='center', **text_props)
        else:
            ax.text(lbl_x, lbl_y + yadj * 2.3, plot_dict[pnt_x, pnt_y][1][0],
                    size=font_size, ha=lbl_ha, va='bottom', **text_props)

        # create the secondary (bottom) part of the label
        if plot_dict[pnt_x, pnt_y][1][1]:
            ax.text(lbl_x, lbl_y - yadj * 2.3, plot_dict[pnt_x, pnt_y][1][1],
                    size=font_size / 1.61, ha=lbl_ha, va='top', **text_props)

        # figure out where the end of the line corresponding to the label is
        lbl_bx = pnt_bxs[pnt_x, pnt_y][-1]
        txt_y = np.clip(pnt_y,
                        *(lbl_bx[:, 1] + np.array([ygap, -ygap]) / 2.1))

        if pnt_x <= lbl_bx[0, 0]:
            txt_x = lbl_bx[0, 0] + 0.19 * np.diff(lbl_bx[:, 0])[0]
        elif pnt_x >= lbl_bx[1, 0]:
            txt_x = lbl_bx[1, 0] - 0.19 * np.diff(lbl_bx[:, 0])[0]
        else:
            txt_x = np.mean(lbl_bx[:, 0])

        # figure out where the end of the line corresponding to the point is
        x_delta, y_delta = (pnt_x - txt_x), (pnt_y - txt_y)
        ln_x, ln_y = x_delta / xadj, y_delta / yadj
        ln_mag = (ln_x ** 2 + ln_y ** 2) ** 0.5
        ln_cos, ln_sin = ln_x / ln_mag, ln_y / ln_mag

        crc_x = pnt_x - (pnt_gaps[pnt_x, pnt_y][0] * ln_cos)
        crc_y = pnt_y - (pnt_gaps[pnt_x, pnt_y][1] * ln_sin)
        crc_bx = np.array([[crc_x - xgap / 2.9, crc_y - ygap / 2.9],
                           [crc_x + xgap / 2.9, crc_y + ygap / 2.9]])

        # if the label is sufficiently far away from the plot element it
        # annotates, create a connecting line between the two
        if not check_overlap(lbl_bx, crc_bx):
            line_props = line_args.copy()

            if line_dict and (pnt_x, pnt_y) in line_dict:
                line_props.update(line_dict[pnt_x, pnt_y])

            ax.plot([crc_x, txt_x], [crc_y, txt_y], **line_props)

    return {pos: lbl for pos, lbl in lbl_pos.items() if lbl}

