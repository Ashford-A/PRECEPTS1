
def place_annot(x_vec, y_vec, size_vec, annot_vec, x_range, y_range,
                gap_adj=150):
    placed_annot = []
    for i, (xval, yval, size_val, annot) in enumerate(zip(
            x_vec, y_vec, size_vec, annot_vec)):

        lbl_xgap = x_range * (size_val ** 0.47) / (gap_adj * 10)
        lbl_ygap = y_range * (size_val ** 0.47) / (gap_adj * 10)

        if all((xs > (xval + x_range * (9 / gap_adj)))
               or (xs < (xval - x_range * (1 / gap_adj)))
               or (ys > (yval + y_range * (3 / gap_adj)))
               or (ys < (yval - y_range * (1 / gap_adj)))
               for xs, ys in zip(x_vec[:i] + x_vec[(i + 1):],
                                 y_vec[:i] + y_vec[(i + 1):])):
 
            placed_annot += [(xval + lbl_xgap, yval + lbl_ygap,
                              annot, 'left')]

        elif all((xs > (xval + x_range * (9 / gap_adj)))
                 or (xs < (xval - x_range * (1 / gap_adj)))
                 or (ys > (yval + y_range * (1 / gap_adj)))
                 or (ys < (yval - y_range * (3 / gap_adj)))
                 for xs, ys in zip(x_vec[:i] + x_vec[(i + 1):],
                                   y_vec[:i] + y_vec[(i + 1):])):
 
            placed_annot += [(xval + lbl_xgap, yval - lbl_ygap,
                              annot, 'left')]
 
        elif all((xs > (xval + x_range * (1 / gap_adj)))
                 or (xs < (xval - x_range * (9 / gap_adj)))
                 or (ys > (yval + y_range * (3 / gap_adj)))
                 or (ys < (yval - y_range * (1 / gap_adj)))
                 for xs, ys in zip(x_vec[:i] + x_vec[(i + 1):],
                                   y_vec[:i] + y_vec[(i + 1):])):
 
            placed_annot += [(xval - lbl_xgap, yval + lbl_ygap,
                              annot, 'right')]
 
        elif all((xs > (xval + x_range * (1 / gap_adj)))
                 or (xs < (xval - x_range * (9 / gap_adj)))
                 or (ys > (yval + y_range * (1 / gap_adj)))
                 or (ys < (yval - y_range * (3 / gap_adj)))
                 for xs, ys in zip(x_vec[:i] + x_vec[(i + 1):],
                                   y_vec[:i] + y_vec[(i + 1):])):
 
            placed_annot += [(xval - lbl_xgap, yval - lbl_ygap,
                              annot, 'right')]
 
    return placed_annot

