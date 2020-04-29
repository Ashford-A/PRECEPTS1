
import numpy as np
from colorsys import hls_to_rgb


def detect_log_distr(tune_distr):
    distr_diff = np.array(tune_distr[1:]) - np.array(tune_distr[:-1])
    diff_min = np.log10(np.min(distr_diff))
    diff_max = np.log10(np.max(distr_diff))

    return (diff_max - diff_min) > 4./3


def warning_on_one_line(message, category, filename, lineno,
                        file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def ordinal_label(n):
    return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20
                                                        else n % 10, "th"))


def choose_label_colour(gene, clr_seed=15707, clr_lum=0.5, clr_sat=0.8):
    np.random.seed(int((clr_seed + np.prod([ord(char) for char in gene]))
                       % (2 ** 14)))

    return hls_to_rgb(h=np.random.uniform(size=1)[0], l=clr_lum, s=clr_sat)

