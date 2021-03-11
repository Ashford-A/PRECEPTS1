
import numpy as np
from colorsys import hls_to_rgb
import dill as pickle
from importlib import import_module


def get_label(mut):
    return next(mut.label_iter())


def get_subtype(mtype):
    return next(mtype.subtype_iter())[1]


def compare_muts(*muts_lists):
    return len(set(tuple(sorted(muts_list))
                   for muts_list in muts_lists)) == 1


def get_distr_transform(tune_distr):
    distr_diff = np.array(tune_distr[1:]) - np.array(tune_distr[:-1])
    diff_min = np.log10(np.min(distr_diff))
    diff_max = np.log10(np.max(distr_diff))

    if (diff_max - diff_min) > 4./3:
        trans_fx = np.log10
    else:
        trans_fx = lambda x: x

    return trans_fx


def warning_on_one_line(message, category, filename, lineno,
                        file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def choose_label_colour(gene, clr_seed=15707, clr_lum=0.5, clr_sat=0.8):
    np.random.seed(int((clr_seed + np.prod([ord(char) for char in gene]))
                       % (2 ** 14)))

    return hls_to_rgb(h=np.random.uniform(size=1)[0], l=clr_lum, s=clr_sat)


def transfer_model(trnsf_fl, clf, use_feats):
    with open(trnsf_fl, 'rb') as f:
        trnsf_preds = clf.predict_omic(pickle.load(f).train_data(
            pheno=None, include_feats=use_feats)[0], lbl_type='raw')

    return trnsf_preds


def load_mut_clf(clf_lbl):
    if clf_lbl[:6] == 'Stan__':
        use_module = import_module('HetMan.experiments.utilities'
                                   '.stan_models.{}'.format(
                                       clf_lbl.split('Stan__')[1]))
        mut_clf = getattr(use_module, 'UsePipe')

    else:
        mut_clf = eval(clf_lbl)

    return mut_clf

