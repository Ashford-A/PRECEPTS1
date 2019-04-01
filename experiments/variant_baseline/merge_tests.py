
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.variant_baseline import *
from HetMan.experiments.variant_baseline.fit_tests import load_variants_list

import argparse
import pandas as pd
import dill as pickle
from itertools import product
from operator import itemgetter


class MergeError(Exception):
    pass


def merge_cohort_data(out_dir):
    meta_file = os.path.join(out_dir, "metadata.txt")

    if not os.path.isfile(meta_file):
        raise MergeError("Meta-data file for output directory\n{}\nnot "
                         "present, directory is either locked due to a "
                         "concurrent process running or meta-data file needs "
                         "to be instantiated using\n\ttouch {}".format(
                             out_dir, meta_file))

    with open(meta_file, 'r') as fl:
        meta_data = fl.read()
    os.remove(meta_file)

    coh_files = [fl for fl in os.listdir(os.path.join(out_dir))
                 if 'cohort-data__' in fl]

    cdata = pickle.load(open(os.path.join(out_dir, coh_files[0]), 'rb'))

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('expr_source', type=str,
                        choices=list(expr_sources.keys()),
                        help='which TCGA expression data source to use')
    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")

    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of mutated samples needed to test a gene"
        )

    parser.add_argument('classif', type=str,
                        help='the name of a mutation classifier')
    parser.add_argument('--use_dir', type=str, default=base_dir)

    args = parser.parse_args()
    out_files = [(fl, int(fl.split('out__cv-')[1].split('_task-')[0]),
                  int(fl.split('_task-')[1].split('.p')[0]))
                  for fl in os.listdir(os.path.join(args.use_dir, 'output'))
                 if 'out__cv-' in fl]

    out_files = sorted(out_files, key=itemgetter(1, 2))
    task_count = len(set(task for _, _, task in out_files))
    out_list = [pickle.load(open(os.path.join(args.use_dir, 'output', fl),
                                 'rb'))
                for fl, _, _ in out_files]

    use_clf = set(ols['Clf'] for ols in out_list)
    if len(use_clf) != 1:
        raise ValueError("Each experiment must be run "
                         "with exactly one classifier!")

    vars_list = load_variants_list(
        args.use_dir, args.expr_source, args.cohort, args.samp_cutoff)

    fit_acc = {
        samp_set: pd.concat([
            pd.concat([
                pd.DataFrame.from_dict({mtype: acc[samp_set]
                                        for mtype, acc in ols['Acc'].items()})
                for i, ols in enumerate(out_list) if i % task_count == task_id
                ], axis=0)
            for task_id in range(task_count)
        ], axis=1).transpose()
        for samp_set in ['train', 'test']
        }

    tune_list = tuple(product(*[
        vals for _, vals in tuple(use_clf)[0].tune_priors]))

    tune_acc = {
        stat_lbl: pd.concat([
            pd.concat([
                pd.DataFrame.from_dict({mtype: acc['tune'][stat_lbl]
                                        for mtype, acc in ols['Acc'].items()},
                                       orient='index', columns=tune_list)
                for i, ols in enumerate(out_list) if i % task_count == task_id
                ], axis=1, sort=True)
            for task_id in range(task_count)
        ], axis=0) for stat_lbl in ['mean', 'std']
        }

    par_df = pd.concat([
        pd.concat([pd.DataFrame.from_dict(ols['Params'], orient='index')
                   for i, ols in enumerate(out_list)
                   if i % task_count == task_id], axis=1)
        for task_id in range(task_count)
        ], axis=0)

    tune_time = {
        stage_lbl: {
            stat_lbl: pd.concat([
                pd.concat([
                    pd.DataFrame.from_dict({
                        mtype: tm['tune'][stage_lbl][stat_lbl]
                        for mtype, tm in ols['Time'].items()},
                        orient='index', columns=tune_list)
                    for i, ols in enumerate(out_list)
                    if i % task_count == task_id
                    ], axis=1, sort=True)
                for task_id in range(task_count)
                ], axis=0) for stat_lbl in ['avg', 'std']
            } for stage_lbl in ['fit', 'score']
        }

    pickle.dump({'Tune': {'Acc': tune_acc, 'Time': tune_time}, 'Fit': fit_acc,
                 'Params': par_df, 'Clf': tuple(use_clf)[0]},
                open(os.path.join(args.use_dir, "out-data.p"), 'wb'))


if __name__ == "__main__":
    main()

