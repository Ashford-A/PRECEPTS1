
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.stan_baseline import *
import argparse
import pandas as pd
import dill as pickle

from glob import glob
from itertools import product
from operator import itemgetter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('use_dir', type=str, default=base_dir)
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

    use_clf = set(type(ols['Clf']) for ols in out_list)
    if len(use_clf) != 1:
        raise MergeError("Each experiment must be run "
                         "with exactly one classifier!")

    use_tpr = set(ols['Clf'].tune_priors for ols in out_list)
    if len(use_tpr) != 1:
        raise MergeError("Each experiment must be run with the same tuning "
                         "priors across all classifer instances!")

    with open(os.path.join(args.use_dir, 'setup', "vars-list.p"), 'rb') as fl:
        vars_list = sorted(pickle.load(fl))

    fit_acc = {
        fmth: {
            samp_set: pd.DataFrame.from_records({
                mtype: pd.DataFrame.from_records([
                    ols['Acc'][fmth][samp_set]
                    for i, ols in enumerate(out_list)
                    if i % task_count == task_id
                    ]).unstack()
                for task_id, mtype in enumerate(vars_list)
                }).transpose()
            for samp_set in ['train', 'test']}
        for fmth in ['optim', 'sampl', 'varit']
        }

    for fmth in ['optim', 'sampl', 'varit']:
        for samp_set in ['train', 'test']:
            for acc_mth in ['AUC', 'AUPR']:
                assert fit_acc[fmth][samp_set][acc_mth].shape[1] == 10, (
                    "Some CVs are missing from the accuracy results!")

    tune_list = tuple(product(*[vals for _, vals in tuple(use_tpr)[0]]))
    tune_acc = {
        stat_lbl: {
            mtype: pd.concat([pd.DataFrame(ols['Acc'][fmth]['tune'][stat_lbl],
                                           index=tune_list,
                                           columns=[i // len(vars_list)])
                              for i, ols in enumerate(out_list)
                              if i % task_count == task_id], axis=1)
            for task_id, mtype in enumerate(vars_list)
            }
        for stat_lbl in ['mean', 'std']
        }

    for stat_lbl in ['mean', 'std']:
        for mtype in vars_list:
            assert tune_acc[stat_lbl][mtype].shape == (18, 10), (
                "Some CV parameter combinations are missing in the "
                "tuning accuracy results!"
                )

    par_df = pd.concat([
        pd.concat([
            pd.DataFrame.from_records({
                (fmth, i // len(vars_list)): {(par, mtype): val
                                              for par, val in f_vals.items()}
                for fmth, f_vals in ols['Params'].items()
                })
            for i, ols in enumerate(out_list) if i % task_count == task_id],
            axis=1,
            )
        for task_id, mtype in enumerate(vars_list)
        ], axis=0, sort=True)
    par_df.columns = pd.MultiIndex.from_tuples(par_df.columns)

    for par_lbl in dict(tuple(use_tpr)[0]):
        for fmth in ['optim', 'varit', 'sampl']:
            assert par_df.loc[par_lbl, fmth].shape == (len(vars_list), 10), (
                "Some CV parameter combinations are missing in the tuning "
                "chosen parameter results!"
                )

    fin_time = pd.concat([
        pd.DataFrame.from_dict({
            (mtype, i // len(vars_list)): {
                (fmth, time_tp): tp_val
                for time_tp, tp_val in ols['Time'][fmth]['final'].items()
                }
            for i, ols in enumerate(out_list) if i % task_count == task_id
            }, orient='index')
        for task_id, mtype in enumerate(vars_list)
        ], axis=0)

    tune_time = pd.concat([
        pd.concat([
            pd.DataFrame.from_dict({
                (mtype, stage_lbl, stat_lbl): pd.Series(
                    ols['Time'][fmth]['tune'][stage_lbl][stat_lbl],
                    index=[(i // len(vars_list), *par) for par in tune_list]
                    )
                for stat_lbl in ['avg', 'std'] for stage_lbl in ['fit', 'score']
                }, orient='index')
            for i, ols in enumerate(out_list) if i % task_count == task_id
            ], axis=1, sort=True)
        for task_id, mtype in enumerate(vars_list)
        ])

    pickle.dump({'Tune': {'Acc': tune_acc, 'Time': tune_time},
                 'Fit': {'Acc': fit_acc, 'Time': fin_time},
                 'Params': par_df, 'Clf': tuple(use_clf)[0]},
                open(os.path.join(args.use_dir, "out-data.p"), 'wb'))


if __name__ == "__main__":
    main()

