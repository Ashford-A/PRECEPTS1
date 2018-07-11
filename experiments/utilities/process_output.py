
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from glob import glob
import dill as pickle
import pandas as pd


def get_output_files(out_dir):
    file_list = glob(os.path.join(out_dir, 'out__task-*.p'))

    base_names = [os.path.basename(fl).split('out__')[1] for fl in file_list]
    task_ids = [int(nm.split('task-')[1].split('.p')[0]) for nm in base_names]

    return file_list, task_ids


def load_infer_output(out_dir):
    file_list, task_ids = get_output_files(out_dir)

    out_df = pd.concat([
        pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Infer'],
                               orient='index')
        for fl in file_list
        ])
 
    if all(isinstance(x, tuple) for x in out_df.index):
        out_df.index = pd.MultiIndex.from_tuples(out_df.index)

    return out_df.sort_index()


def load_infer_tuning(out_dir):
    file_list, task_ids = get_output_files(out_dir)

    tune_df = pd.concat([
        pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Tune'],
                               orient='index')
        for fl in file_list
        ])

    if all(isinstance(x, tuple) for x in tune_df.index):
        tune_df.index = pd.MultiIndex.from_tuples(tune_df.index)

    use_clf = set(pickle.load(open(fl, 'rb'))['Info']['Clf']
                  for fl in file_list)

    if len(use_clf) != 1:
        raise ValueError("Each inference isolation experiment must be run "
                         "with exactly one classifier!")

    return tune_df, tuple(use_clf)[0]

