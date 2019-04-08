
import os
import sys
base_dir = os.path.dirname(__file__)
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


def cdata_hash(cdata):
    expr_hash = tuple(dict(cdata.omic_data.sum().round(5)).items())
    mut_str = cdata.train_mut.get_newick()

    return expr_hash, tuple(mut_str.count(k) for k in sorted(set(mut_str)))


def merge_cohort_data(out_dir, cur_mdl=None):
    meta_file = os.path.join(out_dir, "metadata.txt")

    if not os.path.isfile(meta_file):
        raise MergeError("Meta-data file for output directory\n{}\nnot "
                         "present, directory is either locked due to a "
                         "concurrent process running or meta-data file needs "
                         "to be instantiated using:\n\ttouch {}".format(
                             out_dir, meta_file))

    with open(meta_file, 'r') as fl:
        meta_data = fl.read()
    os.remove(meta_file)

    if len(meta_data) > 0:
        meta_dict = [[meta_str.split(': ')[0],
                      meta_str.split(': ')[1].split(', ')]
                     for meta_str in meta_data.split('\n') if meta_str]

    else:
        meta_dict = []

    cur_cdatas = {
        fl.split('data_')[1].split('.p')[0]: pickle.load(
            open(os.path.join(out_dir, fl), 'rb'))
        for fl in os.listdir(os.path.join(out_dir)) if 'cohort-data_v' in fl
        }

    new_cdatas = {
        fl.split('data__')[1].split('.p')[0]: pickle.load(
            open(os.path.join(out_dir, fl), 'rb'))
        for fl in os.listdir(os.path.join(out_dir)) if 'cohort-data__' in fl
        }

    for mdl, cdata in new_cdatas.items():
        if cdata.cv_seed != 0:
            raise MergeError("Cohort for model {} does not have a "
                             "cross-validation seed of zero!".format(mdl))

        if cdata.test_samps is not None:
            raise MergeError("Cohort for model {} does not have an empty "
                             "testing sample set!".format(mdl))

    cur_chsums = {cdata_hash(cdata): vrs for vrs, cdata in cur_cdatas.items()}
    new_chsums = {mdl: cdata_hash(cdata) for mdl, cdata in new_cdatas.items()}

    if len(cur_cdatas) > 0:
        new_version = max(int(vrs[1:]) for vrs in cur_cdatas) + 1
    else:
        new_version = 0

    for mdl, new_chsum in new_chsums.items():
        os.remove(os.path.join(out_dir, "cohort-data__{}.p".format(mdl)))

        if new_chsum not in cur_chsums:
            vrs_str = 'v{}'.format(new_version)
            new_fl = os.path.join(out_dir, "cohort-data_{}.p".format(vrs_str))

            pickle.dump(new_cdatas[mdl], open(new_fl, 'wb'))
            cur_cdatas[vrs_str] = new_cdatas[mdl]
            del(new_cdatas[mdl])

            cur_chsums[new_chsum] = vrs_str
            meta_dict += [[vrs_str, [mdl]]]
            new_version += 1

        else:
            meta_indx = [i for i, (vrs, _) in enumerate(meta_dict)
                         if vrs == cur_chsums[new_chsum]]
            if mdl not in meta_dict[meta_indx[0]][1]:
                meta_dict[meta_indx[0]][1] += [mdl]

    if cur_mdl is None:
        use_vrs = max(int(vrs[1:]) for vrs in cur_cdatas)
        cdata = cur_cdatas['v{}'.format(use_vrs)]

    else:
        pass

    with open(meta_file, 'w') as fl:
        fl.write('\n'.join("{}: {}".format(meta_val[0],
                                           ', '.join(meta_val[1]))
                           for meta_val in meta_dict))

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

    assert fit_acc['train'].shape[0] == len(vars_list), (
        "Training fit accuracies missing for some tested mutations!")
    assert fit_acc['test'].shape[0] == len(vars_list), (
        "Testing fit accuracies missing for some tested mutations!")

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

    assert tune_acc['mean'].shape[0] == len(vars_list), (
        "Mean of tuning test accuracies missing for some tested mutations!")
    assert tune_acc['std'].shape[0] == len(vars_list), (
        "Variance of tuning test accuracies missing for some "
        "tested mutations!"
        )

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

