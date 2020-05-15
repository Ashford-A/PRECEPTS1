
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

import argparse
import bz2
from pathlib import Path
import dill as pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random

from dryadic.features.mutations import MuType
from HetMan.experiments.subvariant_isolate import cna_mtypes, ex_mtypes
from HetMan.experiments.utilities.mutations import ExMcomb
from HetMan.experiments.subvariant_isolate.merge_isolate import (
    compare_muts, calculate_auc)
from HetMan.experiments.subgrouping_isolate.merge_isolate import (
    calculate_siml)

from itertools import cycle
from functools import reduce
from operator import or_, add


def main():
    parser = argparse.ArgumentParser(
        "Processes and consolidates the distributed output of an iteration "
        "of the dyad isolation experiment for use in further analyses."
        )

    parser.add_argument(
        "use_dir", type=str,
        help="root folder of where experiment output was saved"
        )
    parser.add_argument(
        "--cores", '-c', type=int, default=1,
        help="the number of compute cores available to this script"
        )

    parser.add_argument(
        "--test", action='store_true',
        help="don't write anything to file and perform diagnostic tests?"
        )
    args = parser.parse_args()

    # get list of output files from all parallelized jobs
    file_list = tuple(Path(os.path.join(args.use_dir, 'output')).glob(
        "out__cv-*_task*.p"))
    assert (len(file_list) % 40) == 0, "Missing output files detected!"

    # initialize list for storing raw output data
    task_count = len(file_list) // 40
    out_data = [[None for task_id in range(task_count)]
                for cv_id in range(40)]

    # populate raw output list with data from each combination of
    # parallelized task split and cross-validation run
    for out_fl in file_list: 
        base_name = out_fl.stem.split('out__')[1]
        cv_id = int(base_name.split('cv-')[1].split('_')[0])
        task_id = int(base_name.split('task-')[1])

        with open(out_fl, 'rb') as f:
            out_data[cv_id][task_id] = pickle.load(f)

    # find the mutation classifier that was used in this experiment
    use_clfs = set(out_dict['Clf'] for ols in out_data for out_dict in ols)
    assert len(use_clfs) == 1, ("Each experiment must be run with "
                                "exactly one classifier!")
    use_clf = tuple(use_clfs)[0]

    # find the hyper-parameter tuning grid that was used in this experiment
    use_tune = set(out_dict['Clf'].tune_priors for ols in out_data
                   for out_dict in ols)
    assert len(use_tune) == 1, ("Each experiment must be run with "
                                "exactly one set of tuning priors!")

    # load the -omic datasets for this experiment's tumour cohort
    with bz2.BZ2File(os.path.join(args.use_dir, 'setup',
                                  "cohort-data.p.gz"), 'r') as f:
        cdata = pickle.load(f)

    use_mtree = tuple(cdata.mtrees.values())[0]
    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    out_dfs = {k: {ex_lbl: [None for cv_id in range(40)]
                   for ex_lbl in ['All', 'Iso', 'IsoShal']}
               for k in ['Pars', 'Time', 'Acc', 'Pred']}

    pred_lists = {ex_lbl: [None for cv_id in range(40)]
                  for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for cv_id, ols in enumerate(out_data):
        for k in out_dfs:
            for ex_lbl in ['All', 'Iso', 'IsoShal']:
                out_dfs[k][ex_lbl][cv_id] = pd.DataFrame({
                    mtype: out_vals[ex_lbl]
                    for out_dict in ols
                    for mtype, out_vals in out_dict[k].items()
                    }).transpose()

        cdata_samps = sorted(cdata.get_samples())
        random.seed((cv_id // 4) * 3901 + 23)
        random.shuffle(cdata_samps)

        cdata.update_split(13101 + 103 * cv_id,
                           test_samps=cdata_samps[(cv_id % 4)::4])

        samps_dict = {'train': cdata.get_train_samples(),
                      'test': cdata.get_test_samples()}

        for ex_lbl in ['All', 'Iso', 'IsoShal']:
            use_muts = out_dfs['Pred'][ex_lbl][cv_id].index
            assert sorted(use_muts) == sorted(muts_list), (
                "Mutations with `{}` predictions for c-v fold <{}> don't "
                "match those enumerated during setup!".format(ex_lbl, cv_id)
                )

            pred_lists[ex_lbl][cv_id] = pd.DataFrame(
                index=use_muts, columns=cdata.get_samples()).applymap(
                    lambda x: [])

            pred_lists[ex_lbl][cv_id][samps_dict['test']] = pd.DataFrame(
                [preds for preds in out_dfs['Pred'][ex_lbl][cv_id].test],
                index=use_muts, columns=samps_dict['test']
                ).applymap(lambda x: [x])

            if 'train' in out_dfs['Pred'][ex_lbl][cv_id]:
                train_mat = out_dfs['Pred'][ex_lbl][cv_id].train[
                    ~out_dfs['Pred'][ex_lbl][cv_id].train.isnull()]

                for mtype, train_preds in train_mat.iteritems():
                    cur_genes = mtype.get_labels()

                    mut_samps = reduce(or_,
                                       [use_mtree[gene].get_samples()
                                        for gene in cur_genes])

                    shal_samps = reduce(
                        or_,
                        [dict(cna_mtypes)['Shal'].get_samples(use_mtree[gene])
                         for gene in cur_genes]
                        )

                    mtype_samps = {
                        samp for samp, phn in zip(samps_dict['train'],
                                                  cdata.train_pheno(mtype))
                        if phn
                        }

                    if ex_lbl == 'Iso':
                        use_samps = mut_samps - mtype_samps
                    elif ex_lbl == 'IsoShal':
                        use_samps = mut_samps - mtype_samps - shal_samps

                    out_samps = sorted(use_samps & set(samps_dict['train']))
                    pred_lists[ex_lbl][cv_id].loc[mtype][out_samps] = [
                        [x] for x in train_preds]

    pred_dfs = {ex_lbl: reduce(add, pred_mats)
                for ex_lbl, pred_mats in pred_lists.items()}
    assert (pred_dfs['All'].applymap(len) == 10).values.all(), (
        "Incorrect number of testing CV scores!")

    pars_dfs = {ex_lbl: pd.concat(out_dfs['Pars'][ex_lbl], axis=1)
                for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for pars_df in pars_dfs.values():
        assert pars_df.shape[1] == (40 * len(use_clf.tune_priors)), (
            "Tuned parameter values missing for some CVs!")

    time_dfs = {ex_lbl: pd.concat(out_dfs['Time'][ex_lbl], axis=1)
                for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for time_df in time_dfs.values():
        assert time_df.shape[1] == 80, (
            "Algorithm fitting times missing for some CVs!")
        assert (time_df.applymap(len) == use_clf.test_count).values.all(), (
            "Algorithm fit times missing for some hyper-parameter values!")

    acc_dfs = {ex_lbl: pd.concat(out_dfs['Acc'][ex_lbl], axis=1)
               for ex_lbl in ['All', 'Iso', 'IsoShal']}

    for acc_df in acc_dfs.values():
        assert acc_df.shape[1] == 120, (
            "Algorithm tuning accuracies missing for some CVs!")
        assert (acc_df.applymap(len) == use_clf.test_count).values.all(), (
            "Algorithm tuning stats missing for some hyper-parameter values!")

    for out_dfs in [pred_dfs, pars_dfs, time_dfs, acc_dfs]:
        for out_df in out_dfs.values():
            assert compare_muts(out_df.index, muts_list), (
                "Mutations for which predictions were made do not match "
                "the list of mutations enumerated during setup!"
                )

    if not args.test:
        with bz2.BZ2File(os.path.join(args.use_dir, "out-pred.p.gz"),
                         'w') as fl:
            pickle.dump(pred_dfs, fl, protocol=-1)

        with bz2.BZ2File(os.path.join(args.use_dir, "out-tune.p.gz"),
                         'w') as fl:
            pickle.dump([pars_dfs, time_dfs, acc_dfs, use_clf],
                        fl, protocol=-1)

    cdata.update_split(test_prop=0)
    train_samps = np.array(cdata.get_train_samples())
    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in muts_list}

    if not args.test:
        with bz2.BZ2File(os.path.join(args.use_dir, "out-pheno.p.gz"),
                         'w') as fl:
            pickle.dump(pheno_dict, fl, protocol=-1)

    auc_dicts = {
        ex_lbl: {
            'CV': pd.DataFrame.from_records(
                tuple(zip(cycle(muts_list), Parallel(
                    backend='threading',
                    n_jobs=args.cores, pre_dispatch=args.cores
                    )(delayed(calculate_auc)(
                        pheno_dict[mtype],
                        pred_dfs[ex_lbl].loc[mtype, train_samps],
                        cv_indx=cv_id
                        ) for cv_id in range(10) for mtype in muts_list)
                    ))
                ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0],

            'mean': pd.Series(dict(zip(muts_list, Parallel(
                backend='threading',
                n_jobs=args.cores, pre_dispatch=args.cores
                )(delayed(calculate_auc)(
                    pheno_dict[mtype],
                    pred_dfs[ex_lbl].loc[mtype, train_samps],
                    use_mean=True
                    ) for mtype in muts_list)
                )))
            }
        for ex_lbl in ['All', 'Iso', 'IsoShal']
        }

    for ex_lbl in ['All', 'Iso', 'IsoShal']:
        auc_dicts[ex_lbl]['CV'].name = None
        auc_dicts[ex_lbl]['CV'].index.name = None

    if not args.test:
        with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"),
                         'w') as fl:
            pickle.dump(auc_dicts, fl, protocol=-1)

    random.seed(9903)
    sub_inds = [random.choices([False, True], k=len(cdata.get_samples()))
                for _ in range(50)]

    conf_lists = {
        ex_lbl: {
            'mean': pd.DataFrame.from_records(
                tuple(zip(cycle(muts_list), Parallel(
                    backend='threading',
                    n_jobs=args.cores, pre_dispatch=args.cores
                    )(delayed(calculate_auc)(
                        pheno_dict[mtype][sub_indx],
                        pred_dfs[ex_lbl].loc[mtype, train_samps[sub_indx]],
                        use_mean=True
                        ) for sub_indx in sub_inds for mtype in muts_list)
                    ))
                ).pivot_table(index=0, values=1, aggfunc=list).iloc[:, 0]
            }
        for ex_lbl in ['All', 'Iso', 'IsoShal']
        }

    for ex_lbl in ['All', 'Iso', 'IsoShal']:
        conf_lists[ex_lbl]['mean'].name = None
        conf_lists[ex_lbl]['mean'].index.name = None

    if not args.test:
        with bz2.BZ2File(os.path.join(args.use_dir, "out-conf.p.gz"),
                         'w') as fl:
            pickle.dump(conf_lists, fl, protocol=-1)

    mcomb_lists = {
        'Iso': {mut for mut in muts_list
                if (isinstance(mut, ExMcomb)
                    and not (mut.all_mtype
                             & dict(cna_mtypes)['Shal']).is_empty())},
        'IsoShal': {mut for mut in muts_list
                    if (isinstance(mut, ExMcomb)
                        and (mut.all_mtype
                             & dict(cna_mtypes)['Shal']).is_empty())}
        }

    for ex_lbl, ex_mtype in ex_mtypes:
        for mcomb in mcomb_lists[ex_lbl]:
            cur_genes = mcomb.get_labels()
            gene_ex = reduce(or_, [MuType({('Gene', cur_gene): ex_mtype})
                                   for cur_gene in cur_genes])

            all_mtype = reduce(
                or_,
                [MuType({('Gene', cur_gene): use_mtree[cur_gene].allkey()})
                 for cur_gene in cur_genes]
                )

            pheno_dict[ex_lbl, mcomb] = np.array(cdata.train_pheno(
                all_mtype - gene_ex))

    siml_lists = {
        ex_lbl: dict(zip(mcomb_lists[ex_lbl], Parallel(
            backend='threading',
            n_jobs=args.cores, pre_dispatch=args.cores
            )(delayed(calculate_siml)(
                mcomb, pheno_dict, (ex_lbl, mcomb),
                pred_dfs[ex_lbl].loc[mcomb, train_samps]
                ) for mcomb in mcomb_lists[ex_lbl])
            ))
        for ex_lbl in ['Iso', 'IsoShal']
        }

    if not args.test:
        with bz2.BZ2File(os.path.join(args.use_dir, "out-siml.p.gz"),
                         'w') as fl:
            pickle.dump(siml_lists, fl, protocol=-1)


if __name__ == "__main__":
    main()

