
import os
import argparse
import bz2
from pathlib import Path
import dill as pickle
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        "Concatenates the distributed output of an iteration "
        "of the subgrouping isolation experiment for use in further analyses."
        )

    parser.add_argument('use_dir', type=str)
    parser.add_argument('--ex_lbls', type=str, nargs='+',
                        choices=['All', 'Iso', 'IsoShal'],
                        default=['All', 'Iso', 'IsoShal'])
    args = parser.parse_args()

    with open(os.path.join(args.use_dir, 'setup', "muts-list.p"), 'rb') as f:
        muts_list = pickle.load(f)

    pheno_dict = dict()
    for pheno_file in Path(args.use_dir, 'merge').glob("out-pheno_*.p.gz"):
        with bz2.BZ2File(pheno_file, 'r') as fl:
            pheno_dict.update(pickle.load(fl))

    assert sorted(muts_list) == sorted(pheno_dict.keys()), (
        "Tested mutations missing from list of mutations' sample statuses!")
    assert len({len(phns) for phns in pheno_dict.values()}) == 1, (
        "Inconsistent number of samples across mutation phenotype data!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pheno.p.gz"), 'w') as fl:
        pickle.dump(pheno_dict, fl, protocol=-1)

    pred_dfs = {ex_lbl: pd.DataFrame() for ex_lbl in args.ex_lbls}
    for pred_file in Path(args.use_dir, 'merge').glob("out-pred_*.p.gz"):
        with bz2.BZ2File(pred_file, 'r') as fl:
            pred_data = pickle.load(fl)

        for ex_lbl in args.ex_lbls:
            pred_dfs[ex_lbl] = pd.concat(
                [pred_dfs[ex_lbl], pred_data[ex_lbl]])

    for ex_lbl in args.ex_lbls:
        assert sorted(muts_list) == sorted(pred_dfs[ex_lbl].index), (
            "Tested mutations missing from merged classifier predictions!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pred.p.gz"), 'w') as fl:
        pickle.dump(pred_dfs, fl, protocol=-1)

    tune_dfs = [{ex_lbl: pd.DataFrame() for ex_lbl in args.ex_lbls}
                for _ in range(3)] + [None]

    for tune_file in Path(args.use_dir, 'merge').glob("out-tune_*.p.gz"):
        with bz2.BZ2File(tune_file, 'r') as fl:
            tune_data = pickle.load(fl)

        if tune_dfs[3] is None:
            tune_dfs[3] = tune_data[3]
        else:
            assert tune_dfs[3] == tune_data[3], (
                "Inconsistent mutation classifiers between gather tasks!")

        for ex_lbl in args.ex_lbls:
            for i in range(3):
                tune_dfs[i][ex_lbl] = pd.concat(
                    [tune_dfs[i][ex_lbl], tune_data[i][ex_lbl]])

    for ex_lbl in args.ex_lbls:
        for i in range(3):
            assert sorted(muts_list) == sorted(tune_dfs[i][ex_lbl].index), (
                "Tested mutations missing from merged tuning statistics!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-tune.p.gz"), 'w') as fl:
        pickle.dump(tune_dfs, fl, protocol=-1)

    auc_dfs = {ex_lbl: pd.DataFrame() for ex_lbl in args.ex_lbls}
    for auc_file in Path(args.use_dir, 'merge').glob("out-aucs_*.p.gz"):
        with bz2.BZ2File(auc_file, 'r') as fl:
            auc_data = pickle.load(fl)
 
        for ex_lbl in args.ex_lbls:
            auc_dfs[ex_lbl] = pd.concat([auc_dfs[ex_lbl],
                                         pd.DataFrame(auc_data[ex_lbl])])

    for ex_lbl in args.ex_lbls:
        assert sorted(muts_list) == sorted(auc_dfs[ex_lbl].index), (
            "Tested mutations missing from merged classifier accuracies!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"), 'w') as fl:
        pickle.dump(auc_dfs, fl, protocol=-1)

    conf_dfs = {ex_lbl: pd.DataFrame() for ex_lbl in args.ex_lbls}
    for conf_file in Path(args.use_dir, 'merge').glob("out-conf_*.p.gz"):
        with bz2.BZ2File(conf_file, 'r') as fl:
            conf_data = pickle.load(fl)

        for ex_lbl in args.ex_lbls:
            conf_dfs[ex_lbl] = pd.concat([conf_dfs[ex_lbl],
                                          pd.DataFrame(conf_data[ex_lbl])])

    for ex_lbl in args.ex_lbls:
        assert sorted(muts_list) == sorted(conf_dfs[ex_lbl].index), (
            "Tested mutations missing from merged subsampled accuracies!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-conf.p.gz"), 'w') as fl:
        pickle.dump(conf_dfs, fl, protocol=-1)


if __name__ == "__main__":
    main()

