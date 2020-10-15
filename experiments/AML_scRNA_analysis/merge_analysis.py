
import os
import argparse
import bz2
from pathlib import Path
import dill as pickle
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        "Concatenates the distributed output of an iteration "
        "of the subgrouping testing experiment for use in further analyses."
        )

    parser.add_argument('use_dir', type=str)
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

    pred_df = pd.DataFrame()
    for pred_file in Path(args.use_dir, 'merge').glob("out-pred_*.p.gz"):
        with bz2.BZ2File(pred_file, 'r') as fl:
            pred_data = pickle.load(fl)
        pred_df = pred_df.append(pred_data)

    assert sorted(muts_list) == sorted(pred_df.index), (
        "Tested mutations missing from merged classifier predictions!")
    with bz2.BZ2File(os.path.join(args.use_dir, "out-pred.p.gz"), 'w') as fl:
        pickle.dump(pred_df, fl, protocol=-1)

    tune_dfs = [pd.DataFrame() for _ in range(3)] + [None]
    for tune_file in Path(args.use_dir, 'merge').glob("out-tune_*.p.gz"):
        with bz2.BZ2File(tune_file, 'r') as fl:
            tune_data = pickle.load(fl)

        if tune_dfs[3] is None:
            tune_dfs[3] = tune_data[3]
        else:
            assert tune_dfs[3] == tune_data[3], (
                "Inconsistent mutation classifiers between gather tasks!")

        for i in range(3):
            tune_dfs[i] = tune_dfs[i].append(tune_data[i])

    for i in range(3):
        assert sorted(muts_list) == sorted(tune_dfs[i].index), (
            "Tested mutations missing from merged tuning statistics!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-tune.p.gz"), 'w') as fl:
        pickle.dump(tune_dfs, fl, protocol=-1)

    sc_df = pd.DataFrame()
    for sc_file in Path(args.use_dir, 'merge').glob("out-sc_*.p.gz"):
        with bz2.BZ2File(sc_file, 'r') as fl:
            sc_data = pickle.load(fl)
        sc_df = sc_df.append(sc_data)

    assert sorted(muts_list) == sorted(sc_df.index), (
        "Tested mutations missing from single-cell predictions!")
    with bz2.BZ2File(os.path.join(args.use_dir, "out-sc.p.gz"), 'w') as fl:
        pickle.dump(sc_df, fl, protocol=-1)

    auc_df = pd.DataFrame()
    for auc_file in Path(args.use_dir, 'merge').glob("out-aucs_*.p.gz"):
        with bz2.BZ2File(auc_file, 'r') as fl:
            auc_data = pickle.load(fl)
        auc_df = auc_df.append(pd.DataFrame(auc_data))

    assert sorted(muts_list) == sorted(auc_df.index), (
        "Tested mutations missing from merged classifier accuracies!")
    with bz2.BZ2File(os.path.join(args.use_dir, "out-aucs.p.gz"), 'w') as fl:
        pickle.dump(auc_df, fl, protocol=-1)

    conf_df = pd.DataFrame()
    for conf_file in Path(args.use_dir, 'merge').glob("out-conf_*.p.gz"):
        with bz2.BZ2File(conf_file, 'r') as fl:
            conf_data = pickle.load(fl)
        conf_df = conf_df.append(pd.DataFrame(conf_data))

    assert sorted(muts_list) == sorted(conf_df.index), (
        "Tested mutations missing from merged subsampled accuracies!")
    with bz2.BZ2File(os.path.join(args.use_dir, "out-conf.p.gz"), 'w') as fl:
        pickle.dump(conf_df, fl, protocol=-1)


if __name__ == "__main__":
    main()

