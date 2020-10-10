
from ..subgrouping_tour import cis_lbls
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

    pred_dfs = {cis_lbl: pd.DataFrame() for cis_lbl in cis_lbls}
    for pred_file in Path(args.use_dir, 'merge').glob("out-pred_*.p.gz"):
        with bz2.BZ2File(pred_file, 'r') as fl:
            pred_data = pickle.load(fl)

        for cis_lbl in cis_lbls:
            pred_dfs[cis_lbl] = pred_dfs[cis_lbl].append(pred_data[cis_lbl])

    for cis_lbl, pred_df in pred_dfs.items():
        assert sorted(muts_list) == sorted(pred_df.index), (
            "Tested mutations missing from merged classifier predictions "
            "using cis-exclusion method `{}`!".format(cis_lbl)
            )

    with bz2.BZ2File(os.path.join(args.use_dir, "out-pred.p.gz"), 'w') as fl:
        pickle.dump(pred_dfs, fl, protocol=-1)

    tune_dfs = {cis_lbl: [pd.DataFrame() for _ in range(3)]
                for cis_lbl in cis_lbls}
    tune_dfs['Clf'] = None

    for tune_file in Path(args.use_dir, 'merge').glob("out-tune_*.p.gz"):
        with bz2.BZ2File(tune_file, 'r') as fl:
            tune_data = pickle.load(fl)

        for cis_lbl in cis_lbls:
            if tune_dfs['Clf'] is None:
                tune_dfs['Clf'] = tune_data[3]
            else:
                assert tune_dfs['Clf'] == tune_data[3], (
                    "Inconsistent mutation classifiers between gather tasks!")

            for i in range(3):
                tune_dfs[cis_lbl][i] = tune_dfs[cis_lbl][i].append(
                    tune_data[i][cis_lbl])

    for cis_lbl in cis_lbls:
        for i in range(3):
            assert sorted(muts_list) == sorted(tune_dfs[cis_lbl][i].index), (
                "Tested mutations missing from merged tuning statistics!")

    with bz2.BZ2File(os.path.join(args.use_dir, "out-tune.p.gz"), 'w') as fl:
        pickle.dump(tune_dfs, fl, protocol=-1)

    auc_df = pd.DataFrame()
    for auc_file in Path(args.use_dir, 'merge').glob("out-aucs_*.p.gz"):
        with bz2.BZ2File(auc_file, 'r') as fl:
            auc_data = pickle.load(fl)
        auc_df = auc_df.append(auc_data)

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

