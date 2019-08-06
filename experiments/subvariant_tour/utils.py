
import numpy as np
import pandas as pd


def calculate_aucs(infer_dfs, cdata):
    pheno_dict = {mtype: np.array(cdata.train_pheno(mtype))
                  for mtype in tuple(infer_dfs.values())[0].index}

    auc_df = pd.DataFrame({cis_lbl: {
        mtype: (
            np.greater.outer(
                np.concatenate(infer_vals.values[pheno_dict[mtype]]),
                np.concatenate(infer_vals.values[~pheno_dict[mtype]])
                ).mean()
            + 0.5 * np.equal.outer(
                np.concatenate(infer_vals.values[pheno_dict[mtype]]),
                np.concatenate(infer_vals.values[~pheno_dict[mtype]])
                ).mean()
            ) for mtype, infer_vals in infer_df.iterrows()
        }
        for cis_lbl, infer_df in infer_dfs.items()})

    return auc_df, pheno_dict

