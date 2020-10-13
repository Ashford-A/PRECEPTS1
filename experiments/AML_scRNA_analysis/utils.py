
from .data_dirs import expr_dir
import os
from scipy.io import mmread
import numpy as np
import pandas as pd


def load_scRNA_expr():
    with open(os.path.join(expr_dir, "genes.tsv"), 'r') as f:
        gene_names = f.readlines()

    with open(os.path.join(expr_dir, "BM_combined_sept15_barcodes.tsv"),
              'r') as f:
        samp_bars = f.readlines()

    gene_names = [gene.strip() for gene in gene_names]
    samp_bars = [bar.strip() for bar in samp_bars]

    expr_df = pd.DataFrame.sparse.from_spmatrix(
        mmread(os.path.join(expr_dir, "BM_combined_sept15_matrix.mtx")),
        index=samp_bars, columns=gene_names
        )

    return np.log10(expr_df + 0.01).astype(pd.SparseDtype("float", -2.))

