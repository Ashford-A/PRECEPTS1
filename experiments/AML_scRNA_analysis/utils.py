
from .data_dirs import expr_dir
import os
from scipy.io import mmread
import numpy as np
import pandas as pd


def load_scRNA_expr():
    with open(os.path.join(expr_dir, "genes.tsv"), 'r') as f:
        gene_names = f.readlines()

    with open(os.path.join(expr_dir, "barcodes.tsv"),
              'r') as f:
        samp_bars = f.readlines()

    gene_names = [gene.strip() for gene in gene_names]
    samp_bars = [bar.strip() for bar in samp_bars]

    expr_df = pd.DataFrame.sparse.from_spmatrix(
        mmread(os.path.join(expr_dir, "matrix.mtx")),
        index=samp_bars, columns=gene_names
        )

    return np.log10(expr_df + 0.01).astype(pd.SparseDtype("float", -2.))

def filter_mtype(mtype, gene):
    if isinstance(mtype, RandomType):
        if mtype.base_mtype is None:
            filter_stat = False
        else:
            filter_stat = get_label(mtype.base_mtype) == gene

    else:
        filter_stat = get_label(mtype) == gene

    return filter_stat
