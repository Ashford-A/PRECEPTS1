
import os
import pandas as pd
from dryadic.features.cohorts.utils import log_norm


def load_beat_expression(expr_file, tx_map):
    expr = pd.read_csv(expr_file, sep='\t', index_col=0)
    tx_gene = pd.read_csv(tx_map, sep='\t', index_col=0)
    expr = log_norm(expr.loc[expr.index.isin(tx_gene.index)].transpose())

    expr.columns = pd.MultiIndex.from_arrays(
        [tx_gene.loc[expr.columns]['gene'], expr.columns],
        names=['Gene', 'Transcript']
        )

    return expr


def load_beat_mutations(muts_file):
    return pd.read_csv(muts_file, sep='\t')

