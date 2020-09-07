
from ...experiments.utilities.data_dirs import oncogene_list
import pandas as pd


def get_gene_list(min_sources=2):
    gene_df = pd.read_csv(oncogene_list, sep='\t', skiprows=1, index_col=0)

    return gene_df.index[
        (gene_df.loc[:, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                         'FOUNDATION ONE', 'MSK-IMPACT']]
         == 'Yes').sum(axis=1) >= min_sources
        ].tolist()

