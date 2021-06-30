"""Parsing data downloaded from OncoKB (oncokb.org)."""

from ...experiments.utilities.data_dirs import oncogene_list
import pandas as pd


def get_gene_list(min_sources=2):
    """Parsing a downloaded OncoKB cancer gene list.

    This function loads a table of cancer genes downloaded from OncoKB (see
    the 'Cancer Genes' tab on the OncoKB home page or oncokb.org/cancerGenes).
    This table pools together a number of sources for genes considered to be
    associated with cancer; we choose four non-redundant of these sources and
    find the genes listed in at least some threshold number of them. This is
    useful for filtering the list of mutations and mutation subgroupings we
    consider in experiments to remove those not likely to be linked to
    tumorigenesis.

    Args:
        min_sources (int, optional)
            The minimum number of sources a gene needs to be listed in.

    Returns:
        gene_list (list)

    """

    gene_df = pd.read_csv(oncogene_list, sep='\t', skiprows=1, index_col=0)

    return gene_df.index[
        (gene_df.loc[:, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                         'FOUNDATION ONE', 'MSK-IMPACT']]
         == 'Yes').sum(axis=1) >= min_sources
        ].tolist()

