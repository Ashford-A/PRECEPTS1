
import os
import pandas as pd
from dryadic.features.cohorts.utils import log_norm


def load_beat_expression(beataml_dir):
    expr = pd.read_csv(os.path.join(beataml_dir, "matrices",
                                    "CTD2_TPM_transcript.tsv"),
                       sep='\t', index_col=0)

    return log_norm(expr.transpose())

