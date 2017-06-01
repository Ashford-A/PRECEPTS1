
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains a series of baseline tests of classifier performance.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from HetMan.features.cohorts import VariantCohort
from HetMan.features.variants import MuType
from HetMan.predict.classifiers import MKBMTL
from HetMan.experiments.predict_activity.config import *

import pickle
import time
import os
import sys
import synapseclient

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

in_path = os.path.join(os.path.dirname(__file__), 'data/')
out_path = os.path.join(os.path.dirname(__file__), 'output/')


def main(argv):
    """Runs the experiment."""

    syn = synapseclient.Synapse()
    syn.login("grzadkow")
    cdata = VariantCohort(syn, 'TCGA-OV', mut_genes=['TTN'],
                          mut_levels=('Gene', 'Form', 'Exon'),
                          cv_info={'Prop': 0.8, 'Seed':argv[-1]})
    cdata.train_expr_ = cdata.train_expr_.sort_index()

    prot_data = pd.read_csv(in_path + 'PNLL-causality-formatted.txt.zip',
                            sep='\t')
    prot_vec = prot_data.ix[prot_data['ID'] == 'TTN', :]
    prot_vec = prot_vec.loc[:, prot_vec.columns.isin(cdata.train_expr_.index)]
    prot_vec = prot_vec.dropna(axis=1)
    use_indx = cdata.train_expr_.index.isin(prot_vec.columns)

    base_cor = spearmanr(
        np.array(prot_vec)[0],
        np.array(cdata.train_expr_.ix[prot_vec.columns,'TTN'])
        )

    mtypes = [
        MuType({('Gene', 'TTN'): {('Form', 'Missense_Mutation'): None}}),
        MuType({('Gene', 'TTN'): {('Form', 'Nonsense_Mutation'): None}}),
        ]

    mut_list = [cdata.train_mut_.status(cdata.train_expr_.index, mtype) for
                mtype in mtypes]

    clf = MKBMTL(path_keys={(((), ('controls-state-change-of', )), )})
    clf.named_steps['fit'].R = 5
    clf.fit_coh(cohort=cdata, mtypes=mtypes)
    H_cor = [
        spearmanr(clf.named_steps['fit'].H_mat['mu'][i,use_indx],
                  np.array(prot_vec)[0])
        for i in range(clf.named_steps['fit'].R)]

    print(clf.named_steps['fit'].bw_mat['mu'].round(2))
    print(clf.eval_coh(cohort=cdata, mtypes=mtypes))

    # saves classifier results to file
    out_file = out_path + argv[0] + '_' + argv[1] + '__run' + argv[-1] + '.p'
    print(out_file)
    out_data = {'H_cor': H_cor, 'base': base_cor}
    pickle.dump(out_data, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

