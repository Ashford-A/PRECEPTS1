
"""Finding the expression signature of a list of variant types in a cohort.

This script takes a list of variant sub-types and find the pattern of
expression perturbation that a given classification pipeline associates with
their presence in a cohort of samples. A sub-type can be any subset of the
mutations present in a gene or a group of genes, as defined by shared
properties. These properties can include form (i.e. splice site mutations,
missense mutations, frameshift mutations), location (i.e. 5th exon, 123rd
protein), PolyPhen score, and so on.

To allow for parallelization, we split the list of sub-types into equally
sized tasks that are each tested on a separate cluster array job. The split
is done by taking the modulus of each type's position in the given master list
of types. We repeat this process for multiple splits of the TCGA cohort into
training/testing cohorts, as defined by the given cross-validation ID.

Args:

Examples:

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts import VariantCohort
from HetMan.predict.classifiers import *

import numpy as np
import pandas as pd

import synapseclient
import dill as pickle
from functools import reduce
from glob import glob
import os

firehose_dir = '/home/exacloud/lustre1/CompBio/mgrzad/input-data/firehose'


def load_output(out_dir):

    out_list = [
        [pickle.load(open(fl, 'rb'))
         for fl in sorted(glob(os.path.join(
             out_dir, "results/out__cv-*_task-{}*".format(task_id))))]
        for task_id in range(6)
        ]

    acc_data = pd.concat(
        pd.concat([pd.DataFrame.from_dict(x['Acc'], orient='index')
                   for x in ols],
                  axis=1)
        for ols in out_list
        )

    coef_data = pd.DataFrame()
    for ols in out_list:
        coef_list = [pd.DataFrame.from_dict(x['Coef'], orient='index')
                     for x in ols]   

        cmn_gns = reduce(lambda x, y: x & y, [x.columns for x in coef_list])
        cmn_types = reduce(lambda x, y: x & y, [x.index for x in coef_list])

        coef_mat = reduce(lambda x, y: x.where(x > y, y),
                          [ls.loc[cmn_types, cmn_gns].fillna(0.0)
                           for ls in coef_list])

        coef_data = pd.concat([coef_data, coef_mat])

    coef_data = coef_data.fillna(0.0)

    return acc_data, coef_data


def main(argv):
    """Runs the experiment."""
    print(argv)

    # gets the directory where output will be saved and the name of the TCGA
    # cohort under consideration, loads the list of gene sub-variants 
    mtype_list = pickle.load(
        open(os.path.join(argv[0], 'tmp', 'mtype_list.p'), 'rb'))

    # loads the pipeline used for classifying variants, gets the mutated
    # genes for each variant under consideration
    mut_clf = eval(argv[2])
    use_genes = reduce(lambda x, y: x | y,
                       [set(dict(mtype).keys()) for mtype in mtype_list])

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation id for this task
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")
    syn.login()

    cdata = VariantCohort(
        cohort=argv[1], mut_genes=list(use_genes),
        mut_levels=['Gene', 'Form_base', 'Exon', 'Protein'],
        expr_source='Firehose', data_dir=firehose_dir,
        cv_seed=(int(argv[3]) + 5) * 31, cv_prop=0.9, syn=syn
        )

    out_acc = {mtype: -1 for mtype in mtype_list}
    out_coef = {mtype: {gn: 0 for gn in cdata.genes} for mtype in mtype_list}

    # for each sub-variant, check if it has been assigned to this task
    for i, mtype in enumerate(mtype_list):
        if i % 6 == int(argv[4]):
            print(mtype)

            # gets the genes that this variant mutates, initializes the
            # classification pipeline
            ex_genes = list(dict(mtype).keys())
            clf = mut_clf()

            # tunes the classifier using the training cohort
            clf.tune_coh(cdata, mtype, exclude_genes=ex_genes,
                         tune_splits=12, test_count=36, parallel_jobs=12)

            # fits the tuned classifier on the training cohort, evaluates its
            # performance on the testing cohort and saves the results
            clf.fit_coh(cdata, mtype, exclude_genes=ex_genes)
            out_acc[mtype] = clf.eval_coh(cdata, mtype,
                                          exclude_genes=ex_genes)
            out_coef[mtype].update({gn: coef for gn, coef in
                                    clf.get_coef().items()})

            out_coef[mtype] = {
                gn: coef for gn, coef in out_coef[mtype].items()
                if coef != 0
                }

        else:
            del(out_acc[mtype])
            del(out_coef[mtype])

    # saves the performance measurements for each variant to file
    out_file = os.path.join(argv[0], 'results',
                            'out__cv-{}_task-{}.p'.format(argv[3], argv[4]))
    pickle.dump({'Acc': out_acc, 'Coef': out_coef},
                open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

