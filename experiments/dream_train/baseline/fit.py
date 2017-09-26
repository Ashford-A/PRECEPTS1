
"""Finding the downstream expression effect of gene sub-variants.

Args:

Examples:

"""

import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.cohorts import DreamCohort
from HetMan.predict.pipelines import ProteinPipe
import HetMan.predict.regressors as regr

import pickle
import synapseclient


class ElasticNet(regr.ElasticNet, ProteinPipe):
    pass


def main(argv):
    """Runs the experiment."""

    out_dir = os.path.join(base_dir, 'output', argv[0], argv[1], argv[2])
    gene_list = pickle.load(
        open(os.path.join(out_dir, 'tmp', 'gene_list.p'), 'rb'))
    regr_cls = eval(argv[2])

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/CompBio/"
                                "mgrzad/input-data/synapse")

    syn.login()
    cdata = DreamCohort(syn, cohort=argv[0], omic_type=argv[1],
                        cv_seed=(int(argv[3]) + 3) * 37)

    out_rval = {gene: 0 for gene in gene_list}
    for i, gene in enumerate(gene_list):
        if i % 8 == int(argv[4]):

            print(gene)
            regr_obj = regr_cls()

            regr_obj.tune_coh(cdata, gene,
                              tune_splits=4, test_count=32, parallel_jobs=8)
            regr_obj.fit_coh(cdata, gene)

            print(regr_obj)
            out_rval[gene] = regr_obj.eval_coh(cdata, gene)
            print(out_rval[gene])

        else:
            del(out_rval[gene])

    # saves classifier results to file
    out_file = os.path.join(out_dir, 'results',
                            'out__cv-{}_task-{}.p'.format(argv[3], argv[4]))
    pickle.dump(out_rval, open(out_file, 'wb'))


if __name__ == "__main__":
    main(sys.argv[1:])

