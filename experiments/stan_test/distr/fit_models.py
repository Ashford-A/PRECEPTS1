
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../../..')])

from HetMan.features.variants import MuType
from HetMan.features.cohorts.mut import VariantCohort
from HetMan.predict.stan_margins import *

import synapseclient
import dill as pickle
import numpy as np
import argparse

firehose_dir = '/home/exacloud/lustre1/share_your_data_here/precepts/firehose'


def main():
    """Runs the experiment."""

    parser = argparse.ArgumentParser(
        description=("Find the signatures a classifier predicts for a list "
                     "of sub-types.")
        )

    parser.add_argument('model_name', type=str,
                        help='the name of a Stan model')
    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('gene', type=str, help='a gene with mutated samples')

    parser.add_argument('cv_id', type=int,
                        help='a random seed used for cross-validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')
    args = parser.parse_args()

    out_path = os.path.join(base_dir, 'output',
                            args.model_name, args.cohort, args.gene)

    use_mtype = MuType({('Gene', args.gene): None})
    clf_stan = eval("model_dict['{}']".format(args.model_name))

    if args.verbose:
        print("Starting signature portrayal for cross-validation "
              "ID {} ...".format(args.cv_id))
        print('Using the following Stan model:\n\n{}'.format(
            clf_stan.named_steps['fit'].model_code))

    # logs into Synapse using locally-stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = ("/home/exacloud/lustre1/share_your_data_here/"
                                "precepts/synapse/")
    syn.login()

    # loads the expression data and gene mutation data for the given TCGA
    # cohort, with the training/testing cohort split defined by the
    # cross-validation ID for this sub-job
    cdata = VariantCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', data_dir=firehose_dir, syn=syn,
        cv_prop=1.0, cv_seed=1298 + 93 * args.cv_id
        )

    infer_mat = np.array(clf_stan.infer_coh(
                cdata, use_mtype, exclude_genes=set([args.gene]),
                infer_splits=4, infer_folds=4, parallel_jobs=4
                ))

    pickle.dump(infer_mat,
                open(os.path.join(
                    out_path, 'out__cv-{}.p'.format(args.cv_id)
                    ), 'wb')
               )


base_model = '''
    data {
        int<lower=1> N;         // number of samples
        int<lower=1> G;         // number of genetic features
        matrix[N, G] expr;      // RNA-seq expression values
        int<lower=0, upper=1> mut[N];   // mutation status
    }

    parameters {
        real alpha;
        vector[G] gn_wghts;
    }

    model {
        alpha ~ normal(0, 1);
        gn_wghts ~ normal(0, 1);
        mut ~ bernoulli_logit(alpha + expr * gn_wghts);
    }
'''

class LogitOptim(StanOptimizing, LogitStan):
        pass


model_dict = {
    'base': StanPipe(LogitOptim(base_model))
    }


if __name__ == "__main__":
    main()

