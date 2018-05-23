
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.predict.basic.classifiers import *
from HetMan.experiments.SMMART_analysis.utils import load_patient_expression

from importlib import import_module
from functools import reduce
from operator import and_
import pandas as pd

import synapseclient
import argparse
import dill as pickle


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('gene', type=str, help='a mutated gene')
    parser.add_argument('classif', type=str, help='a mutated gene')

    parser.add_argument(
        'toil_dir', type=str,
        help='the directory where toil expression data is saved'
        )
    parser.add_argument('syn_root', type=str,
                        help='Synapse cache root directory')
    parser.add_argument(
        'patient_dir', type=str,
        help='directy where SMMART patient RNAseq abundances are stored'
        )

    parser.add_argument('--cv_id', type=int, default=0)
    parser.add_argument(
        '--tune_splits', type=int, default=8,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=32,
        help='how many hyper-parameter values to test in each tuning split'
        )
    parser.add_argument(
        '--parallel_jobs', type=int, default=4,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    out_dir = os.path.join(base_dir, 'output', 'gene_models',
                           args.cohort, args.gene)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir,
                            '{}__cv-{}.p'.format(args.classif, args.cv_id))

    if args.classif[:6] == 'Stan__':
        use_module = import_module('HetMan.experiments.utilities'
                                   '.stan_models.{}'.format(
                                       args.classif.split('Stan__')[1]))
        mut_clf = getattr(use_module, 'UsePipe')
    
    else:
        mut_clf = eval(args.classif)

    base_mtype = MuType({('Gene', args.gene): None})
    clf = mut_clf()
    expr_dict = load_patient_expression(args.patient_dir)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = args.syn_root
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='toil', expr_dir=args.toil_dir, syn=syn,
        collapse_txs=True, cv_seed=(args.cv_id * 59) + 121, cv_prop=0.8
        )

    gene_ids = {
        gn: cdata.feat_annot[gn]['gene_id'].replace('"', '').split('.')[0]
        for gn in cdata.subset_genes(exclude_genes={args.gene})
        }

    use_ids = reduce(and_, [set(x) for x in expr_dict.values()])
    use_ids &= set(gene_ids.values())
    use_genes = {gn: gene_id for gn, gene_id in gene_ids.items()
                 if gene_id in use_ids}

    expr_df = pd.DataFrame.from_dict(
        {patient: {gn: expr[gene_id] for gn, gene_id in use_genes.items()}
         for patient, expr in expr_dict.items()},
        orient='index'
        ).loc[:, cdata.subset_genes(include_genes=set(use_genes))]

    clf.tune_coh(
        cdata, base_mtype, include_genes=set(use_genes),
        tune_splits=args.tune_splits, test_count=args.test_count,
        parallel_jobs=args.parallel_jobs
        )

    clf.fit_coh(cdata, base_mtype, include_genes=set(use_genes))
    clf_params = clf.get_params()
    tuned_params = {par: clf_params[par] for par, _ in mut_clf.tune_priors}

    tcga_vals = {
        samp: val for samp, val in zip(
            sorted(cdata.test_samps),
            clf.predict_test(cdata, base_mtype, include_genes=set(use_genes))
            )
        }

    patient_vals = {
        samp: val for samp, val in zip(expr_df.index,
                                       clf.predict_omic(expr_df))
        }

    pickle.dump(
        {'TCGAvals': tcga_vals, 'SMMARTvals': patient_vals,
         'Info': {'TunePriors': mut_clf.tune_priors,
                  'TuneSplits': args.tune_splits,
                  'TestCount': args.test_count,
                  'TunedParams': tuned_params}},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

