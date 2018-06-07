
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.SMMART_analysis.cohorts import CancerCohort
from HetMan.features.mutations import MuType
from HetMan.predict.basic.classifiers import *

import argparse
import synapseclient
from importlib import import_module

from pathlib import Path
import dill as pickle


def load_output(cohort, gene, classif):
    out_dir = Path(os.path.join(base_dir, 'output', 'gene_models',
                                cohort, gene))

    return [pickle.load(open(str(out_fl), 'rb'))
            for out_fl in out_dir.glob('{}__cv-*.p'.format(classif))]


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

    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=16,
        help='how many hyper-parameter values to test in each tuning split'
        )

    parser.add_argument(
        '--infer_splits', type=int, default=20,
        help='how many cohort splits to use for inference bootstrapping'
        )
    parser.add_argument(
        '--infer_folds', type=int, default=4,
        help=('how many parts to split the cohort into in each inference '
              'cross-validation run')
        )

    parser.add_argument(
        '--parallel_jobs', type=int, default=4,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--cv_id', type=int, default=0)
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

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = args.syn_root
    syn.login()

    cdata = CancerCohort(
        cancer=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        tcga_dir=args.toil_dir, patient_dir=args.patient_dir, syn=syn,
        collapse_txs=True, cv_seed=(args.cv_id * 59) + 121, cv_prop=1.0
        )
    smrt_samps = {samp for samp in cdata.samples if samp[:4] != 'TCGA'}

    clf.tune_coh(
        cdata, base_mtype,
        exclude_genes={args.gene}, exclude_samps=smrt_samps,
        tune_splits=args.tune_splits, test_count=args.test_count,
        parallel_jobs=args.parallel_jobs
        )

    clf_params = clf.get_params()
    tuned_params = {par: clf_params[par] for par, _ in mut_clf.tune_priors}

    infer_mat = clf.infer_coh(
        cdata, base_mtype,
        force_test_samps=smrt_samps, exclude_genes={args.gene},
        infer_splits=args.infer_splits, infer_folds=args.infer_folds
        )

    pickle.dump(
        {'Infer': infer_mat,
         'Info': {'TunePriors': mut_clf.tune_priors,
                  'TuneSplits': args.tune_splits,
                  'TestCount': args.test_count,
                  'TunedParams': tuned_params}},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

