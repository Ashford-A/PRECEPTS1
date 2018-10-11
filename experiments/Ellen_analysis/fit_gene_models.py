
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.Ellen_analysis import *
from HetMan.experiments.Ellen_analysis.utils import load_ellen_expression
from HetMan.features.cohorts.patients import PatientMutationCohort
from dryadic.features.mutations import MuType
from dryadic.learning.pipelines import PresencePipe

from dryadic.learning.selection import SelectMeanVar
from dryadic.learning.stan.base import StanOptimizing
from dryadic.learning.stan.logistic import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression

import argparse
import synapseclient
from pathlib import Path
import numpy as np
import dill as pickle


def load_output():
    out_dir = Path(os.path.join(base_dir, 'output', 'gene_models'))

    return [pickle.load(open(str(out_fl), 'rb'))
            for out_fl in out_dir.glob('cv-*.p')]


class OptimModel(BaseLogistic, StanOptimizing):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e4}})


class StanPipe(PresencePipe):

    tune_priors = (
        ('fit__C', tuple(10 ** np.linspace(-3, 6, 24))),
        )

    feat_inst = SelectMeanVar(mean_perc=75)
    norm_inst = StandardScaler()
    fit_inst = LogisticRegression(penalty='l1', max_iter=200,
                                  class_weight='balanced')

    def __init__(self):
        super().__init__([('feat', self.feat_inst), ('norm', self.norm_inst),
                          ('fit', self.fit_inst)])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many training cohort splits to use for tuning'
        )
    parser.add_argument(
        '--test_count', type=int, default=24,
        help='how many hyper-parameter values to test in each tuning split'
        )

    parser.add_argument(
        '--infer_splits', type=int, default=24,
        help='how many cohort splits to use for inference bootstrapping'
        )
    parser.add_argument(
        '--infer_folds', type=int, default=4,
        help=('how many parts to split the cohort into in each inference '
              'cross-validation run')
        )

    parser.add_argument(
        '--parallel_jobs', type=int, default=12,
        help='how many parallel CPUs to allocate the tuning tests across'
        )

    parser.add_argument('--cv_id', type=int, default=0)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    out_dir = os.path.join(base_dir, 'output', 'gene_models')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir,
                            'cv-{}.p'.format(args.cv_id))

    #test_dir = os.path.join(base_dir, '..', 'mut_baseline', 'output',
    #                        'Firehose', 'PAAD__samps-25')
    #test_models = os.listdir(test_dir)

    #test_dict = dict()
    #for test_model in test_models:

    #    test_fls = [
    #        test_fl for test_fl in os.listdir(
    #            os.path.join(test_dir, test_model))
    #        if 'out__' in test_fl
    #        ]

    #    log_fls = [
    #        log_fl for log_fl in os.listdir(os.path.join(
    #            test_dir, test_model, 'slurm'))
    #        if 'fit-' in log_fl
    #        ]

    #    if len(log_fls) > 0 and len(log_fls) == (len(test_fls) * 2):
    #        test_dict[test_model] = load_baseline('Firehose', 'PAAD', 25,
    #                                              test_model)[0]

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    mut_clf = StanPipe()
    test_mtypes = [
        MuType({('Gene', 'TP53'): {('Scale', 'Point'): None}}),
        MuType({('Gene', 'CDH1'): {('Scale', 'Point'): None}}),
        MuType({('Gene', 'ERBB2'): {('Copy', 'HomGain'): None}}),
        MuType({('Gene', 'HIST1H2AC'): {('Copy', 'HetDel'): None}})
        ]
    use_genes = {mtype.subtype_list()[0][0] for mtype in test_mtypes}

    cdata = PatientMutationCohort(
        patient_expr=load_ellen_expression(ellen_data), patient_muts=None,
        tcga_cohort='BRCA', mut_genes=use_genes, mut_levels=['Gene', 'Form'],
        expr_source='toil', var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=toil_dir, copy_dir=copy_dir,
        cv_seed=(args.cv_id * 59) + 121, cv_prop=1.0,
        collapse_txs=True, syn=syn
        )

    tuned_params = {mtype: None for mtype in test_mtypes}
    infer_mats = {mtype: None for mtype in test_mtypes}

    for mtype in test_mtypes:
        mut_gene = mtype.subtype_list()[0][0]
        ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                    if annot['chr'] == cdata.gene_annot[mut_gene]['chr']}

        mut_clf.tune_coh(
            cdata, mtype,
            exclude_genes=ex_genes, exclude_samps=cdata.patient_samps,
            tune_splits=args.tune_splits, test_count=args.test_count,
            parallel_jobs=args.parallel_jobs
            )

        clf_params = mut_clf.get_params()
        tuned_params[mtype] = {par: clf_params[par]
                               for par, _ in StanPipe.tune_priors}
        print(tuned_params)
        
        infer_mats[mtype] = mut_clf.infer_coh(
            cdata, mtype,
            force_test_samps=ellen_samps, exclude_genes=ex_genes,
            infer_splits=args.infer_splits, infer_folds=args.infer_folds
            )

    pickle.dump(
        {'Infer': infer_mats, 'Tune': tuned_params},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

