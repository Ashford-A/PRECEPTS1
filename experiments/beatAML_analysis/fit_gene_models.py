
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.beatAML_analysis import *
from HetMan.experiments.beatAML_analysis.utils import load_beat_expression
from HetMan.features.cohorts.patients import PatientMutationCohort
from dryadic.features.mutations import MuType
from dryadic.learning.pipelines import PresencePipe

from dryadic.learning.selection import SelectMeanVar
from dryadic.learning.stan.base import StanOptimizing
from dryadic.learning.stan.margins import *
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import argparse
import synapseclient
from pathlib import Path
import dill as pickle


def load_output():
    out_dir = Path(os.path.join(base_dir, 'output', 'gene_models'))

    return [pickle.load(open(str(out_fl), 'rb'))
            for out_fl in out_dir.glob('cv-*.p')]


class OptimModel(GaussLabels, StanOptimizing):

    def run_model(self, **fit_params):
        super().run_model(**{**fit_params, **{'iter': 1e4}})

    def predict_proba(self, X):
        return self.calc_pred_labels(X)


class StanPipe(PresencePipe):

    tune_priors = (
        ('fit__alpha', tuple(10 ** np.linspace(-4, -2, 51))),
        )

    feat_inst = SelectMeanVar(mean_perc=80)
    norm_inst = StandardScaler()
    fit_inst = OptimModel(model_code=gauss_model)

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
        '--test_count', type=int, default=12,
        help='how many hyper-parameter values to test in each tuning split'
        )

    parser.add_argument(
        '--infer_splits', type=int, default=12,
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

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    mut_clf = StanPipe()
    gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)

    use_genes = gene_df.index[
        (gene_df.loc[:, ['Vogelstein', 'Sanger CGC',
                         'Foundation One', 'MSK-IMPACT']]
         == 'Yes').sum(axis=1) > 1
        ]

    cdata = PatientMutationCohort(
        patient_expr=load_beat_expression(beataml_expr), patient_muts=None,
        tcga_cohort='LAML', mut_genes=use_genes, mut_levels=['Gene', 'Form'],
        expr_source='toil', var_source='mc3', copy_source='Firehose',
        annot_file=annot_file, expr_dir=toil_dir, copy_dir=copy_dir,
        cv_seed=(args.cv_id * 59) + 121, cv_prop=1.0,
        collapse_txs=False, syn=syn
        )

    use_mtypes = set()
    for gene, muts in cdata.train_mut:
        if len(muts) >= 15:

            if 'Copy' in dict(muts) and len(muts['Copy']) >= 15:
                if 'HomDel' in dict(muts['Copy']):
                    if len(muts['Copy']['HomDel']) >= 15:
                        use_mtypes |= {MuType({('Gene', gene): {
                            ('Scale', 'Copy'): {('Copy', 'HomDel'): None}}})}

                if 'HomGain' in dict(muts['Copy']):
                    if len(muts['Copy']['HomGain']) >= 15:
                        use_mtypes |= {MuType({('Gene', gene): {
                            ('Scale', 'Copy'): {('Copy', 'HomGain'): None}}})}

                loss_mtype = MuType({('Copy', ('HomDel', 'HetDel')): None})
                if 'HetDel' in dict(muts['Copy']):
                    if len(loss_mtype.get_samples(muts['Copy'])) >= 15:
                        use_mtypes |= {MuType({('Gene', gene): {
                            ('Scale', 'Copy'): loss_mtype}})}

                gain_mtype = MuType({('Copy', ('HomGain', 'HetGain')): None})
                if 'HetGain' in dict(muts['Copy']):
                    if len(gain_mtype.get_samples(muts['Copy'])) >= 15:
                        use_mtypes |= {MuType({('Gene', gene): {
                            ('Scale', 'Copy'): gain_mtype}})}

            if 'Point' in dict(muts) and len(muts['Point']) >= 15:
                use_mtypes |= {MuType({('Gene', gene): {
                    ('Scale', 'Point'): None}})}

                use_mtypes |= {
                    MuType({('Gene', gene): {('Scale', 'Point'): mtype}})
                    for mtype in muts['Point'].branchtypes(min_size=15)
                    }

    tuned_params = {mtype: None for mtype in use_mtypes}
    infer_mats = {mtype: None for mtype in use_mtypes}

    for mtype in use_mtypes:
        mut_gene = mtype.subtype_list()[0][0]
        ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                    if annot['chr'] == cdata.gene_annot[mut_gene]['chr']}

        mut_clf.tune_coh(
            cdata, mtype,
            exclude_genes=ex_genes, exclude_samps=cdata.patient_samps,
            tune_splits=args.tune_splits, test_count=args.test_count,
            parallel_jobs=args.parallel_jobs
            )
        print(mut_clf)

        clf_params = mut_clf.get_params()
        tuned_params[mtype] = {par: clf_params[par]
                               for par, _ in StanPipe.tune_priors}
        
        infer_mats[mtype] = mut_clf.infer_coh(
            cdata, mtype,
            force_test_samps=cdata.patient_samps, exclude_genes=ex_genes,
            infer_splits=args.infer_splits, infer_folds=args.infer_folds
            )

    pickle.dump(
        {'Infer': infer_mats, 'Tune': tuned_params},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

