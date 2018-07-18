
import os
base_dir = os.path.dirname(__file__)

import sys
sys.path.extend([os.path.join(base_dir, '../../..')])
from HetMan.experiments.cna_isolate import *

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.mutations import MuType
from HetMan.experiments.utilities.classifiers import *

import argparse
import synapseclient
from importlib import import_module
import dill as pickle

import numpy as np
import pandas as pd
from glob import glob


def get_output_files(out_dir):
    file_list = glob(os.path.join(out_dir, 'out__task-*.p'))

    base_names = [os.path.basename(fl).split('out__')[1] for fl in file_list]
    task_ids = [int(nm.split('task-')[1].split('.p')[0]) for nm in base_names]

    return file_list, task_ids


def load_infer_output(out_dir):
    file_list, task_ids = get_output_files(out_dir)

    out_df = pd.concat([
        pd.DataFrame.from_dict(pickle.load(open(fl, 'rb'))['Infer'],
                               orient='index')
        for fl in file_list
        ])
 
    if hasattr(out_df.iloc[0, 0][0], '__len__'):
        out_df = out_df.applymap(lambda x: [y[0] for y in x])

    if all(isinstance(x, tuple) for x in out_df.index):
        out_df.index = pd.MultiIndex.from_tuples(out_df.index)

    return out_df.sort_index()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cohort', type=str, help='a TCGA cohort')
    parser.add_argument('gene', type=str, help='a mutated gene')
    parser.add_argument('classif', type=str,
                        help='a classifier in HetMan.predict.classifiers')

    parser.add_argument(
        '--cv_id', type=int, default=9981,
        help='the random seed to use for cross-validation draws'
        )

    parser.add_argument(
        '--task_count', type=int, default=10,
        help='how many parallel tasks the list of types to test is split into'
        )
    parser.add_argument('--task_id', type=int, default=0,
                        help='the subset of subtypes to assign to this task')

    # optional arguments controlling how classifier tuning is to be performed
    parser.add_argument(
        '--tune_splits', type=int, default=4,
        help='how many cohort splits to use for tuning'
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

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='turns on diagnostic messages')

    args = parser.parse_args()
    out_dir = os.path.join(base_dir, 'output',
                           args.cohort, args.gene, args.classif)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, 'out__task-{}.p'.format(args.task_id))
    ctf_list = pickle.load(
        open(os.path.join(base_dir, 'setup', 'ctf_lists',
                          '{}_{}.p'.format(args.cohort, args.gene)),
             'rb')
        )

    if args.classif[:6] == 'Stan__':
        use_module = import_module('HetMan.experiments.utilities'
                                   '.stan_models.{}'.format(
                                       args.classif.split('Stan__')[1]))
        mut_clf = getattr(use_module, 'UsePipe')

    else:
        mut_clf = eval(args.classif)

    # log into Synapse using locally stored credentials
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    cdata = MutationCohort(
        cohort=args.cohort, mut_genes=[args.gene], mut_levels=['Gene'],
        expr_source='Firehose', var_source='mc3', expr_dir=firehose_dir,
        copy_source='Firehose', copy_dir=copy_dir, copy_discrete=False,
        syn=syn, cv_seed=args.cv_id, cv_prop=1.0
        )

    base_mtype = MuType({('Gene', args.gene): None})
    base_samps = base_mtype.get_samples(cdata.train_mut)
    ex_genes = {gene for gene, annot in cdata.gene_annot.items()
                if annot['chr'] == cdata.gene_annot[args.gene]['chr']}

    out_iso = {ctf: None for ctf in ctf_list}
    out_tune = {ctf: {par: None for par, _ in mut_clf.tune_priors}
                for ctf in ctf_list}

    mut_stat = np.array(cdata.train_mut.status(cdata.copy_data.index))
    copy_vals = cdata.copy_data.loc[~mut_stat, args.gene]

    for i, ctf in enumerate(ctf_list):
        if (i % args.task_count) == args.task_id:
            if args.verbose:
                print("Isolating {} {} ...".format(*ctf))

            clf = mut_clf()
            ex_samps = set(copy_vals.index[copy_vals.between(*ctf)])
            ex_samps |= base_samps

            if ctf[0] < 0:
                ex_samps |= set(copy_vals.index[copy_vals > -ctf[1]])
                use_pheno = {'Gene': args.gene, 'CNA': 'Loss',
                             'Cutoff': ctf[0]}

            else:
                ex_samps |= set(copy_vals.index[copy_vals < -ctf[0]])
                use_pheno = {'Gene': args.gene, 'CNA': 'Gain',
                             'Cutoff': ctf[1]}

            clf.tune_coh(
                cdata, use_pheno,
                exclude_genes=ex_genes, exclude_samps=ex_samps,
                tune_splits=args.tune_splits, test_count=args.test_count,
                parallel_jobs=args.parallel_jobs
                )

            clf_params = clf.get_params()
            for par, _ in mut_clf.tune_priors:
                out_tune[ctf][par] = clf_params[par]

            out_iso[ctf] = clf.infer_coh(
                cdata, use_pheno,
                exclude_genes=ex_genes, force_test_samps=ex_samps,
                infer_splits=args.infer_splits, infer_folds=args.infer_folds,
                parallel_jobs=args.parallel_jobs
                )

        else:
            del(out_iso[ctf])

    pickle.dump(
        {'Infer': out_iso, 'Tune': out_tune,
         'Info': {'Clf': mut_clf,
                  'TunePriors': mut_clf.tune_priors,
                  'TuneSplits': args.tune_splits,
                  'TestCount': args.test_count}},
        open(out_file, 'wb')
        )


if __name__ == "__main__":
    main()

