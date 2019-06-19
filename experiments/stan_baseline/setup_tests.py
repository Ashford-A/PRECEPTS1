
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

from HetMan.experiments.stan_baseline import *
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.cohorts.beatAML import BeatAmlCohort
from dryadic.features.mutations import MuType

import argparse
import synapseclient
import dill as pickle
from itertools import product


def get_cohort_data(cohort, use_gene, cv_seed=None, test_prop=0):
    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    if cohort == 'beatAML':
        cdata = BeatAmlCohort(mut_genes=[use_gene],
                              mut_levels=['Gene', 'Form', 'Exon', 'Protein'],
                              expr_file=beatAML_files['expr'],
                              samp_file=beatAML_files['samps'], syn=syn,
                              annot_file=annot_file, cv_seed=cv_seed,
                              test_prop=test_prop)

    else:
        cdata = MutationCohort(
            cohort=cohort.split('_')[0],
            mut_levels=['Form_base', 'Protein'],
            mut_genes=[use_gene], expr_source='Firehose',
            var_source='mc3', copy_source='Firehose', annot_file=annot_file,
            type_file=type_file, expr_dir=expr_dir, copy_dir=copy_dir,
            syn=syn, cv_seed=cv_seed, test_prop=test_prop,
            annot_fields=['transcript'], use_types=parse_subtypes(cohort)
            )

    return cdata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument('gene', type=str, help="which TCGA cohort to use")
    parser.add_argument('--setup_dir', type=str, default=base_dir)

    args = parser.parse_args()
    out_path = os.path.join(args.setup_dir, 'setup')
    cdata = get_cohort_data(args.cohort, args.gene)
    vars_list = set()

    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f)

    # find subsets of point mutations with enough affected samples for each
    # mutated gene in the cohort
    if 'Point' in dict(cdata.mtree) and len(cdata.mtree['Point']) >= 20:
        vars_list |= cdata.mtree['Point'].branchtypes(min_size=20)
        if len(dict(cdata.mtree['Point'])) > 1:
            vars_list |= {MuType({('Scale', 'Point'): None})}

    if 'Copy' in dict(cdata.mtree):
        if 'DeepDel' in dict(cdata.mtree['Copy']):
            if len(cdata.mtree['Copy']['DeepDel']) >= 20:
                vars_list |= {MuType({('Copy', 'DeepDel'): None})}

        if 'DeepGain' in dict(cdata.mtree['Copy']):
            if len(cdata.mtree['Copy']['DeepGain']) >= 20:
                vars_list |= {MuType({('Copy', 'DeepGain'): None})}

    # filter out mutations that do not have enough wild-type samples
    vars_list = {mtype for mtype in vars_list
                 if (len(mtype.get_samples(cdata.mtree))
                     <= (len(cdata.get_samples()) - 20))}

    # remove mutations that are functionally equivalent to another mutation
    vars_list -= {mtype1 for mtype1, mtype2 in product(vars_list, repeat=2)
                  if (mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                      and (mtype1.get_samples(cdata.mtree)
                           == mtype2.get_samples(cdata.mtree)))}

    # save the enumerated mutations, and the number of such mutations, to file
    with open(os.path.join(out_path, "vars-list.p"), 'wb') as fl:
        pickle.dump(sorted(vars_list), fl)
    with open(os.path.join(out_path, "vars-count.txt"), 'w') as f:
        f.write(str(len(vars_list)))


if __name__ == '__main__':
    main()

