
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '..', '..', '..')])

from HetMan.experiments.subvariant_test import *
from HetMan.experiments.subvariant_test import (
    pnt_mtype, copy_mtype, gain_mtype, loss_mtype)
from HetMan.experiments.subvariant_tour.utils import RandomType
from HetMan.experiments.utilities.load_input import parse_subtypes
from HetMan.features.cohorts.tcga import list_cohorts

from HetMan.features.cohorts.tcga import MutationCohort
from HetMan.features.cohorts.beatAML import BeatAmlCohort
from HetMan.features.cohorts.metabric import MetabricCohort
from HetMan.features.cohorts.ccle import CellLineCohort
from dryadic.features.mutations import MuType

import numpy as np
import pandas as pd

import argparse
import synapseclient
import dill as pickle
import subprocess

import random
from itertools import product


def get_cohort_data(cohort, expr_source, mut_lvls=None, use_genes=None,
                    leaf_annot=None, gene_annot=None):
    if mut_lvls is None:
        mut_lvls = [['Gene', 'Scale', 'Copy', 'Exon', 'Location', 'Protein']]

    if use_genes is None:
        gene_df = pd.read_csv(gene_list, sep='\t', skiprows=1, index_col=0)
        use_genes = gene_df.index[
            (gene_df.loc[
                :, ['Vogelstein', 'SANGER CGC(05/30/2017)',
                    'FOUNDATION ONE', 'MSK-IMPACT']]
                == 'Yes').sum(axis=1) > 1
            ].tolist()

    else:
        use_genes = list(use_genes)

    if leaf_annot is None:
        leaf_annot = ('ref_count', 'alt_count')

    if gene_annot is None:
        gene_annot = ['transcript', 'exon']

    syn = synapseclient.Synapse()
    syn.cache.cache_root_dir = syn_root
    syn.login()

    if cohort == 'beatAML':
        if expr_source != 'toil__gns':
            raise ValueError("Only gene-level Kallisto calls are available "
                             "for the beatAML cohort!")

        cdata = BeatAmlCohort(
            mut_levels=mut_lvls, mut_genes=use_genes,
            expr_source=expr_source, expr_file=beatAML_files['expr'],
            samp_file=beatAML_files['samps'], syn=syn, annot_file=annot_file,
            domain_dir=domain_dir, leaf_annot=leaf_annot,
            cv_seed=8713, test_prop=0, annot_fields=gene_annot
            )

    elif cohort.split('_')[0] == 'METABRIC':
        if expr_source != 'microarray':
            raise ValueError("Only Illumina microarray mRNA calls are "
                             "available for the METABRIC cohort!")

        if '_' in cohort:
            use_subtypes = cohort.split('_')[1]
        else:
            use_subtypes = None

        cdata = MetabricCohort(
            mut_levels=mut_lvls, mut_genes=use_genes,
            expr_source=expr_source, metabric_dir=metabric_dir,
            annot_file=annot_file, domain_dir=domain_dir,
            use_types=use_subtypes, cv_seed=8713, test_prop=0,
            annot_fields=gene_annot
            )

    elif cohort.split('_')[0] == 'CCLE':
        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        if source_base == 'microarray':
            expr_dir = ccle_dir
        else:
            expr_dir = expr_sources[source_base]

        cdata = CellLineCohort(
            mut_levels=mut_lvls, mut_genes=use_genes, expr_source=source_base,
            ccle_dir=ccle_dir, annot_file=annot_file, domain_dir=domain_dir,
            expr_dir=expr_dir, collapse_txs=collapse_txs,
            cv_seed=8713, test_prop=0
            )

    else:
        source_info = expr_source.split('__')
        source_base = source_info[0]
        collapse_txs = not (len(source_info) > 1 and source_info[1] == 'txs')

        cdata = MutationCohort(
            cohort=cohort.split('_')[0], mut_levels=mut_lvls,
            mut_genes=use_genes, expr_source=source_base, var_source='mc3',
            copy_source='Firehose', annot_file=annot_file,
            domain_dir=domain_dir, type_file=type_file, leaf_annot=leaf_annot,
            expr_dir=expr_sources[source_base], copy_dir=copy_dir,
            collapse_txs=collapse_txs, syn=syn, cv_seed=8713, test_prop=0,
            annot_fields=gene_annot, use_types=parse_subtypes(cohort)
            )

    return cdata


def load_cohort(cohort, expr_source, use_path=None, **coh_args):
    if use_path is not None and os.path.exists(use_path):
        try:
            with open(use_path, 'rb') as f:
                cdata = pickle.load(f)

        except:
            cdata = get_cohort_data(cohort, expr_source, **coh_args)

    else:
        cdata = get_cohort_data(cohort, expr_source, **coh_args)

    return cdata


def main():
    parser = argparse.ArgumentParser("Enumerate the subgroupings for which "
                                     "expression signatures will be tested.")

    parser.add_argument('expr_source', type=str,
                        help="which TCGA expression data source to use")

    parser.add_argument('cohort', type=str, help="which TCGA cohort to use")
    parser.add_argument(
        'samp_cutoff', type=int,
        help="minimum number of affected samples needed to test a mutation"
        )

    parser.add_argument('mut_levels', type=str,
                        help="the mutation property levels to consider")
    parser.add_argument('out_dir', type=str, default=base_dir)

    # parse command line arguments, figure out where output will be stored
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, 'setup')
    base_path = os.path.join(args.out_dir.split('subvariant_test')[0],
                             'subvariant_test')

    # figure out where to save tumour cohorts used for training and
    # testing, load training expression and mutation data
    coh_dir = os.path.join(base_path, args.expr_source, 'setup')
    coh_path = os.path.join(coh_dir, "cohort-data__{}.p".format(args.cohort))
    cdata = load_cohort(args.cohort, args.expr_source, coh_path)

    # figure out which mutation attributes to use
    use_lvls = tuple(args.mut_levels.split('__'))
    if 'Copy' in cdata.muts['Scale'].values:
        lbls_key = ('Gene', 'Scale', 'Copy') + use_lvls
    else:
        lbls_key = ('Gene', 'Scale') + use_lvls

    # initialize list of enumerated subgroupings, get number of samples in
    # the cohort, load required mutation attributes in training dataset
    use_mtypes = set()
    n_samps = len(cdata.get_samples())
    if lbls_key not in cdata.mtrees:
        cdata.add_mut_lvls(lbls_key)

    # save training tumour cohort to file for use in this experiment, and in
    # duplicate for when other iterations of this experiment need it
    with open(coh_path, 'wb') as f:
        pickle.dump(cdata, f, protocol=-1)
    with open(os.path.join(out_path, "cohort-data.p"), 'wb') as f:
        pickle.dump(cdata, f, protocol=-1)

    # for each gene with enough samples harbouring its point mutations in the
    # cohort, find the subgroupings composed of at most two branches
    for gene, muts in cdata.mtrees[lbls_key]:
        if len(pnt_mtype.get_samples(muts)) >= args.samp_cutoff:
            pnt_mtypes = {
                mtype for mtype in muts['Point'].combtypes(
                    comb_sizes=(1, 2), min_type_size=args.samp_cutoff)
                if (args.samp_cutoff <= len(mtype.get_samples(muts))
                    <= (n_samps - args.samp_cutoff))
                }

            # remove subgroupings that have only one child subgrouping
            # containing all of their samples
            pnt_mtypes -= {
                mtype1 for mtype1, mtype2 in product(pnt_mtypes, repeat=2)
                if mtype1 != mtype2 and mtype1.is_supertype(mtype2)
                and mtype1.get_samples(muts) == mtype2.get_samples(muts)
                }

            # remove groupings that contain all of the gene's point mutations
            pnt_mtypes = {MuType({('Scale', 'Point'): mtype})
                          for mtype in pnt_mtypes
                          if (len(mtype.get_samples(muts['Point']))
                              < len(muts['Point'].get_samples()))}

            # check if this gene had at least five samples with deep gains or
            # deletions that weren't all already carrying point mutations
            copy_mtypes = {
                mtype for mtype in [gain_mtype, loss_mtype]
                if ((5 <= len(mtype.get_samples(muts)) <= (n_samps - 5))
                    and not (mtype.get_samples(muts)
                             <= muts['Point'].get_samples())
                    and not (muts['Point'].get_samples()
                             <= mtype.get_samples(muts)))
                }

            # find the enumerated point mutations for this gene that can be
            # combined with CNAs to produce a novel set of mutated samples
            dyad_mtypes = {
                pnt_mtype | cp_mtype
                for pnt_mtype, cp_mtype in product(pnt_mtypes, copy_mtypes)
                if ((pnt_mtype.get_samples(muts) - cp_mtype.get_samples(muts))
                    and (cp_mtype.get_samples(muts)
                         - pnt_mtype.get_samples(muts)))
                }

            # if we are using the base list of mutation attributes, add the
            # gene-wide set of all point mutations...
            gene_mtypes = pnt_mtypes | dyad_mtypes
            if args.mut_levels == 'Exon__Location__Protein':
                gene_mtypes |= {pnt_mtype}

                # ...as well as CNA-only subgroupings...
                gene_mtypes |= {
                    mtype for mtype in copy_mtypes
                    if (args.samp_cutoff <= len(mtype.get_samples(muts))
                        <= (n_samps - args.samp_cutoff))
                    }

                # ...and finally the CNA + all point mutations subgroupings
                gene_mtypes |= {
                    pnt_mtype | mtype for mtype in copy_mtypes
                    if (args.samp_cutoff
                        <= len((pnt_mtype | mtype).get_samples(muts))
                        <= (n_samps - args.samp_cutoff))
                    }

            use_mtypes |= {MuType({('Gene', gene): mtype})
                           for mtype in gene_mtypes}

    print(len({mtype for mtype in use_mtypes
               if (mtype & copy_mtype).is_empty()}))
    import pdb; pdb.set_trace()
    # set a random seed for use in picking random subgroupings
    lvls_seed = np.prod([(ord(char) % 7 + 3)
                         for i, char in enumerate(args.mut_levels)
                         if (i % 5) == 1])

    # makes sure random subgroupings are the same between different runs
    # of this experiment
    mtype_list = sorted(use_mtypes)
    random.seed((88701 * lvls_seed + 1313) % (2 ** 16))
    random.shuffle(mtype_list)

    # generate random subgroupings chosen from all samples in the cohort
    use_mtypes |= {
        RandomType(size_dist=len(mtype.get_samples(cdata.mtrees[lbls_key])),
                   seed=(lvls_seed * (i + 19) + 1307) % (2 ** 22))
        for i, (mtype, _) in enumerate(product(mtype_list, range(2)))
        if (mtype & copy_mtype).is_empty()
        }

    # generate random subgroupings chosen from samples mutated for each gene
    use_mtypes |= {
        RandomType(
            size_dist=len(mtype.get_samples(cdata.mtrees[lbls_key])),
            base_mtype=MuType({('Gene', mtype.get_labels()[0]): pnt_mtype}),
            seed=(lvls_seed * (i + 23) + 7391) % (2 ** 22)
            )
        for i, (mtype, _) in enumerate(product(mtype_list, range(2)))
        if ((mtype & copy_mtype).is_empty()
            and (len(mtype.get_samples(cdata.mtrees[lbls_key]))
                 < sum(cdata.train_pheno(
                     MuType({('Gene', mtype.get_labels()[0]): pnt_mtype})))))
        }

    # save enumerated subgroupings and number of subgroupings to file
    with open(os.path.join(out_path, "muts-list.p"), 'wb') as f:
        pickle.dump(sorted(use_mtypes), f, protocol=-1)
    with open(os.path.join(out_path, "muts-count.txt"), 'w') as fl:
        fl.write(str(len(use_mtypes)))

    if args.expr_source in {'Firehose', 'microarray'}:
        trnsf_src = 'Firehose'
    else:
        trnsf_src = args.expr_source.split('__')[0]

    # get list of available cohorts for this source of expression data
    coh_list = list_cohorts(trnsf_src, expr_dir=expr_dir, copy_dir=copy_dir)
    coh_list -= {args.cohort}
    coh_list |= {'METABRIC', 'CCLE'}

    # initiate set of genetic expression features, reset random seed
    use_feats = set(cdata.get_features())
    random.seed()

    for coh in random.sample(coh_list, k=len(coh_list)):
        coh_base = coh.split('_')[0]
        coh_tag = "cohort-data__{}.p".format(coh)
        coh_path = os.path.join(coh_dir, coh_tag)

        if coh_base in {'METABRIC', 'CCLE'}:
            use_src = 'microarray'
        else:
            use_src = str(trnsf_src)

        trnsf_cdata = load_cohort(coh, use_src, coh_path)
        if lbls_key not in trnsf_cdata.mtrees:
            trnsf_cdata.add_mut_lvls(lbls_key)

        use_feats &= set(trnsf_cdata.get_features())
        with open(coh_path, 'wb') as f:
            pickle.dump(trnsf_cdata, f, protocol=-1)

        copy_prc = subprocess.run(['cp', coh_path,
                                   os.path.join(out_path, coh_tag)],
                                  check=True)

    with open(os.path.join(out_path, "feat-list.p"), 'wb') as f:
        pickle.dump(use_feats, f, protocol=-1)


if __name__ == '__main__':
    main()

