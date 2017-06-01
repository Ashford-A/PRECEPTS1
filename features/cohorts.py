
"""
HetMan (Heterogeneity Manifold)
Classification of mutation sub-types using expression data.
This file contains classes that consolidate -omics datasets for use in
testing classifiers.
"""

# Author: Michal Grzadkowski <grzadkow@ohsu.edu>

from .expression import get_expr_bmeg, parse_tcga_barcodes
from .variants import get_variants_mc3, MuTree
from .pathways import parse_sif
from .annot import get_gencode

import numpy as np
from scipy.stats import fisher_exact
import random


class Cohort(object):

    def _validate_dims(self,
                       mtype=None, include_samps=None, exclude_samps=None,
                       gene_list=None, use_test=False):
        if include_samps is not None and exclude_samps is not None:
            raise ValueError("Cannot specify samples to be included and"
                             "samples to be excluded at the same time!")

        # get samples and genes from the specified cohort as specified
        if use_test:
            samps = self.test_samps_
            genes = set(self.test_expr_.columns)
        else:
            samps = self.train_samps_
            genes = set(self.train_expr_.columns)

        # remove samples and/or genes as necessary
        if include_samps is not None:
            samps &= set(include_samps)
        elif exclude_samps is not None:
            samps -= set(exclude_samps)
        if gene_list is not None:
            genes &= set(gene_list)

        # if a mutation type is specified include samples with that mutation
        if mtype is not None:
            samps |= mtype.get_samples(self.train_mut_)

        return samps, genes


class VariantCohort(Cohort):

    def __init__(self,
                 syn, cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 cv_info=None):
        self.cohort_ = cohort
        if cv_info is None:
            cv_info = {'Prop': 2.0/3, 'Seed':1}
        self.intern_cv_ = cv_info['Seed'] ** 2
        self.mut_genes = mut_genes

        # loads gene expression and mutation data, as well as pathway
        # neighbourhood for mutated genes
        expr = get_expr_bmeg(cohort)
        expr.index = [x[-1] for x in expr.index.str.split(':')]
        variants = get_variants_mc3(syn)
        self.path_ = parse_sif(mut_genes)
        annot = get_gencode()

        # filters out genes that don't have any variation across the samples
        # or are not included in the annotation data
        expr = expr.loc[:, expr.apply(lambda x: np.var(x) > 0.005)].dropna()
        annot = {g:a for g,a in annot.items()
                 if a['gene_name'] in expr.columns}
        annot_genes = [a['gene_name'] for g,a in annot.items()]
        expr = expr.loc[:, annot_genes]
        expr.index = parse_tcga_barcodes(expr.index)
        expr = expr.loc[:, ~expr.columns.duplicated()]
        expr = expr.loc[~expr.index.duplicated(), :]

        # gets set of samples shared across expression and mutation datasets,
        # subsets these datasets to use only these samples
        self.samples = set(variants['Sample']) & set(expr.index)
        expr = expr.loc[self.samples, :]
        variants = variants.loc[variants['Gene'].isin(mut_genes), :]

        # gets annotation data for the genes whose mutations
        # are under consideration
        annot_data = {mut_g: {'ID': g, 'Chr': a['chr'],
                              'Start': a['Start'], 'End': a['End']}
                      for g, a in annot.items() for mut_g in mut_genes
                      if a['gene_name'] == mut_g}
        self.annot = annot
        self.mut_annot = annot_data

        # gets subset of samples to use for training
        random.seed(a=cv_info['Seed'])
        self.cv_seed = random.getstate()
        if cv_info['Prop'] < 1.0 and cv_info['Prop'] > 0.0:
            self.train_samps_ = frozenset(
                random.sample(
                    population=self.samples,
                    k=int(round(len(self.samples) * cv_info['Prop'])))
                )
            self.test_samps_ = self.samples - self.train_samps_
            self.test_mut_ = MuTree(
                muts=variants.loc[variants['Sample'].isin(self.test_samps_), :],
                levels=mut_levels)
            self.test_expr_ = expr.loc[self.test_samps_]

        elif cv_info['Prop'] == 1:
            self.train_samps_ = self.samples
            self.test_samps_ = None

        else:
            raise ValueError("Improper cross-validation ratio that is"
                             "not > 0 and <= 1.0")

        # creates training and testing expression and mutation datasets
        self.train_expr_ = expr.loc[self.train_samps_, :]
        self.train_mut_ = MuTree(
            muts=variants.loc[variants['Sample'].isin(self.train_samps_), :],
            levels=mut_levels)

    def mutex_test(self, mtype1, mtype2):
        """Checks the mutual exclusivity of two mutation types in the
           training data using a one-sided Fisher's exact test.

        Parameters
        ----------
        mtype1,mtype2 : MuTypes
            The mutation types to be compared.

        Returns
        -------
        pval : float
            The p-value given by the test.
        """
        samps1 = mtype1.get_samples(self.train_mut_)
        samps2 = mtype2.get_samples(self.train_mut_)

        if not samps1 or not samps2:
            raise ValueError("Both sets must be non-empty!")

        all_samps = set(self.train_expr_.index)
        both_samps = samps1 & samps2
        _, pval = fisher_exact(
            [[len(all_samps - (samps1 | samps2)),
              len(samps1 - both_samps)],
             [len(samps2 - both_samps),
              len(both_samps)]],
            alternative='less')

        return pval


class DrugCohort(Cohort):

    def __init__(self, drug, source='CCLE', random_state=None):
        exp_data = {}
        drug_data = {}

        if source == 'CCLE':
            for i in oph.query().has(
                "gid", "cohort:CCLE").outgoing(
                    "hasSample").incoming("expressionForSample").execute():

                if ('properties' in i
                    and 'serializedExpressions' in i['properties']):
                    drug_querr = oph.query().has("gid", i['gid']).outgoing(
                        "expressionForSample").outEdge(
                            "responseToCompound").values(
                                ['gid', 'responseSummary']).execute()
                    drug_data[i['gid']] = {}

                    for dg in drug_querr:
                        if dg:
                            dg_parse = dg.split(':')
                            if dg_parse[0] == 'responseCurve':
                                cur_drug = dg_parse[-1]

                            elif cur_drug == drug:
                                drug_data[i['gid']] = [
                                    x['value'] for x in json.loads(dg)
                                    if x['type'] == 'AUC'][0]

                s = json.loads(i['properties']['serializedExpressions'])
                exp_data[i['gid']] = s

            drug_resp = pd.Series({k:v for k,v in drug_data.items() if v})
            drug_expr = exp_norm(pd.DataFrame(exp_data).transpose())
            drug_expr = drug_expr.loc[drug_resp.index, :]

        elif source == 'ioria':
            cell_expr = pd.read_csv(
                '/home/users/grzadkow/compbio/input-data/ioria-landscape/'
                'cell-line/Cell_line_RMA_proc_basalExp.txt',
                sep='\t', comment='#')
            cell_expr = cell_expr.ix[~pd.isnull(cell_expr['GENE_SYMBOLS']), :]
            drug_annot = pd.read_csv('/home/users/grzadkow/compbio/'
                                     'scripts/HetMan/experiments/'
                                     'drug_predictions/input/drug_annot.txt',
                                     sep='\t', comment='#')

            cell_expr.index = cell_expr['GENE_SYMBOLS']
            cell_expr = cell_expr.ix[:, 2:]

            drug_resp = pd.read_csv('/home/users/grzadkow/compbio/input-data'
                                    '/ioria-landscape/cell-line/drug-auc.txt',
                                    sep='\t', comment='#')
            drug_match = process.extractOne(drug, drug_annot['Name'])
            drug_lbl = 'X' + str(drug_annot['Identifier'][drug_match[-1]])
            drug_resp = drug_resp.loc[:, drug_lbl]
            drug_resp = drug_resp[~pd.isnull(drug_resp)]
            drug_expr = cell_expr.loc[:, drug_resp.index].transpose().dropna(
                axis=0, how='all').dropna(axis=1, how='any')
            drug_resp = drug_resp.loc[drug_expr.index]

        random.seed(a=random_state)
        cv_seed = random.getstate()
        self.train_samps_ = frozenset(
            random.sample(population=list(drug_expr.index),
                          k=int(round(drug_expr.shape[0] * 0.8)))
            )
        self.test_samps_ = frozenset(
            set(drug_expr.index) - self.train_samps_)

        self.random_state = random_state
        self.drug_resp = drug_resp
        self.drug_expr = drug_expr

