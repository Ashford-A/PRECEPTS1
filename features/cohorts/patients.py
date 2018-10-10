
from ...features.cohorts.tcga import MutationCohort
import numpy as np
import pandas as pd


class PatientMutationCohort(MutationCohort):

    def __init__(self,
                 patient_expr, patient_muts, tcga_cohort,
                 mut_genes, mut_levels, expr_source, var_source, copy_source,
                 annot_file, domain_dir=None, top_genes=100, samp_cutoff=None,
                 cv_prop=2./3, cv_seed=None, **coh_args):

        super().__init__(tcga_cohort, mut_genes, mut_levels, expr_source,
                         var_source, copy_source, annot_file, domain_dir,
                         top_genes, samp_cutoff, cv_prop, cv_seed, **coh_args)

        if np.all(patient_expr.columns.str.match('^ENST')):
            use_txs = set(self.omic_data.columns.get_level_values('ENST'))
            use_txs &= set(patient_expr.columns)

            use_omics = self.omic_data.loc[
                :, self.omic_data.columns.get_level_values('ENST').isin(
                    use_txs)
                ]

            patient_expr = patient_expr.loc[
                :, self.omic_data.columns.get_level_values('ENST')]
            patient_expr.columns = use_omics.columns

            self.omic_data = pd.concat([use_omics, patient_expr], sort=False)
            self.genes = frozenset(
                self.omic_data.columns.get_level_values('Gene'))

        else:
            if np.all(patient_expr.columns.str.match('^ENSG')):
                patient_expr.columns = patient_expr.columns.str.replace(
                    '\.[0-9]+$', '')
 
                gene_ids = {ant['gene_id'].split('.')[0]: gn
                            for gn, ant in self.gene_annot.items()}
 
                patient_expr = patient_expr.loc[
                    :, patient_expr.columns.isin(gene_ids)]
                patient_expr.columns = [gene_ids[gn]
                                        for gn in patient_expr.columns]

            elif np.mean([gn in patient_expr for gn in self.gene_annot]) > 0.5:
                patient_expr = patient_expr.loc[
                    :, patient_expr.columns.isin(self.gene_annot)]

            else:
                raise ValueError("Unrecognized format of patient expression "
                                 "genomic features!")
 
            self.genes = frozenset(self.genes & set(patient_expr.columns))
            self.omic_data = pd.concat(
                [self.omic_data.loc[:, self.genes], patient_expr], sort=False)

        self.samples = frozenset(self.samples | set(patient_expr.index))
        self.patient_samps = frozenset(patient_expr.index)

