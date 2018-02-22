
from .mut import BaseMutationCohort
from ..expression import get_expr_firehose, get_expr_bmeg, log_norm_expr
from ..variants import get_variants_mc3, get_variants_firehose

from ..annot import get_gencode
from ..utils import match_tcga_samples


class MutationCohort(BaseMutationCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        mut_genes (:obj:`list` of :obj:`str`)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`)
            What variant annotation levels to consider.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant data from.
        cv_seed (int): The random seed to use for cross-validation sampling.
        cv_prop (float): Proportion of samples to use for cross-validation.

    Attributes:
        path (dict): Pathway Commons neighbourhood for the mutation genes.
        train_mut (.variants.MuTree): Training cohort mutations.
        test_mut (.variants.MuTree): Testing cohort mutations.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = VariantCohort(
        >>>     cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )
        >>>
        >>> cdata2 = VariantCohort(
        >>>     cohort='TCGA-PAAD', mut_genes=['KRAS'],
        >>>     mut_levels=['Form_base', 'Exon', 'Location'],
        >>>     expr_source='Firehose', data_dir='../input-data/firehose',
        >>>     cv_seed=98, cv_prop=0.8, syn=syn
        >>>     )

    """

    def __init__(self,
                 cohort, mut_genes, mut_levels=('Gene', 'Form'),
                 expr_source='BMEG', var_source='mc3', top_genes=100,
                 samp_cutoff=None, cv_prop=2.0/3, cv_seed=None, **coh_args):

        # loads gene expression data from the given source, log-normalizes it
        # and removes missing values as necessary
        if expr_source == 'BMEG':
            expr_mat = get_expr_bmeg(cohort)
            expr = log_norm_expr(expr_mat.fillna(expr_mat.min().min()))

        elif expr_source == 'Firehose':
            expr_mat = get_expr_firehose(cohort, coh_args['data_dir'])
            expr = expr_mat.fillna(expr_mat.min().min())

        else:
            raise ValueError("Unrecognized source of expression data!")

        # loads gene variant data from the given source
        if var_source is None:
            var_source = expr_source

        if var_source == 'mc3':
            variants = get_variants_mc3(coh_args['syn'])

        elif var_source == 'Firehose':
            variants = get_variants_firehose(cohort, coh_args['data_dir'])

        else:
            raise ValueError("Unrecognized source of variant data!")

        self.cohort = cohort
        matched_samps = match_tcga_samples(expr.index, variants['Sample'])

        # gets the genes for which we have both expression and annotation data
        gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                      for ens, at in get_gencode().items()
                      if at['gene_name'] in expr.columns}

        super().__init__(expr, variants, matched_samps, gene_annot, mut_genes,
                         mut_levels, top_genes, samp_cutoff, cv_prop, cv_seed)
