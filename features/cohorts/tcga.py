
from ..data.expression import get_expr_firehose, get_expr_bmeg, get_expr_toil
from ..data.copies import get_copies_firehose
from .mut_freq import BaseMutFreqCohort

from dryadic.features.cohorts.mut import (
    BaseMutationCohort, BaseCopyCohort, BaseTransferMutationCohort)
from dryadic.features.cohorts.utils import (
    match_tcga_samples, get_gencode, log_norm, drop_duplicate_genes)

import numpy as np
import pandas as pd

import os
from functools import reduce
from re import sub as gsub
import synapseclient
from operator import and_
from itertools import cycle, combinations

from pathlib import Path
from sklearn.preprocessing import scale


cohort_subtypes = {
    'nonbasal': ['LumA', 'LumB', 'Her2', 'Normal'],
    'luminal': ['LumA', 'LumB'],
    }

tcga_subtypes = {
    'BRCA': ('LumA', 'luminal', 'nonbasal'),
    'HNSC': ('HPV-', 'HPV+'),
    'CESC': ('SquamousCarcinoma', ),
    'LGG': ('IDHmut-non-codel', ),
    }


def parse_subtypes(cohort):
    use_subtypes = None
    coh_info = cohort.split('_')

    if len(coh_info) > 1:
        if coh_info[1] in cohort_subtypes:
            use_subtypes = cohort_subtypes[coh_info[1]]
        else:
            use_subtypes = [coh_info[1]]

    return use_subtypes


def choose_subtypes(use_types, base_coh, type_file):
    type_data = pd.read_csv(type_file, sep='\t', index_col=0, comment='#')
    type_data = type_data[type_data.DISEASE == base_coh]

    return set(type_data.index[type_data.SUBTYPE.isin(use_types)])


def get_expr_data(cohort, expr_source, **expr_args):
    """Loads RNA-seq expression data for a given TCGA cohort and data source.

    Args:
        cohort (str): A cohort in TCGA with expression data.
        expr_source (str): A repository which contains TCGA datasets.

    """
    if expr_source == 'BMEG':
        expr_mat = get_expr_bmeg(cohort)

    elif expr_source == 'Firehose':
        expr_mat = get_expr_firehose(cohort, expr_args['expr_dir'])

    elif expr_source == 'toil':
        expr_mat = get_expr_toil(cohort, expr_args['expr_dir'],
                                 expr_args['collapse_txs'])

    else:
        raise ValueError("Unrecognized source of expression data!")

    return log_norm(expr_mat.fillna(0.0))


def get_variant_data(cohort, var_source, **var_args):
    if var_source == 'mc3':
        mc3 = var_args['syn'].get('syn7824274')

        field_dict = (
            ('Gene', 0), ('Chr', 4), ('Start', 5), ('End', 6), ('Strand', 7),
            ('Form', 8), ('RefAllele', 10), ('TumorAllele', 12),
            ('Sample', 15), ('HGVS', 34), ('Protein', 36), ('Transcript', 37),
            ('Exon', 38), ('depth', 39), ('ref_count', 40), ('alt_count', 41),
            ('SIFT', 71), ('PolyPhen', 72), ('Filter', 108)
            )

        if 'mut_fields' not in var_args or var_args['mut_fields'] is None:
            use_fields, use_cols = tuple(zip(*field_dict))

        else:
            use_fields, use_cols = tuple(zip(*[
                (name, col) for name, col in field_dict
                if name in {'Sample', 'Filter'} | set(var_args['mut_fields'])
                ]))

        # imports mutation data into a DataFrame, parses TCGA sample barcodes
        # and PolyPhen scores
        i = 0
        while i < 10:
            #TODO: handle I/O errors on the cohort/experiment level?
            try:
                var_data = pd.read_csv(mc3.path, engine='c', dtype='object',
                                       sep='\t', header=None,
                                       usecols=use_cols, names=use_fields,
                                       comment='#', skiprows=1)
                break

            except OSError:
                i = i + 1

        #TODO: more fine-grained Filtering control?
        var_data = var_data.loc[~var_data.Filter.str.contains(
            'nonpreferredpair')]

        for annt, null_val in zip(['PolyPhen', 'SIFT'], [0, 1]):
            if annt in var_data:
                var_data[annt] = var_data[annt].apply(
                    lambda val: (np.float(gsub('\)$', '',
                                               gsub('^.*\(', '', val)))
                                 if val != '.' else null_val)
                    )

                if annt == 'SIFT':
                    var_data[annt] = 1 - var_data[annt]

        var_data.Sample = var_data.Sample.apply(
            lambda smp: '-'.join(smp.split('-')[:4]))
    
    elif var_source == 'Firehose':
        mut_tar = tarfile.open(glob.glob(os.path.join(
            data_dir, "stddata__2016_01_28", cohort, "20160128",
            "*Mutation_Packager_Oncotated_Calls.Level_3*tar.gz"
            ))[0])

        mut_list = []
        for mut_fl in mut_tar.getmembers():

            try:
                mut_tbl = pd.read_csv(
                    BytesIO(mut_tar.extractfile(mut_fl).read()),
                    sep='\t', skiprows=4, usecols=[0, 8, 15, 37, 41],
                    names=['Gene', 'Form', 'Sample', 'Exon', 'Protein']
                    )
                mut_list += [mut_tbl]

            except:
                print("Skipping mutations for {}".format(mut_fl))
            
        muts = pd.concat(mut_list)
        muts.Sample = muts.Sample.apply(
            lambda smp: "-".join(smp.split("-")[:4]))
        mut_tar.close()

    elif var_source == 'BMEG':
        oph = Ophion("http://bmeg.io")
        mut_list = {samp: {} for samp in sample_list}
        gene_lbls = ["gene:" + gn for gn in gene_list]

        print(oph.query().has("gid", "biosample:" + sample_list[0])
              .incoming("variantInBiosample")
              .outEdge("variantInGene").mark("variant")
              .inVertex().has("gid", oph.within(gene_lbls)).count().execute())
              # .mark("gene").select(["gene", "variant"]).count().execute())

        for samp in sample_list:
            for i in oph.query().has("gid", "biosample:" + samp)\
                    .incoming("variantInBiosample")\
                    .outEdge("variantInGene").mark("variant")\
                    .inVertex().has("gid", oph.within(gene_lbls))\
                    .mark("gene").select(["gene", "variant"]).execute():
                dt = json.loads(i)
                gene_name = dt["gene"]["properties"]["symbol"]
                mut_list[samp][gene_name] = {
                    k: v for k, v in dt["variant"]["properties"].items()
                    if k in mut_fields}

        mut_table = pd.DataFrame(mut_list)

    else:
        raise ValueError("Unrecognized source of variant data!")

    return var_data


def add_mutations(cohort, var_source, copy_source, expr, gene_annot,
                  **mut_args):
    """Adds variant calls to loaded TCGA expression data.

    Args:
        cohort (str): A cohort in TCGA with variant call data.
        var_source (str): A repository which contains TCGA datasets.
        expr (:obj:`pd.DataFrame`, shape = [n_samps, n_genes])
        gene_annot (dict): Annotation for each of the expressed genes.
        add_cna (bool, optional): Whether to add discrete copy number calls.
                                  Default is to only use point mutations.

    """
    var_data = get_variant_data(cohort, var_source, **mut_args)

    # load copy number alteration data from the given source
    if copy_source == 'Firehose':
        if 'copy_dir' not in mut_args:
            copy_dir = mut_args['expr_dir']
        else:
            copy_dir = mut_args['copy_dir']

        copy_data = get_copies_firehose(cohort, copy_dir, discrete=True)

    else:
        raise ValueError("Unrecognized source of copy number data!")

    copy_df = pd.DataFrame(copy_data.stack()).reset_index()
    copy_df.columns = ['Sample', 'Gene', 'Copy']

    expr_match, var_match, copy_match = match_tcga_samples(
        expr.index, var_data.Sample.values, copy_df.Sample.values)

    new_expr = expr.loc[
        expr.index.isin(expr_match),
        expr.columns.get_level_values('Gene').isin(gene_annot)
        ]
    new_expr.index = [expr_match[old_samp] for old_samp in new_expr.index]

    new_vars = var_data[var_data.isin({
        'Sample': var_match.keys(), 'Gene': gene_annot.keys()}).loc[
            :, ['Gene', 'Sample']].all(axis=1)].copy()
    new_vars.Sample = new_vars.Sample.apply(lambda samp: var_match[samp])
 
    new_copy = copy_df.loc[(copy_df.Copy != 0)
                           & copy_df.Sample.isin(copy_match)
                           & copy_df.Gene.isin(gene_annot)
                           , :].copy()

    new_copy.Sample = new_copy.Sample.apply(lambda samp: copy_match[samp])
    new_copy.Copy = new_copy.Copy.map({-2: 'DeepDel', -1: 'ShalDel',
                                       1: 'ShalGain', 2: 'DeepGain'})

    return new_expr, new_vars, new_copy


def list_cohorts(data_source, **data_args):
    """Finds all the TCGA cohorts available in a given data repository.

    Args:
        data_source (str): A repository which contains TCGA datasets.

    """

    if data_source == 'BMEG':
        pass

    elif data_source == 'Firehose':
        cohorts = {pth.parent.parent.name
                   for pth in Path(data_args['expr_dir']).glob(
                       "stddata__*/**/*__RSEM_genes_normalized__*")}

    elif data_source == 'toil':
        cohorts = {pth.name.split('TCGA_')[1].split('_tpm')[0]
                   for pth in Path(data_args['expr_dir']).glob(
                       "TCGA/TCGA_*_tpm.tsv.gz")}

    else:
        raise ValueError("Unrecognized source of expression data!")

    cohorts &= {pth.parent.parent.name
                for pth in Path(data_args['copy_dir']).glob(
                    "analyses__*/**/*CopyNumber_Gistic2.Level_4*")}

    return cohorts


def process_input_datasets(cohort, expr_source, var_source, copy_source,
                           annot_dir, type_file, **data_args):

    base_coh = cohort.split('_')[0]
    use_types = parse_subtypes(cohort)

    expr = drop_duplicate_genes(get_expr_data(base_coh, expr_source,
                                              **data_args))

    annot_file = os.path.join(annot_dir, "gencode.v19.annotation.gtf.gz")
    if 'annot_fields' in data_args:
        annot_data = get_gencode(annot_file, data_args['annot_fields'])
    else:
        annot_data = get_gencode(annot_file)

    # restructure annotation data around expression gene labels
    use_genes = set(expr.columns.get_level_values('Gene'))
    annot_dict = {at['gene_name']: {**{'Ens': ens}, **at}
                  for ens, at in annot_data.items()
                  if at['gene_name'] in use_genes}

    expr, variants, copy_df = add_mutations(base_coh, var_source, copy_source,
                                            expr, annot_dict, **data_args)

    use_samps = set(expr.index)
    if use_types is not None:
        use_samps &= choose_subtypes(use_types, base_coh, type_file)

    expr_data = expr.loc[use_samps]
    variants = variants.loc[variants.Sample.isin(use_samps)]
    copy_df = copy_df.loc[copy_df.Sample.isin(use_samps)]

    return expr_data, variants, copy_df, annot_dict


class MutationCohort(BaseMutationCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        mut_genes (:obj:`list` of :obj:`str`)
            Which genes' variants to include.
        mut_levels (:obj:`list` of :obj:`str`)
            What variant annotation levels to consider.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant call data from.
        copy_source (:obj:`str`, optional)
            Where to load continuous copy number alteration data from. The
            default is to not use continuous CNA scores.
        top_genes (:obj:`int`, optional)
            How many of the genes in the cohort ordered by descending
            mutation frequency to load mutation data for.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = MutationCohort(
        >>>     cohort='BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )
        >>>
        >>> cdata2 = MutationCohort(
        >>>     cohort='PAAD', mut_genes=['KRAS'],
        >>>     mut_levels=['Form_base', 'Exon', 'Location'],
        >>>     expr_source='Firehose', expr_dir='../input-data/firehose',
        >>>     cv_seed=98, cv_prop=0.8, syn=syn
        >>>     )
        >>>
        >>> cdata3 = MutationCohort(
        >>>     cohort='LUSC', mut_genes=None, mut_levels=['Gene', 'Form'],
        >>>     expr_source='Firehose', var_source='mc3',
        >>>     copy_source='Firehose', expr_dir='../input-data/firehose',
        >>>     copy_dir='../input-data/firehose', samp_cutoff = 0.2,
        >>>     syn=syn, cv_prop=0.75, cv_seed=5601
        >>>     )
        >>>
    """

    def __init__(self,
                 cohort, mut_levels=None, mut_genes=None, expr_source=None,
                 var_source=None, copy_source=None, annot_dir=None,
                 type_file=None, leaf_annot=('ref_count', 'alt_count'),
                 cv_seed=None, test_prop=0, **coh_args):
        self.cohort = cohort

        expr_data, mut_data, gene_annot = process_input_datasets(
            cohort, expr_source, var_source, copy_source,
            annot_dir, type_file=None, **coh_args
            )

        super().__init__(expr_data, mut_data, mut_levels, mut_genes,
                         gene_annot, leaf_annot, cv_seed, test_prop)


class CopyCohort(BaseCopyCohort):

    def __init__(self,
                 cohort, copy_genes, expr_source, copy_source, annot_file,
                 type_file, annot_fields=None, use_types=None, cv_seed=None,
                 test_prop=0, **coh_args):
        self.cohort = cohort

        # load expression and gene annotation datasets
        expr = drop_duplicate_genes(get_expr_data(cohort, expr_source,
                                                  **coh_args))
        annot_data = get_gencode(annot_file, annot_fields)

        # restructure annotation data around expression gene labels
        self.gene_annot = {
            at['gene_name']: {**{'Ens': ens}, **at}
            for ens, at in annot_data.items()
            if at['gene_name'] in set(expr.columns.get_level_values('Gene'))
            }

        if copy_source == 'Firehose':
            if 'copy_dir' not in coh_args:
                copy_dir = coh_args['expr_dir']
            else:
                copy_dir = coh_args['copy_dir']

            copies = get_copies_firehose(cohort, copy_dir, discrete=False)

        else:
            raise ValueError("Unrecognized source of copy number data!")

        expr_match, copy_match = match_tcga_samples(expr.index, copies.index)

        expr_df = expr.loc[
            expr.index.isin(expr_match),
            expr.columns.get_level_values('Gene').isin(self.gene_annot)
            ]
        expr_df.index = [expr_match[old_samp] for old_samp in expr_df.index]

        self.event_feats = copies.columns[
            copies.columns.str.match("[0-9]+(p|q).?")]
        copy_df = copies.loc[copies.index.isin(copy_match),
                             (copies.columns.isin(self.gene_annot)
                              | copies.columns.isin(self.event_feats))]

        self.gene_annot.update(zip(
            self.event_feats,
            [{'Chr': "chr{}".format(lbl)}
             for lbl in self.event_feats.str.replace("(p|q).*", "")]
            ))

        copy_df.index = [copy_match[samp] for samp in copy_df.index]
        use_samps = set(expr_df.index)

        if use_types is not None:
            type_data = pd.read_csv(type_file,
                                    sep='\t', index_col=0, comment='#')
            type_data = type_data[type_data.DISEASE == cohort] 

            use_samps &= set(type_data.index[
                type_data.SUBTYPE.isin(use_types)])

        expr_df = expr_df.loc[use_samps]
        copy_df = copy_df.loc[copy_df.index.isin(use_samps),
                              (set(copy_genes) & set(copy_df.columns))
                              | set(self.event_feats)]
        copy_genes = [copy_gene for copy_gene in copy_genes
                      if copy_gene in copy_df.columns]

        super().__init__(expr_df, copy_df, copy_genes, cv_seed, test_prop)


class MutFreqCohort(BaseMutFreqCohort):
    """An expression dataset used to predict genes' mutations (variants).

    Args:
        cohort (str): The label for a cohort of samples.
        expr_source (str): Where to load the expression data from.
        var_source (str): Where to load the variant data from.
        copy_source (:obj:`str`, optional)
            Where to load the copy number alteration data from. The default
            is to not use any CNA data.
        cv_prop (float): Proportion of samples to use for cross-validation.
        cv_seed (int): The random seed to use for cross-validation sampling.

    Examples:
        >>> import synapseclient
        >>> syn = synapseclient.Synapse()
        >>> syn.login()
        >>>
        >>> cdata = MutationCohort(
        >>>     cohort='TCGA-BRCA', mut_genes=['TP53', 'PIK3CA'],
        >>>     mut_levels=['Gene', 'Form', 'Exon'], syn=syn
        >>>     )

    """

    def __init__(self,
                 cohort, expr_source, var_source, copy_source, annot_file,
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        self.cohort = cohort

        expr = get_expr_data(cohort, expr_source, **coh_args)
        if 'annot_fields' in coh_args:
            annot_data = get_gencode(annot_file, coh_args['annot_fields'])
        else:
            annot_data = get_gencode(annot_file)

        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        expr, variants = add_mutations(cohort, var_source, copy_source,
                                       expr, self.gene_annot, **coh_args)

        super().__init__(expr, variants, cv_prop, cv_seed)


class PanCancerMutCohort(BaseMutationCohort):

    def __init__(self, 
                 mut_genes, mut_levels, expr_source, var_source, copy_source,
                 annot_file, top_genes=100, samp_cutoff=None,
                 cv_prop=2.0/3, cv_seed=None, **coh_args):
        expr_cohorts = list_cohorts(expr_source, **expr_args)

        expr_dict = dict()
        for cohort in expr_cohorts:
            try:
                expr_data = get_expr_data(cohort, expr_source, **coh_args)

                expr_dict[cohort] = pd.DataFrame(scale(expr_data),
                                                 index=expr_data.index,
                                                 columns=expr_data.columns)

            except:
                print('no expression found for {}'.format(cohort))

        # removes samples that appear in more than one cohort
        for coh1, coh2 in combinations(expr_dict, 2):
            if len(expr_dict[coh1].index & expr_dict[coh2].index):
                ovlp1 = expr_dict[coh1].index.isin(expr_dict[coh2].index)
                ovlp2 = expr_dict[coh2].index.isin(expr_dict[coh1].index)

                if np.all(ovlp1):
                    expr_dict[coh2] = expr_dict[coh2].loc[~ovlp2, :]
                elif np.all(ovlp2) or np.sum(~ovlp1) >= np.sum(~ovlp2):
                    expr_dict[coh1] = expr_dict[coh1].loc[~ovlp1, :]
                else:
                    expr_dict[coh2] = expr_dict[coh2].loc[~ovlp2, :]

        expr = pd.concat(list(expr_dict.values()))
        self.gene_annot = {at['gene_name']: {**{'Ens': ens}, **at}
                           for ens, at in get_gencode().items()
                           if at['gene_name'] in expr.columns}

        if var_source == 'mc3':
            variants, matched_samps = add_mutations(
                cohort=None, var_source='mc3', expr=expr,
                gene_annot=self.gene_annot, **coh_args
                )

        else:
            if var_source is None:
                var_source = expr_source
                var_cohorts = expr_cohorts

            else:
                var_args = dict()
                if 'var_dir' in coh_args:
                    var_args['data_dir'] = coh_args['var_dir']
                if 'syn' in coh_args:
                    var_args['syn'] = coh_args['syn']

                var_cohorts = list_cohorts(var_source, **var_args)
                var_cohorts &= expr_dict

            var_dict = dict()
            for cohort in var_cohorts:
                try:
                    var_dict[cohort] = add_mutations(cohort, expr_source,
                                                        **coh_args)

                    if copy_source is not None:
                        var_dict[cohort] = pd.concat([
                            var_dict[cohort],
                            get_copy_data(cohort, copy_source, **coh_args)
                            ])

                except:
                    print('no variants found for {}'.format(cohort))
            
            variants = pd.concat(list(var_dict.values()))

        expr = expr.loc[expr.index.isin(matched_samps[0]),
                        expr.columns.isin(list(self.gene_annot))]
        expr.index = [matched_samps[0][old_samp] for old_samp in expr.index]

        variants = variants.loc[variants['Sample'].isin(matched_samps[1]), :]
        variants['Sample'] = [matched_samps[1][old_samp]
                              for old_samp in variants['Sample']]

        # for each expression cohort, find the samples that were matched to
        # a sample in the mutation call data
        cohort_samps = {
            cohort: set(matched_samps[0][samp]
                        for samp in expr_df.index & matched_samps[0])
            for cohort, expr_df in expr_dict.items()
            }

        # save a list of matched samples for each cohort as an attribute
        self.cohort_samps = {cohort: samps
                             for cohort, samps in cohort_samps.items()
                             if samps}
        copy_data = None
        super().__init__(expr, variants, copy_data, mut_genes, mut_levels,
                         top_genes, samp_cutoff, cv_prop, cv_seed)


class TransferMutationCohort(BaseTransferMutationCohort):

    def __init__(self,
                 cohorts, mut_levels, mut_genes, expr_sources, var_sources,
                 copy_sources, annot_file, domain_dir=None, type_file=None,
                 cv_seed=None, test_prop=0, **coh_args):
        self.cohorts = cohorts

        if isinstance(expr_sources, str):
            expr_sources = [expr_sources]

        if var_sources is None:
            var_sources = expr_sources

        elif 'mc3' in var_sources:
            coh_args = {**coh_args,
                        **{'mc3': get_variants_mc3(coh_args['syn'])}}

        elif isinstance(var_sources, str):
            var_sources = [var_sources]

        if isinstance(copy_sources, str):
            copy_sources = [copy_sources]

        # load expression data for each cohort, get gene annotation
        expr_raw = {coh: drop_duplicate_genes(get_expr_data(coh, expr_src,
                                                            **coh_args))
                    for coh, expr_src in zip(cohorts, cycle(expr_sources))}

        if 'annot_fields' in coh_args:
            annot_data = get_gencode(annot_file, coh_args['annot_fields'])
        else:
            annot_data = get_gencode(annot_file)

        self.gene_annot = {
            at['gene_name']: {**{'Ens': ens}, **at}
            for ens, at in annot_data.items()
            if at['gene_name'] in reduce(
                and_, [expr.columns.get_level_values('Gene')
                       for expr in expr_raw.values()])
            }

        expr_dict = {cohort: None for cohort in cohorts}
        var_dict = {cohort: None for cohort in cohorts}
        for cohort, var_source, copy_source in zip(
                cohorts, cycle(var_sources), cycle(copy_sources)):

            expr_dict[cohort], var_dict[cohort], copy_df = add_mutations(
                cohort, var_source, copy_source, expr_raw[cohort],
                self.gene_annot, **coh_args
                )
 
            var_dict[cohort].loc[:, 'Scale'] = 'Point'
            copy_df.loc[:, 'Scale'] = 'Copy'
            var_dict[cohort] = pd.concat([var_dict[cohort], copy_df],
                                         sort=True)
 
        if 'Gene' in mut_levels:
            scale_lvl = mut_levels.index('Gene') + 1
        else:
            scale_lvl = 0
 
        mut_levels.insert(scale_lvl, 'Scale')
        mut_levels.insert(scale_lvl + 1, 'Copy')

        super().__init__(expr_dict, var_dict,
                         mut_genes, mut_levels, top_genes, samp_cutoff,
                         cv_prop, cv_seed)

