"""
This module produces gene-specific plots of experiment output for all genes
with enumerated subgroupings in at least two of the canonical cohorts listed
in .__init__.py, and with a subgrouping with a classification task AUC of at
least 0.7 in any of these cohorts.

Example usages:
    python -m dryads-research.experiments.subgrouping_test.plot_genes Ridge
    python -m dryads-research.experiments.subgrouping_test.plot_genes SVCrbf

"""

from .plot_gene import *


def main():
    parser = argparse.ArgumentParser(
        'plot_genes',
        description="Creates gene-specific plots for all genes."
        )

    parser.add_argument('classif', help='a mutation classifier')
    parser.add_argument(
        '--seed', type=int,
        help="random seed for fixing plot elements like label placement"
        )

    # parse command line arguments, find experiments matching the given
    # criteria that have run to completion
    args = parser.parse_args()
    out_datas = [
        out_file.parts[-2:] for out_file in Path(base_dir).glob(os.path.join(
            "*__*__samps-*", "out-trnsf__*__{}.p.gz".format(args.classif)))
        ]

    # obtain experiments' subgrouping enumeration criteria, filter out cohorts
    # where the ``base'' mutation annotation level has not yet been tested
    out_list = pd.DataFrame([
        {'Source': '__'.join(out_data[0].split('__')[:-2]),
         'Cohort': out_data[0].split('__')[-2],
         'Samps': int(out_data[0].split("__samps-")[1]),
         'Levels': '__'.join(out_data[1].split(
             "out-trnsf__")[1].split('__')[:-1])}
        for out_data in out_datas
        ]).groupby('Cohort').filter(
            lambda outs: 'Consequence__Exon' in set(outs.Levels))

    if out_list.shape[0] == 0:
        raise ValueError("No experiment output found for these parameters!")

    # for each cohort, find the experiments that have been run with the
    # loosest sample count criteria; filter out non-canonical cohorts
    out_use = out_list.groupby(['Source', 'Cohort', 'Levels'])['Samps'].min()
    out_use = out_use[out_use.index.get_level_values(
        'Cohort').isin(train_cohorts)]
    out_use = out_use[[src != 'toil__gns' or coh == 'beatAML'
                       for src, coh, _, in out_use.index]]

    phn_dict = dict()
    out_aucs = dict()
    trnsf_aucs = dict()

    for (src, coh, lvls), ctf in out_use.iteritems():
        out_tag = "{}__{}__samps-{}".format(src, coh, ctf)

        with bz2.BZ2File(
                os.path.join(base_dir, out_tag,
                             "out-pheno__{}__{}.p.gz".format(
                                 lvls, args.classif)),
                'r') as f:
            phns = pickle.load(f)

        if (src, coh) in phn_dict:
            phn_dict[src, coh].update(phns)
        else:
            phn_dict[src, coh] = phns

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-aucs__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            auc_vals = pickle.load(f)

        auc_vals = auc_vals[[not isinstance(mtype, RandomType)
                             for mtype in auc_vals.index]]

        auc_vals.index = pd.MultiIndex.from_product(
            [[src], [coh], auc_vals.index],
            names=('Source', 'Cohort', 'Mtype')
            )
        out_aucs[src, coh, lvls] = auc_vals

        with bz2.BZ2File(os.path.join(base_dir, out_tag,
                                      "out-trnsf__{}__{}.p.gz".format(
                                          lvls, args.classif)),
                         'r') as f:
            trnsf_data = pickle.load(f)

        for trnsf_coh, trnsf_out in trnsf_data.items():
            if trnsf_out['AUC'].shape[0] > 0:
                auc_vals = trnsf_out['AUC']['mean']

                if (src, coh, trnsf_coh) in trnsf_aucs:
                    trnsf_aucs[src, coh, trnsf_coh] = pd.concat([
                        trnsf_aucs[src, coh, trnsf_coh], auc_vals])
                else:
                    trnsf_aucs[src, coh, trnsf_coh] = auc_vals

    auc_df = pd.concat(out_aucs.values())

    # group non-gene-wide point mutation subgroupings according to the cohort
    # they were enumerated in and the gene they are associated with
    gene_sets = auc_df['mean'][[get_subtype(mtype) != pnt_mtype
                                and pnt_mtype.is_supertype(get_subtype(mtype))
                                for _, _, mtype in auc_df.index]].groupby(
                                    lambda x: (x[0], x[1],
                                               get_label(x[2]))
                                    ).count()

    # find the genes with such subgroupings in at least two cohorts; produce
    # gene-specific plots for each such case
    gene_counts = pd.Series([x[-1] for x in gene_sets.index]).value_counts()
    for gene in gene_counts.index[gene_counts > 1]:
        setattr(args, 'gene', gene)

        auc_data = auc_df[[get_label(mtype) == gene
                           for _, _, mtype in auc_df.index]]

        if (auc_data['mean'] >= 0.7).any():
            os.makedirs(os.path.join(plot_dir, gene), exist_ok=True)

            auc_dict = {
                (src, coh): aucs.reset_index(['Source', 'Cohort'], drop=True)
                for (src, coh), aucs in auc_data.groupby(['Source', 'Cohort'])
                }

            plot_sub_comparisons(auc_dict, phn_dict, args.classif,
                                 args, include_copy=False)
            plot_sub_comparisons(auc_dict, phn_dict, args.classif,
                                 args, include_copy=True)

            trnsf_dict = {
                (src, coh, trnsf_coh): trnsf_df[[
                    not isinstance(mtype, RandomType)
                    and get_label(mtype) == gene for mtype in trnsf_df.index
                    ]]
                for (src, coh, trnsf_coh), trnsf_df in trnsf_aucs.items()
                }

            if any(len({mtype for mtype in trnsf_vals.index
                        if (get_subtype(mtype) != pnt_mtype
                            and (get_subtype(mtype)
                                 & copy_mtype).is_empty())}) > 0
                   for trnsf_vals in trnsf_dict.values()):

                plot_transfer_aucs(trnsf_dict, auc_dict, phn_dict,
                                   args.classif, args)
                plot_transfer_comparisons(trnsf_dict, auc_dict, phn_dict,
                                          args.classif, args)


if __name__ == '__main__':
    main()

