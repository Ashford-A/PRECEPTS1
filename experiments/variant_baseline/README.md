HetMan: Variant Baseline Testing
--------------------------------

This experiment measures the performance and tuning characteristics of
classification algorithms when they are asked to predict frequently-occurring
mutations in a given cohort of tumour samples. We thus also are able to
observe:

- how different classifiers compare to one another in terms of their ability
  to successfully predict the presence of mutations
- the optimal tuning parameter grid for each classifier
- which mutations are associated with an identifiable expression signature in
  each cohort
- the effect that different sources and types (gene-level vs transcript-level)
  of expression data have on classification performance and behaviour


### Running the experiment ###

```bash
sbatch --mem-per-cpu=8000 --time=35:55:00 \
	--account='compbio' -c 1 --exclude=$ex_nodes \
	--output=$slurm_dir/var-base.out --error=$slurm_dir/var-base.err \
	HetMan/experiments/variant_baseline/run_tests.sh \
	-e Firehose -t BRCA_LumA -s 25 -c ridge__base -m 1000 -r
```

