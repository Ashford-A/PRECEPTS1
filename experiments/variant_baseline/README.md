HetMan: Variant Baseline Testing
--------------------------------


### Running the experiment ###

```bash
sbatch --output=$slurm_dir/var-base.out --error=$slurm_dir/var-base.err \
	HetMan/experiments/variant_baseline/run_tests.sh \
	-e Firehose -t BRCA_LumA -s 25 -c ridge__base -m 1000 -r
```

