HetMan: Transferring Subvariant Classifiers Between Cohorts
--------------------------------

### Running the experiment ###

```bash
sbatch --mem-per-cpu=8000 --time=0:55:00 \
	--account='compbio' -c 1 --exclude=$ex_nodes \
	--output=$slurm_dir/subv-trans.out --error=$slurm_dir/subv-trans.err \
	HetMan/experiments/subvariant_transfer/run_transfer.sh \
	-t BRCA_LumA -t STAD -t SKCM -t BLCA -t LUSC \
	-s 20 -l Location__Protein -c Ridge -x Shallow -m 350
```

