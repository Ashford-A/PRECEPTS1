# dryads-research #

The purpose of this project is to investigate whether we can use
classification algorithms to not only predict the presence of different
mutations recurrent in tumour cohorts, but also make biologically useful
inferences about these mutations' downstream effects.

See `experiments/subgrouping_test` for the bulk of the code used in
[our publication](bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04147-y)
based on this project, as well as a README explaining how to set up and run
one of the experiment pipelines.


This repository consists of three major parts:


## features ##

Collecting and processing expression, mutation, and other -omics datasets as
well as phenotypes such as drug response.


## predict ##

Custom machine learning tools to predict presence of gene mutations, drug 
response profiles, etc.


## experiments ##

Scripts for running particular analyses.

