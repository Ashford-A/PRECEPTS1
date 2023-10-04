PRECEPTS1 Workflow Experiments
------------------------------

This directory contains experiments meant to explore and interrogate the
relationships between expression and genomic data. These experiments come 
in several general types:


### scRNA Analysis (AML in this case, but can use custom bulk RNA-seq training/custom scRNA-seq test data) ###

This experiment explores the results from training models to detect the
presence - or absence - of specific cancer-associated mutations in bulk
RNA-seq data. The trained models are then applied to single-cell
data and scores are generated that represent the scale to which the models
predict the corresponding mutations in the samples' DNA. Further analysis
is required to interrogate these mutation prediction scores. These scores
can be validated in a few different ways, including comparing the bulk 
variant allele frequencies and using joint modality scRNA-seq/mutant 
genotyping calls.


### Subgrouping Test ###

This experiment largely focuses on segregating and interrogating many 
possible subgroupings of user-specified biological feature levels. These
scores are generated from models trained using bulk RNA-seq data, then
uses a bulk RNA-seq test-set that was hidden from the model from the 
initial data. These subgroupings can give us information on specific 
mutations' downstream transcriptomic signatures.
