
import os
import pandas as pd

domain_dir = '/home/exacloud/lustre1/CompBio/genomic_resources/'


def get_protein_domains(domain_lbl):
    domain_file = os.path.join(domain_dir,
                               '{}_to_gene.txt.gz'.format(domain_lbl))
    
    domain_data = pd.read_csv(domain_file, sep='\t')
    domain_data.columns = ["Gene", "Transcript",
                           "DomainID", "DomainStart", "DomainEnd"]

    return domain_data

