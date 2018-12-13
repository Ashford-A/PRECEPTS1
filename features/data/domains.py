
import os
import pandas as pd


def get_protein_domains(domain_dir, domain_lbl):
    domain_file = os.path.join(domain_dir,
                               '{}_to_gene.txt.gz'.format(domain_lbl))
    
    domain_data = pd.read_csv(domain_file, sep='\t')
    domain_data.columns = ["Gene", "Transcript",
                           "DomainID", "DomainStart", "DomainEnd"]

    return domain_data

