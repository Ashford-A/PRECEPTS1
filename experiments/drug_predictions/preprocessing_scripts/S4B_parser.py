"""
Parses contents of Iorio et al. Table S4B into a data matrix.
Data downloaded from:
http://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
on July 13, 2016 by Michal Grzadkowski.

Note: if this is to be repeated, ensure that the format of the original Excel
file is such that the entire float of each desired number is visible.
(i.e. if the number format allows 2 decimal places, data will be rounded to
2 decimal places when saved).

Author: Hannah Manning

"""

import os
import sys
import pandas as pd

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path += [base_dir + '/../../../../bergamot']
file_path = base_dir + '/../../../data/drugs/ioria/'


def main():
    S4B_df = pd.read_csv(file_path + 'mmc5_float_S4B_AUCs.txt',
                         sep='\t', header=3, index_col=0)

    # match the format of the previous version, parsed in R
    S4B_df = S4B_df.drop('Sample Names', 1).drop(S4B_df.index[0],0)
    S4B_df.columns = ["X" + i for i in S4B_df.columns]
    S4B_df.index = ["DATA." + i for i in map(str, map(int, S4B_df.index))]

    # write comments in first two lines of file
    outf = open(file_path + 'drug-auc-full.txt', 'w')
    outf.write('# downloaded from http://www.cancerrxgene.org/gdsc1000'
               '/GDSC1000_WebResources/Home.html on July 13, 2016 by Michal '
               'Grzadkowski\n# contains the contents of Table S4B processed '
               'into a data matrix in Python July 19, 2017 by Hannah Manning '
               '(see S4B_parser.py)\n')

    # save the parsed data to file
    S4B_df.to_csv(outf, sep='\t')
    outf.close()


if __name__ == '__main__':
    main()

