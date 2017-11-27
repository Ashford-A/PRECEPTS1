
"""Utilities for loading and processing feature datasets.

This module contains utility functions that are commonly used in the
loading and processing of many different kinds of features. Utilities specific
to a particular type of feature are stored in the module corresponding to that
feature, see eg. :function:`.expression.log_norm_expr`.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ophion import Ophion
import numpy as np


def choose_bmeg_server(server_list=('http://bmeg.compbio.ohsu.edu',
                                    'http://bmeg.io'),
                       verbose=False):
    """Chooses a BMEG server to use based on availability.

    Args:
        server_list (:obj:`tuple` of :obj:`str`), optional
            The list of BMEG servers to try in reverse order of priority.
            The default is the list of servers that were available as of
            October 23, 2017.
        verbose (bool): Whether to print which BMEG server was chosen.

    Returns:
        bmeg_server (str): A BMEG server that is up and responding to queries.

    """
    server_found = False
    server_list = list(server_list)
    bmeg_server = None

    # iterate over the given servers until we find one that is working or
    # there aren't any left to try
    while not server_found and server_list:

        # initialize the query interface and see if we can run a simple query
        bmeg_server = server_list.pop()
        oph = Ophion(bmeg_server)
        try:
            proj_count = oph.query().has("gid", "project:TCGA-BRCA")\
                                .count().execute()[0]

            # if the query runs successfully, check if the query
            # returns a proper value
            if int(proj_count) > 0:
                server_found = True

                if verbose:
                    print("Choosing BMEG server {}".format(bmeg_server))

        except:
            pass

    if not server_found:
        raise RuntimeError("No BMEG server available!")

    return bmeg_server


def match_tcga_samples(samples1, samples2):
    """Finds the tumour samples common between two lists of TCGA barcodes.

    Args:
        samples1, samples2 (:obj:`list` of :obj:`str`)

    Returns:
        samps_match (list)

    """
    samps1 = list(set(samples1))
    samps2 = list(set(samples2))

    # parses the barcodes into their constituent parts
    parse1 = [samp.split('-') for samp in samps1]
    parse2 = [samp.split('-') for samp in samps2]

    # gets the names of the individuals associated with each sample
    partic1 = ['-'.join(prs[:3]) for prs in parse1]
    partic2 = ['-'.join(prs[:3]) for prs in parse2]

    # gets the type of each sample (tumour, control, etc.)
    type1 = np.array([int(prs[3][:2]) for prs in parse1])
    type2 = np.array([int(prs[3][:2]) for prs in parse2])

    # gets the vial each sample was tested in
    vial1 = [prs[3][-1] for prs in parse1]
    vial2 = [prs[3][-1] for prs in parse2]

    # finds the individuals with primary tumour samples in both lists
    partic_use = (set([prt for prt, tp in zip(partic1, type1) if tp < 10])
                  & set([prt for prt, tp in zip(partic2, type2) if tp < 10]))

    # finds the positions of the samples associated with these shared
    # individuals in the original lists of samples
    partic_indx = [([i for i, prt in enumerate(partic1) if prt == cur_prt],
                    [i for i, prt in enumerate(partic2) if prt == cur_prt])
                   for cur_prt in partic_use]

    # matches the samples of individuals with only one sample in each list
    samps_match = [(partic1[indx1[0]], (samps1[indx1[0]], samps2[indx2[0]]))
                   for indx1, indx2 in partic_indx
                   if len(indx1) == len(indx2) == 1]

    # for individuals with more than one sample in at least one of the two
    # lists, finds the sample in each list closest to the primary tumour type
    if len(partic_indx) > len(samps_match):
        choose_indx = [
            (indx1[np.argmin(type1[indx1])], indx2[np.argmin(type2[indx2])])
            for indx1, indx2 in partic_indx
            if len(indx1) > 1 or len(indx2) > 1
            ]

        samps_match += [(partic1[chs1], (samps1[chs1], samps2[chs2]))
                        for chs1, chs2 in choose_indx]

    return samps_match

