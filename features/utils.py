
"""Utilities for loading and processing features.

Author: Michal Grzadkowski <grzadkow@ohsu.edu>

"""

from ophion import Ophion


def choose_bmeg_server(verbose=False):
    """Chooses a BMEG server to use based on availability."""

    # list of BMEG servers to try
    server_list = ['http://bmeg.compbio.ohsu.edu', 'http://bmeg.io']
    
    # iterate over these servers until we find one or there aren't any left
    server_found = False
    while not server_found and server_list:

        # get a new server and intialize the query interface
        bmeg_server = server_list.pop()
        oph = Ophion(bmeg_server)

        # ...check if we can run a simple query...
        try:
            proj_count = oph.query().has(
                "gid", "project:TCGA-BRCA").count().execute()[0]

            # ...if so, check if the query returns a proper value
            if int(proj_count) > 0:
                server_found = True

                if verbose:
                    print("Choosing BMEG server {}".format(bmeg_server))

        except:
            pass

    if not server_found:
        raise RuntimeError("No BMEG server available!")

    return bmeg_server

