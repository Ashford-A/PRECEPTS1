
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
