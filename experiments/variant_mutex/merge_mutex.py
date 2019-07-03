
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.extend([os.path.join(base_dir, '../../..')])

import argparse
import pandas as pd
import dill as pickle
from glob import glob
from HetMan.experiments.variant_mutex.utils import compare_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('use_dir', type=str, default=base_dir)
    args = parser.parse_args()

    out_data = [pickle.load(open(fl, 'rb'))
                for fl in glob(os.path.join(args.use_dir,
                                            'output', "out_task-*.p"))]

    use_clfs = set(out_dict['Info']['Clf'] for out_dict in out_data)
    assert len(use_clfs) == 1, ("Each experiment must be run with "
                                "exactly one classifier!")

    use_tune = set(out_dict['Info']['Clf'].tune_priors
                   for out_dict in out_data)
    assert len(use_tune) == 1, ("Each experiment must be run with "
                                "exactly one set of tuning priors!")

    pairs_list = pickle.load(open(os.path.join(
        args.use_dir, 'setup', "pairs-list.p"), 'rb'))

    out_dfs = {
        k: pd.concat([pd.DataFrame.from_dict(out_dict[k], orient='index')
                      for out_dict in out_data])
        for k in ['Infer', 'Tune']
        }

    assert len(pairs_list) == len(out_dfs['Infer'].index), (
        "Number of mutation pairs tested in the fitting stage does not match "
        "the number of pairs enumerated during setup!"
        )

    for mtype1, mtype2 in out_dfs['Infer'].index:
        assert ((mtype1, mtype2) in pairs_list
                or (mtype2, mtype1) in pairs_list), (
                    "Enumerated pair {} + {} is missing from the list of "
                    "tested pairs!".format(mtype1, mtype2)
                    )

    with open(os.path.join(args.use_dir, "out-data.p"), 'wb') as fl:
        pickle.dump({'Infer': out_dfs['Infer'], 'Tune': out_dfs['Tune'],
                     'Clf': tuple(use_clfs)[0],
                     'TunePriors': tuple(use_tune)[0]}, fl)

    with open(os.path.join(args.use_dir,
                           'setup', "cohort-data.p"), 'rb') as fl:
        cdata = pickle.load(fl)

    with open(os.path.join(args.use_dir, "out-simil.p"), 'wb') as fl:
        pickle.dump(compare_pairs(out_dfs['Infer'], cdata,
                                  get_similarities=True), fl)


if __name__ == "__main__":
    main()

