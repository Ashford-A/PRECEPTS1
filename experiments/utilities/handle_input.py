
import time
import bz2
import dill as pickle


def safe_load(fl, retry_pause=53):
    load_data = None

    while load_data is None:
        try:
            if fl[-2:] == 'gz':
                with bz2.BZ2File(fl, 'r') as data_f:
                    load_data = pickle.load(data_f)

            else:
                with open(fl, 'rb') as data_f:
                    load_data = pickle.load(data_f)

        except:
            print("Failed to load data from\n{}\ntrying again...".format(fl))
            time.sleep(retry_pause)

    return load_data

