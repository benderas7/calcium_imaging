"""
@author: Andrew Bender
"""
# Import necessary modules
import numpy as np
import os

# Set constants
DATA_DIR = '/Users/benderas/NeuroPAL/Compiled/' \
           'worm3_gcamp_Out2/suite2p/combined'
####


def load_results(data_dir=DATA_DIR):
    # Get all .npy files in data directory
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
             f.endswith('.npy')]

    # Load all results
    res = {f.split('/')[-1].split('.')[0]: np.load(f, allow_pickle=True) for f
           in files}
    return res


def main():
    # Load all results
    load_results()
    return


if __name__ == '__main__':
    main()