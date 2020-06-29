"""
@author:benderas7
"""
# Import necessary modules
import h5py
import os

from caiman_code.worm import COMPILED_DIR

#####


def load_results(results_dir):
    fn = os.path.join(results_dir, 'analysis_results.hdf5')
    h5f = h5py.File(fn, 'r')
    return h5f['estimates']


def make_video_roi():
    return


if __name__ == '__main__':
    load_results(results_dir='..')
