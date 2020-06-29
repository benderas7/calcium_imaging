"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
import cv2
import matplotlib.pyplot as plt
from caiman_code.worm import COMPILED_DIR

#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    return cnm


def main():
    # Load results
    cnm = load_results(results_dir='..')
    return


if __name__ == '__main__':


