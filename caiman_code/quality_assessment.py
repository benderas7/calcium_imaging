"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
import caiman_code.funcs as funcs
from caiman_code.worm import COMPILED_DIR
#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    return cnm


def make_movie(cnm):
    # Get images from load memmap
    images = funcs.load_memmap(cnm.mmap_file)

    # Make video for each ROI
    mov = cnm.estimates.play_movie(
        images, display=False, use_color=True, save_movie=True)
    return mov


def main():
    # Load results
    cnm = load_results(results_dir='..')

    # Make movie
    make_movie(cnm)
    return


if __name__ == '__main__':
    main()
