"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
import caiman_code.funcs as funcs
from caiman_code.worm import COMPILED_DIR

# Set parameters
DISP_MOVIE = False
COLOR_COMPS = True
SAVE_NAME = 'results_movie.avi'
#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    return cnm


def make_movie(cnm, disp_movie, color_comps, save_fn):
    # Get images from load memmap
    images = funcs.load_memmap(cnm.mmap_file)

    # Make video for each ROI
    mov = cnm.estimates.play_movie(
        images, display=disp_movie, use_color=color_comps,
        save_movie=bool(save_fn), movie_name=save_fn)
    return mov


def main(results_dir=COMPILED_DIR, disp_movie=DISP_MOVIE,
         color_comps=COLOR_COMPS, save_name=SAVE_NAME):
    # Load results
    cnm = load_results(results_dir)

    # Make movie
    save_fn = os.path.join(results_dir, save_name)
    make_movie(cnm, disp_movie, color_comps, save_fn)
    return


if __name__ == '__main__':
    main(results_dir='..')
