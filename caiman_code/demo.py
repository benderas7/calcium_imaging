# Import necessary modules
import caiman_code.funcs
import logging

# Logging parameters
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
VIDEO_FN = '../CaImAn/example_movies/demoMovie.tif'
DISP_MOVIE = False
SAVE_RESULTS_DIR = '.'

# Dataset dependent parameters
FR = 30  # imaging rate in frames per second
DECAY_TIME = 0.4  # length of a typical transient in seconds

DEFINED_OPTS = {
    # Motion correction parameters
    'strides': (48, 48),  # start a new patch for pw-rigid motion correction
    # every x pixels
    'overlaps': (24, 24),  # overlap between paths (size of patch
    # strides+overlaps),
    'max_shifts': (6, 6),  # maximum allowed rigid shifts (in pixels)
    'max_deviation_rigid': 3,  # maximum shifts deviation allowed for patch
    # with respect to rigid shifts
    'pw_rigid': True,  # flag for performing non-rigid motion correction

    # Parameters for source extraction and deconvolution
    'p': 1,  # order of the autoregressive system
    'nb': 2,  # number of global background components
    'rf': 15,  # half-size of the patches in pixels. e.g., if rf=25, patches
    # are 50x50
    'K': 4,  # number of components per patch
    'stride_cnmf': 6,  # amount of overlap between the patches in pixels
    'gSig': (4, 4),  # expected half size of neurons in pixels
    'method_init': 'greedy_roi',  # initialization method (if analyzing
    # dendritic data using 'sparse_nmf')
    'rolling_sum': True,
    'only_init': True,
    'ssub': 1,  # spatial subsampling during initialization
    'tsub': 1,  # temporal subsampling during intialization
    'merge_thr': 0.85,  # merging threshold, max correlation allowed

    # Parameters for component evaluation
    'min_SNR': 2.0,  # signal to noise ratio for accepting a component
    'rval_thr': 0.85,  # space correlation threshold for accepting a component
    'use_cnn': True,
    'min_cnn_thr': 0.99,  # threshold for CNN based classifier
    'cnn_lowest': 0.1,  # neurons with cnn probability lower than this value
    # are
                }
######


def run_demo(opts_dict, video_fn=VIDEO_FN, log=LOG, log_fn=LOG_FN,
             log_level=LOG_LEVEL, fr=FR, decay_time=DECAY_TIME,
             disp_movie=DISP_MOVIE, save_results_dir=SAVE_RESULTS_DIR):
    caiman_code.funcs.pipeline(
        video_fn, log, log_fn, log_level, fr, decay_time, opts_dict,
        save_results_dir, disp_movie=disp_movie)
    return


if __name__ == '__main__':
    run_demo(DEFINED_OPTS)
