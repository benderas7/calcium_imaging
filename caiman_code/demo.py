# Import necessary modules
import caiman_code.funcs
import logging

# Logging parameters
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
VIDEO_FN = 'CaImAn/example_movies/demoMovie.tif'
DISP_MOVIE = True
SAVE_RESULTS_DIR = '.'

# Dataset dependent parameters
FR = 30  # imaging rate in frames per second
DECAY_TIME = 0.4  # length of a typical transient in seconds

# Motion correction parameters
STRIDES = (48, 48)  # start a new patch for pw-rigid motion correction every
# x pixels
OVERLAPS = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
MAX_SHIFTS = (6, 6)  # maximum allowed rigid shifts (in pixels)
MAX_DEV_RIGID = 3  # maximum shifts deviation allowed for patch with
# respect to rigid shifts
PW_RIGID = True  # flag for performing non-rigid motion correction

# Parameters for source extraction and deconvolution
P = 1  # order of the autoregressive system
GNB = 2  # number of global background components
MERGE_THR = 0.85  # merging threshold, max correlation allowed
RF = 15  # half-size of the patches in pixels. e.g., if rf=25, patches are
# 50x50
STRIDE_CNMF = 6  # amount of overlap between the patches in pixels
K = 4  # number of components per patch
GSIG = [4, 4]  # expected half size of neurons in pixels
METHOD_INIT = 'greedy_roi'  # initialization method (if analyzing dendritic
# data using 'sparse_nmf')
SSUB = 1  # spatial subsampling during initialization
TSUB = 1  # temporal subsampling during intialization

# Parameters for component evaluation
MIN_SNR = 2.0  # signal to noise ratio for accepting a component
RVAL_THR = 0.85  # space correlation threshold for accepting a component
CNN_THR = 0.99  # threshold for CNN based classifier
CNN_LOWEST = 0.1  # neurons with cnn probability lower than this value are
# rejected
######


def run_demo(video_fn=VIDEO_FN, log=LOG, log_fn=LOG_FN, log_level=LOG_LEVEL,
             fr=FR, decay_time=DECAY_TIME, disp_movie=DISP_MOVIE,
             save_results_dir=SAVE_RESULTS_DIR):
    caiman_code.funcs.pipeline(video_fn, log, log_fn, log_level, fr,
                               decay_time, disp_movie, save_results_dir)
    return


if __name__ == '__main__':
    run_demo()
