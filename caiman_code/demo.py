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
DEFINED_OPTS = {'strides': (48, 48),  # start a new patch for pw-rigid motion
                # correction every x pixels
                'max_shifts': (6, 6),  # maximum allowed rigid shifts (in
                # pixels)
                'rf': 15,  # half-size of the patches in pixels. e.g.,
                # if rf=25, patches are 50x50
                'stride_cnmf': 6,   # amount of overlap between the patches
                # in pixels
                'gsig': (4, 4)  # expected half size of neurons in pixels
                }
######


def run_demo(video_fn=VIDEO_FN, log=LOG, log_fn=LOG_FN, log_level=LOG_LEVEL,
             fr=FR, decay_time=DECAY_TIME, disp_movie=DISP_MOVIE,
             save_results_dir=SAVE_RESULTS_DIR, defined_opts=None):
    caiman_code.funcs.pipeline(
        video_fn, log, log_fn, log_level, fr, decay_time, disp_movie,
        save_results_dir, defined_opts=defined_opts)
    return


if __name__ == '__main__':
    run_demo(defined_opts=DEFINED_OPTS)
