# Import necessary modules
import caiman_code.funcs
import logging
import os
import re

# Logging parameters
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
VIDEO_DIR = 'data/NeuroPAL/11.25.19/worm3_gcamp_Out'
DISP_MOVIE = True
SAVE_RESULTS_DIR = '.'

# Dataset dependent parameters
FR = 30  # imaging rate in frames per second
DECAY_TIME = 0.4  # length of a typical transient in seconds
######


def compile_vids_to_arr(video_dir, t_char='t', z_char='z'):
    # Make sure video directory is in fact a directory
    assert os.path.isdir(video_dir)

    # Compile videos
    for fn in [f for f in os.listdir(video_dir) if f.endswith('.tif')]:
        # Parse filename to get t and z - *very specific to worm files*
        tz_str = fn.split('_')[3]
        t = re.findall('\d+', tz_str[tz_str.index(t_char):])[0]
        z = re.findall('\d+', tz_str[tz_str.index(z_char):])[0]

    arr = []
    return arr


def run(video_dir=VIDEO_DIR, log=LOG, log_fn=LOG_FN, log_level=LOG_LEVEL,
        fr=FR, decay_time=DECAY_TIME, disp_movie=DISP_MOVIE,
        save_results_dir=SAVE_RESULTS_DIR):
    # Compile videos into array
    video_fn = compile_vids_to_arr(video_dir)

    # Run pipeline
    caiman_code.funcs.pipeline(video_fn, log, log_fn, log_level, fr,
                               decay_time, disp_movie, save_results_dir)
    return


if __name__ == '__main__':
    run()
