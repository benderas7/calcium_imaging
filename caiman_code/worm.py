# Import necessary modules
import caiman_code.funcs
import logging
import os
import re
from PIL import Image
import numpy as np

# Logging parameters
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
IMG_DIR = 'data/NeuroPAL/11.25.19/worm3_gcamp_Out'
DISP_MOVIE = True
SAVE_RESULTS_DIR = '.'

# Dataset dependent parameters
FR = 30  # imaging rate in frames per second
DECAY_TIME = 0.4  # length of a typical transient in seconds
######


def compile_imgs_to_arr(img_dir, t_char='t', z_char='z'):
    # Make sure video directory is in fact a directory
    assert os.path.isdir(img_dir)

    # Load imagse
    imgs, t_lst, z_lst = [], [], []
    for fn in [f for f in os.listdir(img_dir) if f.endswith('.tif')]:
        # Parse filename to get t and z - *very specific to worm files*
        tz_str = fn.split('_')[3]
        t_lst.append(int(re.findall('\d+', tz_str[tz_str.index(t_char):])[0]))
        z_lst.append(int(re.findall('\d+', tz_str[tz_str.index(z_char):])[0]))

        # Load image
        imgs.append(np.array(Image.open(os.path.join(img_dir, fn))))

    # Compile images into array
    arr = np.zeros((*imgs[0].shape, len(set(z_lst)), len(set(t_lst))))
    for img, t, z in zip(imgs, t_lst, z_lst):
        assert np.sum(arr[:, :, z-1, t-1]) == 0
        arr[:, :, z-1, t-1] = img
    return arr


def run(img_dir=IMG_DIR, log=LOG, log_fn=LOG_FN, log_level=LOG_LEVEL,
        fr=FR, decay_time=DECAY_TIME, disp_movie=DISP_MOVIE,
        save_results_dir=SAVE_RESULTS_DIR):
    # Compile images into array
    video_fn = compile_imgs_to_arr(img_dir)

    # Run pipeline
    caiman_code.funcs.pipeline(video_fn, log, log_fn, log_level, fr,
                               decay_time, disp_movie, save_results_dir)
    return


if __name__ == '__main__':
    run()
