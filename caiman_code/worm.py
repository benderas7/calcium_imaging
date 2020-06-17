# Import necessary modules
import caiman_code.funcs
import logging
import os
import re
from PIL import Image
import numpy as np
from skimage.external.tifffile import imsave

# Logging parameters
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
IMG_DIR = '/Users/benderas/NeuroPAL/11.25.19/worm3_gcamp_Out'
SAVE_RESULTS_DIR = '.'

# Dataset dependent parameters
IS_3D = True
FR = 30  # imaging rate in frames per second
DECAY_TIME = 0.4  # length of a typical transient in seconds
######


def compile_imgs_to_arr(img_dir, t_char='t', z_char='z'):
    # Determine array file name from img_dir
    arr_fn = os.path.join(img_dir, '{}.tif'.format(img_dir.split('/')[-1]))

    # Check if array has already been compiled and saved
    if os.path.exists(arr_fn):
        return arr_fn

    # Make sure video directory is in fact a directory
    assert os.path.isdir(img_dir)

    # Load images
    imgs, t_lst, z_lst = [], [], []
    for fn in [f for f in os.listdir(img_dir) if f.endswith('.tif')]:
        # Parse filename to get t and z - *very specific to worm files*
        tz_str = fn.split('_')[3]
        t_lst.append(int(re.findall('\d+', tz_str[tz_str.index(t_char):])[0]))
        z_lst.append(int(re.findall('\d+', tz_str[tz_str.index(z_char):])[0]))

        # Load image
        imgs.append(np.array(Image.open(os.path.join(img_dir, fn))))

    # Compile images into array
    arr = np.zeros((len(set(t_lst)), *imgs[0].shape, len(set(z_lst)), ))
    for img, t, z in zip(imgs, t_lst, z_lst):
        assert np.sum(arr[t-1, :, :, z-1]) == 0
        arr[t-1, :, :, z-1] = img

    # Save array
    imsave(arr_fn, arr)
    print('Saved array with shape: {} as {}'.format(arr.shape, arr_fn))
    return arr_fn


def run(img_dir=IMG_DIR, log=LOG, log_fn=LOG_FN, log_level=LOG_LEVEL,
        fr=FR, decay_time=DECAY_TIME, save_results_dir=SAVE_RESULTS_DIR,
        is_3d=IS_3D):
    # Compile images into array
    video_fn = compile_imgs_to_arr(img_dir)

    # Run pipeline
    caiman_code.funcs.pipeline(video_fn, log, log_fn, log_level, fr,
                               decay_time, save_results_dir, is_3d=is_3d)
    return


if __name__ == '__main__':
    run()
