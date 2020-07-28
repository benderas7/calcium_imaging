# Import necessary modules
import logging
import os
import re
from PIL import Image
import numpy as np
import h5py
from tifffile import imsave, imread

# Logging parameters
LOG = True
LOG_FN = '/tmp/suite2p.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
IMG_DIR = '/Users/benderas/NeuroPAL/Test'
CAIMAN_DIR = '/Users/benderas/NeuroPAL/Compiled/Test'
COMPILED_DIR = '/Users/benderas/NeuroPAL/Compiled/Test2'
ARR_FORMAT = '.h5'

# Dataset dependent parameters
FR = 485 / 300  # imaging rate in frames per second
DECAY_TIME = 0.5  # length of a typical transient in seconds
######


def compile_imgs_to_arr(img_dir, compiled_dir, arr_format, t_char='t',
                        z_char='z'):
    # Determine array file name from img_dir
    arr_fn = os.path.join(compiled_dir, '{}{}'.format(
        img_dir.split('/')[-1], arr_format))

    # Check if array has already been compiled and saved
    if os.path.exists(arr_fn):
        if '.h5' in arr_fn:
            h5f = h5py.File(arr_fn, 'r')
            arr = h5f['data'][:]
            h5f.close()
            return arr_fn, arr.shape
        elif '.tif' in arr_fn:
            arr = imread(arr_fn)
            return arr_fn, arr.shape

    # Make sure video directory is in fact a directory
    assert os.path.isdir(img_dir)

    # Make compiled dir if necessary
    if not os.path.exists(compiled_dir):
        os.makedirs(compiled_dir)

    # Load images
    imgs, t_lst, z_lst = [], [], []
    files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    for fn in files:
        # Parse filename to get t and z - *very specific to worm files*
        tz_str = fn.split('_')[-2]
        t_lst.append(int(re.findall('\d+', tz_str[tz_str.index(t_char):])[0]))
        z_lst.append(int(re.findall('\d+', tz_str[tz_str.index(z_char):])[0]))

        # Load image
        imgs.append(np.array(Image.open(os.path.join(img_dir, fn))))

    # Compile images into array
    arr = np.zeros((len(set(t_lst)), len(set(z_lst)), *imgs[0].shape))
    for img, t, z in zip(imgs, t_lst, z_lst):
        assert np.sum(arr[t-1, z-1, :, :]) == 0
        arr[t-1, z-1, :, :] = img

    # Save array
    if '.h5' in arr_fn:
        h5f = h5py.File(arr_fn, 'w')
        h5f.create_dataset('data', data=arr)
        h5f.close()
        print('Saved array with shape: {} as {}'.format(arr.shape, arr_fn))
    elif '.tif' in arr_fn:
        imsave(arr_fn, arr)
        print('Saved array with shape: {} as {}'.format(arr.shape, arr_fn))
    return arr_fn, arr.shape


def adapt_caiman_h5(caiman_dir, img_dir, compiled_dir):
    # Extract array from CaImAn h5
    h5 = [os.path.join(caiman_dir, f) for f in os.listdir(caiman_dir) if
          f.endswith('.h5')][0]
    with h5py.File(h5, 'r') as f:
        arr = np.array(f[list(f.keys())[0]])

    # Move z axis to make array compatible with suite2p
    arr = np.moveaxis(arr, 3, 1)

    # Make compiled dir if necessary
    if not os.path.exists(compiled_dir):
        os.makedirs(compiled_dir)

    # Save array
    arr_fn = os.path.join(compiled_dir, '{}.h5'.format(
        img_dir.split('/')[-1]))
    h5f = h5py.File(arr_fn, 'w')
    h5f.create_dataset('data', data=arr)
    h5f.close()
    print('Saved array with shape: {} as {}'.format(arr.shape, arr_fn))
    return


def main(img_dir=IMG_DIR, arr_format=ARR_FORMAT, compiled_dir=COMPILED_DIR,
         caiman_dir=CAIMAN_DIR):
    if not caiman_dir:
        # Compile images into array
        compile_imgs_to_arr(img_dir, compiled_dir, arr_format)
        return

    adapt_caiman_h5(caiman_dir, img_dir, compiled_dir)
    return


if __name__ == '__main__':
    main()
