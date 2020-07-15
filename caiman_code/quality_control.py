"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf.initialization import downscale
import caiman
import numpy as np
import caiman_code.funcs as funcs
from caiman_code.worm import COMPILED_DIR

# Set parameters
SAVE_NAME = 'results_movie.avi'
#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    print('Number of components: {}'.format(cnm.estimates.C.shape[0]))
    return cnm


def play_movie_custom(
        estimates, imgs, q_max=99.75, q_min=2, gain_res=1, magnification=1, 
        include_bck=True, frame_range=slice(None, None, None),
        save_movie=False, movie_name='results_movie.avi', gain_color=4,
        gain_bck=0.2):
    """Adapted from caiman/source_extraction/cnmf/estimates.py for 3D video."""
    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = caiman.movie(imgs[frame_range])
    else:
        imgs = imgs[frame_range]

    cols_c = np.random.rand(estimates.C.shape[0], 1, 3) * gain_color
    cs = np.expand_dims(estimates.C[:, frame_range], -1) * cols_c
    y_rec_color = np.tensordot(estimates.A.toarray(), cs, axes=(1, 0))
    y_rec_color = y_rec_color.reshape(dims + (-1, 3), order='F')
    y_rec_color = np.moveaxis(y_rec_color, -2, 0)

    ac = estimates.A.dot(estimates.C[:, frame_range])
    y_rec = ac.reshape(dims + (-1,), order='F')
    y_rec = np.moveaxis(y_rec, -1, 0)
    if estimates.W is not None:
        ssub_b = int(round(np.sqrt(np.prod(dims) / estimates.W.shape[0])))
        b = imgs.reshape((-1, np.prod(dims)), order='F').T - ac
        if ssub_b == 1:
            b = estimates.b0[:, None] + estimates.W.dot(
                b - estimates.b0[:, None])
        else:
            wb = estimates.W.dot(
                downscale(b.reshape(dims + (b.shape[-1],), order='F'),
                          (ssub_b, ssub_b, 1)).reshape((-1, b.shape[-1]),
                                                       order='F'))
            wb0 = estimates.W.dot(downscale(estimates.b0.reshape(
                dims, order='F'), (ssub_b, ssub_b)).reshape(
                (-1, 1), order='F'))
            b = estimates.b0.flatten('F')[:, None] + (np.repeat(np.repeat(
                (wb - wb0).reshape(((dims[0] - 1) // ssub_b + 1,
                                    (dims[1] - 1) // ssub_b + 1, -1),
                                   order='F'),
                ssub_b, 0), ssub_b, 1)[:dims[0], :dims[1]].reshape(
                (-1, b.shape[-1]), order='F'))
        b = b.reshape(dims + (-1,), order='F')
        b = np.moveaxis(b, -1, 0)
    elif estimates.b is not None and estimates.f is not None:
        b = estimates.b.dot(estimates.f[:, frame_range])
        if 'matrix' in str(type(b)):
            b = b.toarray()
        b = b.reshape(dims + (-1,), order='F')
        b = np.moveaxis(b, -1, 0)
    else:
        b = np.zeros_like(y_rec)

    imgs_by_z = [np.squeeze(arr) for arr in np.split(
        imgs, min(imgs.shape), axis=int(np.argmin(imgs.shape)))]
    y_rec_by_z = [np.squeeze(arr) for arr in np.split(
        y_rec, min(y_rec.shape), axis=int(np.argmin(y_rec.shape)))]
    b_by_z = [np.squeeze(arr) for arr in np.split(
        b, min(b.shape), axis=int(np.argmin(b.shape)))]
    y_rec_color_1z = [np.squeeze(arr) for arr in np.split(
        y_rec_color, min(y_rec_color.shape[:-1]), axis=int(np.argmin(
            y_rec_color.shape[:-1])))]

    for i, (imgs_1z, y_rec_1z, y_rec_color_1z, b_1z) in enumerate(zip(
            imgs_by_z, y_rec_by_z, y_rec_color_1z, b_by_z)):
        y_res = imgs_1z - y_rec_1z - b_1z
        mov = caiman.concatenate((np.repeat(np.expand_dims(
            imgs_1z - (not include_bck) * b_1z, -1), 3, 3),
            y_rec_color_1z + include_bck * np.expand_dims(
                b_1z * gain_bck, -1), np.repeat(np.expand_dims(
                    y_res * gain_res, -1), 3, 3)), axis=2)

        per_i = movie_name.index('.')
        movie_fn = '{}_z{}{}'.format(movie_name[:per_i], i, movie_name[per_i:])

        mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                 save_movie=save_movie, movie_name=movie_fn)
    return cols_c


def make_movie(cnm, save_fn):
    # Get images from load memmap
    images = funcs.load_memmap(cnm.mmap_file)

    # Make video for each ROI
    cols_c = play_movie_custom(
        cnm.estimates, images, save_movie=bool(save_fn), movie_name=save_fn)
    np.save(save_fn.replace('.avi', '_colors.npy'), cols_c)
    return


def main(results_dir=COMPILED_DIR, save_name=SAVE_NAME):
    # Load results
    cnm = load_results(results_dir)

    # Make movie
    save_fn = os.path.join(results_dir, save_name)
    make_movie(cnm, save_fn)
    return


if __name__ == '__main__':
    main()
