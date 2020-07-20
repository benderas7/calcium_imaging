"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf.initialization import downscale
import caiman
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from moviepy.config import change_settings
import moviepy.editor as moviepy
import caiman_code.funcs as funcs
from caiman_code.worm import COMPILED_DIR

# Set parameters
#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    print('Number of components: {}'.format(cnm.estimates.C.shape[0]))
    return cnm


def _movie_one_slice(
        estimates, comp_slice, cols_c, dims, imgs, frame_range, slice_dir,
        q_max=99.75, q_min=2, magnification=1, gain_bck=0.2,
        include_bck=True, movie_name='results_movie.avi'):
    estimates.select_components(idx_components=comp_slice)

    cs = np.expand_dims(
        estimates.C[:, frame_range], -1) * cols_c[:estimates.C.shape[0], :, :]
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
    y_rec_color_by_z = [np.squeeze(arr) for arr in np.split(
        y_rec_color, min(y_rec_color.shape[:-1]), axis=int(np.argmin(
            y_rec_color.shape[:-1])))]

    for i, (imgs_1z, y_rec_1z, y_rec_color_1z, b_1z) in enumerate(zip(
            imgs_by_z, y_rec_by_z, y_rec_color_by_z, b_by_z)):
        per_i = movie_name.index('.')
        movie_fn = '{}_z{}{}'.format(
            movie_name[:per_i], i, movie_name[per_i:])
        movie_fn_full_path = os.path.join(slice_dir, movie_fn)

        if not os.path.exists(movie_fn_full_path):
            mov = caiman.concatenate((np.repeat(np.expand_dims(imgs_1z - (
                not include_bck) * b_1z, -1), 3, 3), y_rec_color_1z +
                                      include_bck * np.expand_dims(
                                          b_1z * gain_bck, -1)), axis=2)

            mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                     save_movie=bool(slice_dir),
                     movie_name=movie_fn_full_path)

            del mov
            
    estimates.restore_discarded_components()
    return


def play_movie_custom(
        estimates, imgs, n_comps_per_slice=12, save_dir=None, cmap='hsv',
        frame_range=slice(None, None, None), gain_color=4,
        colors_name='results_movie_colors.npy'):
    """Adapted from caiman/source_extraction/cnmf/estimates.py for 3D video."""
    dims = imgs.shape[1:]
    if 'movie' not in str(type(imgs)):
        imgs = caiman.movie(imgs[frame_range])
    else:
        imgs = imgs[frame_range]

    cols_c_fn = os.path.join(save_dir, colors_name)
    if os.path.exists(cols_c_fn):
        cols_c = np.load(cols_c_fn)
    else:
        cols_c = plt.get_cmap(cmap)(np.linspace(
            0, 1, n_comps_per_slice))[:, np.newaxis, :-1] * gain_color
        if save_dir:
            np.save(cols_c_fn, cols_c)

    for j in range(int(np.ceil(estimates.C.shape[0] / n_comps_per_slice))):
        comp_slice = [val for val in range(j * n_comps_per_slice, (
                j + 1) * n_comps_per_slice) if val < estimates.C.shape[0]]
        slice_dir = os.path.join(save_dir, 'comps{}-{}'.format(
            comp_slice[0], comp_slice[-1]))
        if not os.path.exists(slice_dir):
            os.makedirs(slice_dir)

        if not len(os.listdir(slice_dir)) >= imgs.shape[-1]:
            _movie_one_slice(estimates, comp_slice, cols_c, dims, imgs,
                             frame_range, slice_dir)
    return cols_c


def make_movie(cnm, save_dir):
    # Get images from load memmap
    imgs = funcs.load_memmap(cnm.mmap_file)

    # Make video for each ROI
    cols_c = play_movie_custom(
        cnm.estimates, imgs, save_dir=save_dir)
    return cols_c, imgs


def stack_movies(movie_dir, n_cols=2):
    """Stack movies from different z-stacks and integrate into one movie
    file for each set of components."""
    # Load folders for each component set
    comps_dirs = [os.path.join(movie_dir, d) for d in os.listdir(movie_dir) if
                  os.path.isdir(os.path.join(movie_dir, d))]

    for comps_dir in comps_dirs:
        # Load movies
        change_settings({"IMAGEMAGICK_BINARY":
                        "/usr/local/Cellar/imagemagick/7.0.10-23/bin/convert"})
        files = [os.path.join(comps_dir, f) for f in os.listdir(comps_dir) if
                 f.endswith('.avi')]
        clips = []
        for f in files:
            clip = moviepy.VideoFileClip(os.path.join(movie_dir, f))
            z = f.split('z')[-1].split('.')[0]
            txt_clip = moviepy.TextClip(z, color='white')
            txt_clip = txt_clip.set_position(('right', 'top')).set_duration(60)
            clips.append(moviepy.CompositeVideoClip([clip, txt_clip]))

        # Make clips array
        if len(clips) % n_cols != 0:
            clips.extend([clips[0].fl_image(lambda im: 0*im)] * (
                    n_cols - len(clips) % n_cols))
        clips_arr = [clips[i:i+n_cols] for i in range(0, len(clips), n_cols)]
        composite = moviepy.clips_array(clips_arr)

        # Save file
        comp_fn = os.path.join(comps_dir, 'composite.mp4')
        if not os.path.exists(comp_fn):
            composite.write_videofile(comp_fn)
    return


def colored_traces(cnm, imgs, cols_c, save_dir, n_comps_per_slice=12, n_cols=3,
                   gain_color=4):
    """Plot and savee traces for each component in color that they are shown
    in the video."""
    # Get total number of components
    n_comps_total = cnm.estimates.C.shape[0]
    for j in range(int(np.ceil(n_comps_total / n_comps_per_slice))):
        # Select desired components
        comp_slice = [val for val in range(j * n_comps_per_slice, (
                j + 1) * n_comps_per_slice) if val < cnm.estimates.C.shape[0]]
        cnm.estimates.select_components(idx_components=comp_slice)

        # Extract traces and spatial footprints of components
        traces = cnm.estimates.C
        spat_fp = cnm.estimates.A.toarray().reshape(
            imgs.shape[1:] + (-1,), order='F')

        # Remove unnecessary dimensions and map values between 0 and 1
        cols_c = np.squeeze(cols_c)
        cols_c = cols_c / gain_color

        # Plot traces
        count = 0
        fig, axes = plt.subplots(n_comps_per_slice, n_cols, figsize=(15, 10))
        for i, (c, trace) in enumerate(zip(cols_c, traces)):
            axes.flatten()[i].plot(trace, c=c)
            xyz = spat_fp[:, :, :, i]
            axes.flatten()[i].set_title(
                'Comp {}; Z:{}; ''Center: {}'.format(
                    count, np.argwhere(np.sum(xyz, axis=(0, 1))).flatten(),
                    [int(val) for val in center_of_mass(xyz)]))
            axes.flatten()[i].set_xlabel('')
            count += 1
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comps{}-{}'.format(
            comp_slice[0], comp_slice[-1])))

        cnm.estimates.restore_discarded_components()
    return


def main(results_dir=COMPILED_DIR):
    # Load results
    cnm = load_results(results_dir)

    # Make movie
    movie_dir = os.path.join(results_dir, 'movies')
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)
    cols_c, imgs = make_movie(cnm, movie_dir)
    stack_movies(movie_dir)

    # Make traces for each component colored as in video
    traces_dir = os.path.join(results_dir, 'traces')
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)
    colored_traces(cnm, imgs, cols_c, traces_dir)
    return


if __name__ == '__main__':
    main()
