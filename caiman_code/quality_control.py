"""
@author:benderas7
"""
# Import necessary modules
import os
from caiman.source_extraction.cnmf import cnmf
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries, mark_boundaries
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
import caiman_code.funcs as funcs
from caiman_code.worm import COMPILED_DIR

# Set parameters
OVERWRITE_VIDS = False
#####


def load_results(results_dir):
    # Determine filename from dir
    fn = os.path.join(results_dir, 'analysis_results.hdf5')

    # Load CNMF object using CaImAn function
    cnm = cnmf.load_CNMF(fn)
    print('Number of components: {}'.format(cnm.estimates.C.shape[0]))
    return cnm


def max_proj_vid(cnm, save_dir, save_name='max_proj'):
    # Get images from load memmap
    imgs = funcs.load_memmap(cnm.mmap_file)

    # Perform max projection
    max_proj = np.max(imgs, axis=3)

    # Write video
    fps = len(max_proj) // 60
    video = VideoWriter(os.path.join(save_dir, '{}.avi'.format(save_name)),
                        VideoWriter_fourcc(*'MJPG'), fps,
                        max_proj.shape[1:][::-1], 0)
    for frame in max_proj:
        video.write((frame * 255).astype(np.uint8))
    video.release()
    return


def make_traces(cnm, imgs, save_dir, n_comps_per_slice=12, n_cols=3):
    """Plot and save traces for each component in color that they are shown
    in the video."""
    # Get total number of components
    n_comps_total = cnm.estimates.C.shape[0]

    count = 0
    for j in range(int(np.ceil(n_comps_total / n_comps_per_slice))):
        # Select desired components
        comp_slice = [val for val in range(j * n_comps_per_slice, (
                j + 1) * n_comps_per_slice) if val < cnm.estimates.C.shape[0]]
        cnm.estimates.select_components(idx_components=comp_slice)

        # Extract traces and spatial footprints of components
        traces = cnm.estimates.C
        spat_fp = cnm.estimates.A.toarray().reshape(
            imgs.shape[1:] + (-1,), order='F')

        # Plot traces
        n_rows = n_comps_per_slice // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        for i, trace in enumerate(traces):
            axes.flatten()[i].plot(trace)
            xyz = spat_fp[:, :, :, i]
            axes.flatten()[i].set_title(
                'Comp {}; Z:{}; ''Center: {}'.format(
                    count, np.argwhere(np.sum(xyz, axis=(0, 1))).flatten(),
                    [int(val) for val in center_of_mass(xyz)]))
            axes.flatten()[i].set_xlabel('')
            count += 1

        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'comps{}-{}'.format(
            comp_slice[0], comp_slice[-1])))

        # Restore components
        cnm.estimates.restore_discarded_components()
    return


def make_movie_each_comp(cnm, save_dir, overwrite=OVERWRITE_VIDS):
    """Make movie for each component of z slice with largest area."""
    # Get images from load memmap
    imgs = funcs.load_memmap(cnm.mmap_file)

    # Get spatial footprints
    spat_fps = cnm.estimates.A.toarray().reshape(
        imgs.shape[1:] + (-1,), order='F')
    spat_fps = np.moveaxis(spat_fps, -1, 0)

    for i, spat_fp in enumerate(spat_fps):
        # Select z slice with largest area for component
        max_z = np.argmax(np.sum(spat_fp, axis=(0, 1)))
        spat_fp_max_z = spat_fp[:, :, max_z]
        imgs_max_z = imgs[:, :, :, max_z]

        # Determine save filename
        save_fn = os.path.join(save_dir, 'comp{}_z{}.avi'.format(i, max_z))

        if not os.path.exists(save_fn) or overwrite:
            # Modulate range of video between min and max of component
            roi_min = np.min(imgs_max_z[:, spat_fp_max_z > 0])
            roi_max = np.max(imgs_max_z[:, spat_fp_max_z > 0])
            imgs_max_z = (imgs_max_z - roi_min) / (roi_max - roi_min)
            imgs_max_z[imgs_max_z > 1] = 1
            imgs_max_z[imgs_max_z < 0] = 0

            # Draw boundary around component in video
            bound = find_boundaries(spat_fp_max_z, mode='inner')
            video_bound = [mark_boundaries(f, bound) for f in imgs_max_z]

            # Save video
            fps = len(video_bound) // 60
            video = VideoWriter(save_fn, VideoWriter_fourcc(*'MJPG'), fps,
                                video_bound[0].shape[:-1][::-1])
            for frame in video_bound:
                video.write((frame * 255).astype(np.uint8))
            video.release()
    return imgs


def main(results_dir=COMPILED_DIR):
    # Load results
    cnm = load_results(results_dir)

    # Make max projection video
    max_proj_vid(cnm, results_dir)

    # Make movie for each component of z slice with largest area
    movie_dir = os.path.join(results_dir, 'movies_each_comp')
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)
    imgs = make_movie_each_comp(cnm, movie_dir)

    # Make traces for each component
    traces_dir = os.path.join(results_dir, 'traces_each_comp')
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)
    make_traces(cnm, imgs, traces_dir)
    return


if __name__ == '__main__':
    main()
