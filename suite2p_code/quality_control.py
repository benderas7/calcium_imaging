"""
@author: Andrew Bender
"""
# Import necessary modules
import numpy as np
import os
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries, mark_boundaries
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc

# Set constants
DATA_DIR = '/Users/benderas/NeuroPAL/Compiled/Test2/suite2p'
####


def get_plane_dirs(data_dir=DATA_DIR):
    # Get directory for each plane
    plane_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                  os.path.isdir(os.path.join(data_dir, f)) and 'plane' in f]
    return plane_dirs


def load_results_one_plane(data_dir=DATA_DIR):
    # Get all .npy files in data directory
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
             f.endswith('.npy')]

    # Load all results
    res = {f.split('/')[-1].split('.')[0]: np.load(f, allow_pickle=True) for f
           in files}
    return res


def make_traces_one_plane(cnm, imgs, save_dir, cols_c=None,
                         n_comps_per_slice=12, n_cols=3, gain_color=16):
    """Plot and save traces for each component in color that they are shown
    in the video."""
    # Get total number of components
    n_comps_total = cnm.estimates.C.shape[0]

    # Remove unnecessary dimensions and map values between 0 and 1
    if cols_c is not None:
        cols_c = np.squeeze(cols_c)
        cols_c = cols_c / gain_color
    else:
        cols_c = [None] * n_comps_per_slice

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

        # Plot traces and center maps
        n_rows = n_comps_per_slice // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig2, ax2 = None, None
        if cols_c[0] is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 10))
        for i, (c, trace) in enumerate(zip(cols_c, traces)):
            # Traces
            axes.flatten()[i].plot(trace, c=c)
            xyz = spat_fp[:, :, :, i]
            axes.flatten()[i].set_title(
                'Comp {}; Z:{}; ''Center: {}'.format(
                    count, np.argwhere(np.sum(xyz, axis=(0, 1))).flatten(),
                    [int(val) for val in center_of_mass(xyz)]))
            axes.flatten()[i].set_xlabel('')

            if c is not None:
                # Center maps
                ax2.scatter(*center_of_mass(xyz)[:2], c=c)
                ax2.annotate(count, center_of_mass(xyz)[:2])
                ax2.set_xlim([0, spat_fp.shape[0]])
                ax2.set_ylim([0, spat_fp.shape[1]])
            count += 1

        # Save figures
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'comps{}-{}'.format(
            comp_slice[0], comp_slice[-1])))
        if cols_c[0] is not None:
            fig2.tight_layout()
            fig2.savefig(os.path.join(save_dir, 'comps{}-{}_centers'.format(
                comp_slice[0], comp_slice[-1])))

        # Restore components
        cnm.estimates.restore_discarded_components()
    return


def make_movie_each_comp_one_plane(cnm, save_dir):
    """Make movie for each component."""
    plane_dirs =
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
        video = VideoWriter(os.path.join(save_dir, 'comp{}_z{}.avi'.format(
            i, max_z)), VideoWriter_fourcc(*'MJPG'), fps, video_bound[0].shape[
            :-1][::-1])
        for frame in video_bound:
            video.write((frame * 255).astype(np.uint8))
        video.release()
    return imgs


def main():
    # Get plane directories
    plane_dirs = get_plane_dirs()

    for plane_dir in plane_dirs:
        # Load all results
        load_results_one_plane()

        # # Make movie for each components
        # make_movie_each_comp_one_plane()
        #
        # # Make traces for each component
        # make_traces_one_plane()

    return


if __name__ == '__main__':
    main()