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
from skimage import io
from tqdm import tqdm

# Set constants
DATA_DIR = '/Users/benderas/NeuroPAL/Compiled/Test2/suite2p'
OVERWRITE_VIDS = False
####


def get_plane_dirs(data_dir=DATA_DIR):
    # Get directory for each plane
    plane_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                  os.path.isdir(os.path.join(data_dir, f)) and 'plane' in f]
    return sorted(plane_dirs)


def load_results_one_plane(plane_dir):
    # Get all .npy files in data directory
    files = [os.path.join(plane_dir, f) for f in os.listdir(plane_dir) if
             f.endswith('.npy')]

    # Load all results
    res = {f.split('/')[-1].split('.')[0]: np.load(f, allow_pickle=True) for f
           in files}
    return res


def make_movie_each_comp_one_plane(res, plane_dir, tif_dir_name='reg_tif',
                                   save_dir_name='movies',
                                   overwrite=OVERWRITE_VIDS):
    """Make movie for each component."""
    # Make directories to save videos if necessary
    save_dir = os.path.join(plane_dir, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get array of motion-corrected frames
    tif_dir = os.path.join(plane_dir, tif_dir_name)
    tif_fns = sorted(os.path.join(tif_dir, f) for f in os.listdir(tif_dir))
    arr = np.concatenate([np.array(io.imread(tif)) for tif in tif_fns])

    tqdm_desc = 'Plane {} Videos'.format(plane_dir.split('e')[-1])
    for i, (stat_one_comp, F_one_comp) in enumerate(tqdm(zip(
            res['stat'], res['F']), total=len(res['stat']), desc=tqdm_desc)):
        # Make filename for video
        vid_fn = os.path.join(save_dir, 'comp{}.avi'.format(i))

        if not os.path.exists(vid_fn) or overwrite:
            # Get spatial footprint for component
            spat_fp = np.zeros(arr.shape[1:])
            for x, y in zip(stat_one_comp['xpix'], stat_one_comp['ypix']):
                spat_fp[y, x] = 1

            # Modulate range of video between min and max of component
            roi_min = np.min(arr[:, spat_fp > 0])
            roi_max = np.max(arr[:, spat_fp > 0])
            arr_one_comp = (arr - roi_min) / (roi_max - roi_min)
            arr_one_comp[arr_one_comp > 1] = 1
            arr_one_comp[arr_one_comp < 0] = 0

            # Draw boundary around component in video
            # noinspection PyTypeChecker
            bound = find_boundaries(spat_fp, mode='inner')
            video_bound = [mark_boundaries(f, bound) for f in arr_one_comp]

            # Save video
            fps = len(video_bound) // 60
            video = VideoWriter(vid_fn, VideoWriter_fourcc(*'MJPG'), fps,
                                video_bound[0].shape[:-1][::-1])
            for frame in video_bound:
                video.write((frame * 255).astype(np.uint8))
            video.release()
    return arr


def make_traces_one_plane(res, plane_dir, tif_dir_name='reg_tif', n_cols=3,
                          n_comps_per_slice=12, save_dir_name='traces'):
    """Plot and save traces for each component in plane."""
    # Make directories to save videos if necessary
    save_dir = os.path.join(plane_dir, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get total number of components
    n_comps_total = res['F'].shape[0]

    # Get array of motion-corrected frames
    tif_dir = os.path.join(plane_dir, tif_dir_name)
    tif_fns = sorted(os.path.join(tif_dir, f) for f in os.listdir(tif_dir))
    arr = np.concatenate([np.array(io.imread(tif)) for tif in tif_fns])

    count = 0
    for j in range(int(np.ceil(n_comps_total / n_comps_per_slice))):
        # Select desired components
        comp_slice = [val for val in range(j * n_comps_per_slice, (
                j + 1) * n_comps_per_slice) if val < n_comps_total]

        # Extract traces and spatial footprints of components
        traces = res['F'][comp_slice, :]
        spat_fps = np.zeros((n_comps_per_slice,) + arr.shape[1:])
        for i, stat_one_comp in enumerate(res['stat'][comp_slice]):
            for x, y in zip(stat_one_comp['xpix'], stat_one_comp['ypix']):
                spat_fps[i, y, x] = 1

        # Plot traces and center maps
        n_rows = n_comps_per_slice // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        for i, trace in enumerate(traces):
            # Traces
            axes.flatten()[i].plot(trace)
            xy = spat_fps[i, :, :]
            axes.flatten()[i].set_title(
                'Comp {}; Center: {}'.format(
                    count, [int(val) for val in center_of_mass(xy)]))
            axes.flatten()[i].set_xlabel('')
            count += 1

        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'comps{}-{}'.format(
            comp_slice[0], comp_slice[-1])))
        plt.close()
    return


def main():
    # Get plane directories
    plane_dirs = get_plane_dirs()

    for plane_dir in plane_dirs:
        # Load all results
        res = load_results_one_plane(plane_dir)

        # Make movie for each components
        make_movie_each_comp_one_plane(res, plane_dir)

        # Make traces for each component
        make_traces_one_plane(res, plane_dir)
    return


if __name__ == '__main__':
    main()
