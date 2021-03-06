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
from natsort import natsorted

# Set constants
DATA_DIR = '/Users/benderas/NeuroPAL/Compiled/worm3_gcamp_Out_2p/suite2p'
MOVIE_LEN = 10  # in seconds
OVERWRITE_VIDS = True
DO_VIDEO_SORT = True
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
                                   save_dir_name='movies', movie_len=MOVIE_LEN,
                                   overwrite=OVERWRITE_VIDS):
    """Make movie for each component."""
    # Make directories to save videos if necessary
    save_dir = os.path.join(plane_dir, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Delete all videos currently in folder if overwrite desired
    if overwrite:
        files = os.listdir(save_dir)
        for f in files:
            os.remove(os.path.join(save_dir, f))

    # Select only ROIs consideered possible cells
    comp_mask = res['iscell'][:, 0] == 1
    res = {key: val[comp_mask] for key, val in res.items() if val.shape}

    # Get array of motion-corrected frames
    tif_dir = os.path.join(plane_dir, tif_dir_name)
    tif_fns = sorted(os.path.join(tif_dir, f) for f in os.listdir(tif_dir))
    arr = np.concatenate([np.array(io.imread(tif)) for tif in tif_fns])

    tqdm_desc = 'Plane {} Videos'.format(plane_dir.split('e')[-1])
    for i, (stat_one_comp, F_one_comp) in enumerate(tqdm(zip(
            res['stat'], res['F']), total=len(res['stat']), desc=tqdm_desc)):
        # Make filename for video
        save_name = 'comp{}.avi'.format(i)
        vid_fn = os.path.join(save_dir, save_name)
        made = np.sum([save_name in files for _, _, files in os.walk(
            save_dir)])

        if not made:
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
            fps = len(video_bound) // movie_len
            video = VideoWriter(vid_fn, VideoWriter_fourcc(*'MJPG'), fps,
                                video_bound[0].shape[:-1][::-1])
            for frame in video_bound:
                video.write((frame * 255).astype(np.uint8))
            video.release()
    return arr, save_dir


def make_traces_one_plane(res, plane_dir, tif_dir_name='reg_tif', n_cols=3,
                          n_comps_per_slice=12, save_dir_name='traces',
                          overwrite=OVERWRITE_VIDS):
    """Plot and save traces for each component in plane."""
    # Make directories to save videos if necessary
    save_dir = os.path.join(plane_dir, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Delete all videos currently in folder if overwrite desired
    if overwrite:
        files = os.listdir(save_dir)
        for f in files:
            os.remove(os.path.join(save_dir, f))

    # Select only ROIs consideered possible cells
    comp_mask = res['iscell'][:, 0] == 1
    res = {key: val[comp_mask] for key, val in res.items() if val.shape}

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

        # Plot traces
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


def sort_videos(movie_dir, folder_options=('good', 'bad', 'mc_prob')):
    """Sort videos by manual inspection of each component's trace and video."""
    # Get movies in folder
    movs = natsorted([f for f in os.listdir(movie_dir) if f.endswith('.avi')])

    # Make subdirectories
    for dir_name in folder_options:
        dir_path = os.path.join(movie_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Evaluate each component
    for comp in movs:
        opt = None
        while opt not in folder_options:
            opt = input('Classify {} as one of the following {}: '.format(
                            comp, folder_options))

        # Move to desired folder
        src_path = os.path.join(movie_dir, comp)
        dest_path = os.path.join(movie_dir, opt, comp)
        os.rename(src_path, dest_path)

    # Report number in each folder
    print('NUMBER IN EACH FOLDER')
    counts = {}
    for dir_name in folder_options:
        dir_path = os.path.join(movie_dir, dir_name)
        counts[dir_name] = len(os.listdir(dir_path))
        print('{}: {}'.format(dir_name, counts[dir_name]))
    return counts


def total_count(plane_counts):
    """Print total number of videos in each folder from all planes."""
    # Sum across planes
    tots = {k: sum([d[k] for d in plane_counts]) for k in plane_counts[0]}

    # Print number of each from all planes
    print('\nTOTAL NUMBER IN EACH FOLDER ACROSS PLANES')
    for key, val in tots.items():
        print('{}: {}'.format(key, val))
    return


def main(do_video_sort=DO_VIDEO_SORT):
    # Get plane directories
    plane_dirs = get_plane_dirs()

    # Initialize plane counts
    plane_counts = []
    for plane_dir in plane_dirs:
        # Load all results
        res = load_results_one_plane(plane_dir)

        # Make movie for each components
        _, movie_dir = make_movie_each_comp_one_plane(res, plane_dir)

        # Make traces for each component
        make_traces_one_plane(res, plane_dir)

        # Do video sort if desired
        if do_video_sort:
            plane_counts.append(sort_videos(movie_dir))

    # Report total number across planes from each folder
    if do_video_sort:
        total_count(plane_counts)
    return


if __name__ == '__main__':
    main()
