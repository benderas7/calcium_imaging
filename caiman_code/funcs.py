"""Largely based off of CaImAn's demo_pipeline.ipynb found here:
https://github.com/flatironinstitute/CaImAn/blob/master/demos/notebooks"""
# Import necessary modules
import glob
import logging
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params


def set_up_logger(fn, level):
    """You can log to a file using the fn parameter, or make the output
    more or less verbose by setting level to logging.DEBUG, logging.INFO,
    logging.WARNING, or logging.ERROR. A fn argument can also be passed
    to store the log file"""
    logging.basicConfig(filename=fn, level=level)
    return


def play_movie(fnames, ds_ratio=0.2, q_max=99.5, fr=30, mag=2):
    """Play the movie. This will require loading the movie in memory which
    in general is not needed by the pipeline. Displaying the movie uses the
    OpenCV library. Press q to close the video panel."""
    m_orig = cm.load_movie_chain(fnames)
    m_orig.resize(1, 1, ds_ratio).play(q_max=q_max, fr=fr, magnification=mag)
    return


def set_opts(fnames, fr, decay_time, opts_dict):
    """Parameters not defined in the dictionary will assume their default
    values. The resulting params object is a collection of subdictionaries
    pertaining to the dataset to be analyzed (params.data), motion correction
    (params.motion), data pre-processing (params.preprocess), initialization
    (params.init), patch processing (params.patch), spatial and temporal
    component (params.spatial), (params.temporal), quality evaluation
    (params.quality) and online processing (params.online)"""
    opts_dict = {**opts_dict, 'fnames': fnames, 'fr': fr,
                 'decay_time': decay_time}
    opts = params.CNMFParams(params_dict=opts_dict)
    return opts


def set_up_local_cluster(backend='local', n_processes=None,
                         single_thread=False):
    """The variable backend determines the type of cluster used. The default
    value 'local' uses the multiprocessing package. The ipyparallel option is
    also available. The resulting variable dview expresses the cluster
    option. If you use dview=dview in the downstream analysis then parallel
    processing will be used. If you use dview=None then no parallel processing
    will be employed."""
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend=backend, n_processes=n_processes, single_thread=single_thread)
    return c, dview, n_processes


def motion_corr(fnames, dview, opts, disp_movie, is_3d=False):
    """Perform motion correction"""
    # Create a motion correction object with the parameters specified. Note
    # that the file is not loaded in memory
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))

    # Run piecewise-rigid motion correction using NoRMCorre
    mc.motion_correct(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)

    # Determine maximum shift to be used for trimming against NaNs
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0

    # Compare with original movie
    if disp_movie and not is_3d:
        m_orig = cm.load_movie_chain(fnames)
        ds_ratio = 0.2
        cm.concatenate(
            [m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
             m_els.resize(1, 1, ds_ratio)], axis=2).play(
            fr=60, gain=15, magnification=2, offset=0)  # press q to exit
    return mc, border_to_0


def mem_mapping(mc, border_to_0, dview, is_3d=False):
    """memory maps the file in order 'C' and then loads the new memory mapped
    file. The saved files from motion correction are memory mapped files stored
    in 'F' order. Their paths are stored in mc.mmap_file."""
    # Memory map the file in order 'C', excluding borders
    fname_new = cm.save_memmap(
        mc.mmap_file, base_name='memmap_', order='C',
        border_to_0=border_to_0, is_3D=is_3d, dview=dview)

    # Load the file with framees in python format (T x X x Y)
    yr, dims, t = cm.load_memmap(fname_new)
    images = np.reshape(yr.T, [t] + list(dims), order='F')
    return images


def run_cnmf(n_processes, opts, dview, images):
    """The FOV is split is different overlapping patches that are subsequently
    processed in parallel by the CNMF algorithm. The results from all the
    patches are merged with special attention to idendtified components on the
    border. The results are then refined by additional CNMF iterations."""
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0). If you want to have
    # deconvolution within each patch change params.patch['p_patch'] to a
    # nonzero value
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    return cnm


def run_pipeline(n_processes, opts, dview, do_mc=True):
    """Run the combined steps of motion correction, memory mapping, and cnmf
    fitting in one step."""
    cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm1.fit_file(motion_correct=do_mc)
    return


def inspect_results(images, cnm):
    """Inspect results by plotting contours of identified compoenntts
    against correlation image. The results of the algorithm are stored in
    the object cnm.estimates."""
    # Plot contours of found components
    cn = cm.local_correlations(images.transpose(1, 2, 0))
    cn[np.isnan(cn)] = 0
    cnm.estimates.plot_contours_nb(img=cn)
    return cnm, cn


def rerun_cnmf(cnm, images, dview):
    """Re-run CNMF algorithm seeded on just the selected components from the
    previous step. Components rejected on the previous  step will not be
    recovered here, so be careful."""
    # Re-run CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)
    return cnm2


def comp_eval(cnm, images, dview, cn, is_3d=False):
    """The processing in patches creates several spurious components. These are
    filtered out by evaluating each component using three different criteria:
    (1) the shape of each component must be correlated with the data at the
    corresponding location within the FOV; (2) a minimum peak SNR is required
    over the length of a transient; (3) each shape passes a CNN based
    classifier."""
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    # Plot contours of selected and rejected components
    cnm.estimates.plot_contours_nb(img=cn, idx=cnm.estimates.idx_components)

    # View traces of accepted and rejected components
    if is_3d:
        cnm.estimates.nb_view_components_3d(
            img=cn, idx=cnm.estimates.idx_components)
    else:
        cnm.estimates.nb_view_components(
            img=cn, idx=cnm.estimates.idx_components)
    if len(cnm.estimates.idx_components_bad) > 0:
        if is_3d:
            cnm.estimates.nb_view_components_3d(
                img=cn, idx=cnm.estimates.idx_components_bad)
        else:
            cnm.estimates.nb_view_components(
                img=cn, idx=cnm.estimates.idx_components_bad)
    else:
        print('No components were rejected.')
    return


def extract_df_over_f(cnm, quantile_min=8, frames_window=250):
    cnm.estimates.detrend_df_f(
        quantileMin=quantile_min, frames_window=frames_window)
    return


def sel_hq_comps(cnm):
    """Select only high quality components."""
    cnm.estimates.select_components(use_object=True)
    return


def disp_results(cnm, cn, color='red', is_3d=False):
    """Display final results."""
    if is_3d:
        cnm.estimates.nb_view_components_3d(img=cn, denoised_color=color)
        return
    cnm.estimates.nb_view_components(img=cn, denoised_color=color)
    return


def save_results(cnm, save_dir):
    cnm.save(os.path.join(save_dir, 'analysis_results.hdf5'))
    return


def clean_log():
    """Remove all log files"""
    log_files = glob.glob('*_LOG_*')
    for log in log_files:
        os.remove(log)
    return


def view_results_movie(cnm, images, border_to_0):
    cnm.estimates.play_movie(images, q_max=99.9, gain_res=2, magnification=2,
                             bpx=border_to_0, include_bck=False)
    return


def pipeline(video_fn, log, log_fn, log_level, fr, decay_time, opts_dict,
             save_results_dir, disp_movie=True, is_3d=False):
    # Set up logger if desired
    if log:
        set_up_logger(log_fn, log_level)

    # Get video for processing
    fnames = [video_fn]

    # Display movie if wanted
    if disp_movie and not is_3d:
        play_movie(fnames)

    # Set options for extraction
    opts = set_opts(fnames, fr, decay_time, opts_dict)

    # Configure local cluster
    c, dview, n_processes = set_up_local_cluster()

    # Perform motion correction
    mc, border_to_0 = motion_corr(fnames, dview, opts, disp_movie, is_3d=is_3d)

    # Perform memory mapping
    images = mem_mapping(mc, border_to_0, dview, is_3d=is_3d)

    # Restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = set_up_local_cluster()

    # Run CNMF on patches in parallel
    cnm = run_cnmf(n_processes, opts, dview, images)

    # Inspect results
    cnm, cn = inspect_results(images, cnm)

    # Re-run CNMF on full FOV
    cnm2 = rerun_cnmf(cnm, images, dview)

    # Evaluate components
    comp_eval(cnm2, images, dview, cn, is_3d=is_3d)

    # Extract dF/F
    extract_df_over_f(cnm2)

    # Select only high quality components
    sel_hq_comps(cnm2)

    # Display final results
    disp_results(cnm2, cn, is_3d=is_3d)

    # Save results if specified
    if save_results_dir:
        save_results(cnm2, save_results_dir)

    # Stop cluster
    cm.stop_server(dview=dview)

    # Clean up logger if necessary
    if log:
        clean_log()

    # View results movie if wanted
    if disp_movie and not is_3d:
        view_results_movie(cnm2, images, border_to_0)
    return
