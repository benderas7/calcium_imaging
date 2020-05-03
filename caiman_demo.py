"""Largely based off of CaImAn's demo_pipeline.ipynb found here:
https://github.com/flatironinstitute/CaImAn/blob/master/demos/notebooks"""
# Import necessary modules
import bokeh.plotting as bpl
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, \
    nb_plot_contour

# Logging parameters
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

# Data and data display parameters
DATA_DIR = 'data/N.01.01'
DISP_MOVIE = True

# Dataset dependent parameters
FR = 30  # imaging rate in frames per second
DECAY_TIME = 0.4  # length of a typical transient in seconds

# Motion correction parameters
STRIDES = (48, 48)  # start a new patch for pw-rigid motion correction every
# x pixels
OVERLAPS = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
MAX_SHIFTS = (6, 6)  # maximum allowed rigid shifts (in pixels)
MAX_DEV_RIGID = 3  # maximum shifts deviation allowed for patch with
# respect to rigid shifts
PW_RIGID = True  # flag for performing non-rigid motion correction

# Parameters for source extraction and deconvolution
P = 1  # order of the autoregressive system
GNB = 2  # number of global background components
MERGE_THR = 0.85  # merging threshold, max correlation allowed
RF = 15  # half-size of the patches in pixels. e.g., if rf=25, patches are
# 50x50
STRIDE_CNMF = 6  # amount of overlap between the patches in pixels
K = 4  # number of components per patch
GSIG = [4, 4]  # expected half size of neurons in pixels
METHOD_INIT = 'greedy_roi'  # initialization method (if analyzing dendritic
# data using 'sparse_nmf')
SSUB = 1  # spatial subsampling during initialization
TSUB = 1  # temporal subsampling during intialization

# parameters for component evaluation
MIN_SNR = 2.0  # signal to noise ratio for accepting a component
RVAL_THR = 0.85  # space correlation threshold for accepting a component
CNN_THR = 0.99  # threshold for CNN based classifier
CNN_LOWEST = 0.1  # neurons with cnn probability lower than this value are
# rejected


######


def set_up_logger(fn=LOG_FN, level=LOG_LEVEL):
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


def set_opts(fnames, fr=FR, decay_time=DECAY_TIME, strides=STRIDES,
             overlaps=OVERLAPS, max_shifts=MAX_SHIFTS,
             max_dev_rigid=MAX_DEV_RIGID, pw_rigid=PW_RIGID, p=P, gnb=GNB,
             rf=RF, k=K, stride_cnmf=STRIDE_CNMF, method_init=METHOD_INIT,
             ssub=SSUB, tsub=TSUB, merge_thr=MERGE_THR, min_snr=MIN_SNR,
             rval_thr=RVAL_THR, cnn_thr=CNN_THR, cnn_lowest=CNN_LOWEST):
    """Parameters not defined in the dictionary will assume their default
    values. The resulting params object is a collection of subdictionaries
    pertaining to the dataset to be analyzed (params.data), motion correction
    (params.motion), data pre-processing (params.preprocess), initialization
    (params.init), patch processing (params.patch), spatial and temporal
    component (params.spatial), (params.temporal), quality evaluation
    (params.quality) and online processing (params.online)"""
    opts_dict = {
        'fnames': fnames, 'fr': fr, 'decay_time': decay_time,
        'strides': strides, 'overlaps': overlaps, 'max_shifts': max_shifts,
        'max_deviation_rigid': max_dev_rigid, 'pw_rigid': pw_rigid, 'p': p,
        'nb': gnb, 'rf': rf, 'K': k, 'stride': stride_cnmf,
        'method_init': method_init, 'rolling_sum': True, 'only_init': True,
        'ssub': ssub, 'tsub': tsub, 'merge_thr': merge_thr, 'min_SNR': min_snr,
        'rval_thr': rval_thr, 'use_cnn': True, 'min_cnn_thr': cnn_thr,
        'cnn_lowest': cnn_lowest}
    opts = params.CNMFParams(params_dict=opts_dict)
    return opts


def motion_corr():
    return


def mem_mapping():
    return


def run_cnmf():
    return


def run_pipeline():
    return


def inspect_results():
    return


def rerun_cnmf():
    return


def comp_eval():
    return


def extract_df_over_f():
    return


def sel_hq_comps():
    return


def disp_results():
    return


def save_results():
    return


def clean_log():
    """Remove all log files"""
    log_files = glob.glob('*_LOG_*')
    for log in log_files:
        os.remove(log)
    return


def view_results_movie():
    return


def main(log=LOG, data_dir=DATA_DIR, disp_movie=DISP_MOVIE):
    # Set up logger if desired
    if log:
        set_up_logger()

    # Get files
    fnames = ['{}/{}'.format(data_dir, f) for f in os.listdir(data_dir)]

    # Display movie if wanted
    if disp_movie:
        play_movie(fnames)

    # Set options for extraction
    opts = set_opts(fnames)

    # Clean up logger if necessary
    if log:
        clean_log()
    return


if __name__ == '__main__':
    main()
