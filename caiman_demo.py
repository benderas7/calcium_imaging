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
from caiman.utils.visualization import plot_contours, nb_view_patches,\
    nb_plot_contour

# Set up constants - logging
LOG = True
LOG_FN = '/tmp/caiman.log'
LOG_LEVEL = logging.WARNING

DATA_DIR = 'data'
######


def set_up_logger(fn=LOG_FN, level=LOG_LEVEL):
    """You can log to a file using the fn parameter, or make the output
    more or less verbose by setting level to logging.DEBUG, logging.INFO,
    logging.WARNING, or logging.ERROR. A fn argument can also be passed
    to store the log file"""
    logging.basicConfig(filename=fn, level=level)
    return


def play_movie():
    return


def setup_params():
    return


def create_params_obj():
    return


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


def main(log=LOG):
    # Set up logger if desired
    if log:
        set_up_logger()

    # Clean up logger if necessary
    if log:
        clean_log()
    return


if __name__ == '__main__':
    main()
