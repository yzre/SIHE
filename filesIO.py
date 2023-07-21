# -*-encoding:utf-8 -*-

import numpy as np
import skimage
import matplotlib.pyplot as plt

def load_vps_2d(filename):
    """
    load vanishing points
    :param filename:
    :return:
    """
    with np.load(filename) as npz:
        vpts_pd_2d = npz['vpts_re']
    return vpts_pd_2d


def load_line_array(filename):
    """
    load lines as well as scores.
    :param filename:
    :return:
    """
    with np.load(filename) as npz:
        nlines = npz["nlines"]
        nscores = npz["nscores"]

    return nlines, nscores


def load_seg_array(filename):
    """
    load segmentation results, the values represent different labels.
    :param filename:
    :return:
    """
    with np.load(filename) as npz:
        seg_array = npz["seg"]
    return seg_array


def load_zgts(filename):
    """
    load the ground truth image of z values, if exists.
    :param filename:
    :return:
    """
    with np.load(filename) as npz:
        zgt = npz["height"]
    return zgt