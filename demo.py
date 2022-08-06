# -*-encoding:utf-8-*-

import math
import os
import sys
import numpy as np
import configparser
from heightMeasurement import heightCalc

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python %s img_path config_fname" % sys.argv[0])
        exit(-1)

    img_path = sys.argv[1]  # the path of the image folder
    cfg_fname = sys.argv[2]  # the path of the config file

    # load configurations
    config = configparser.ConfigParser()
    config.read(cfg_fname)

    # initialize the intrinsic parameters
    cx = cy =320  # the position of the image center
    hvfov = float(config["STREET_VIEW"]["HVFoV"])  # the field of view
    fx = math.tan(np.deg2rad(hvfov/2.0)) * cx  # focal length
    fy = math.tan(np.deg2rad(hvfov/2.0)) * cy  # focal length
    intrins = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # intrinsic matrix

    # loop over the data
    for root, dir, files in os.walk(img_path):
        for file in files:
            if '.jpg' in file:
                img_fname = os.path.join(root, file)

                # the file name of the vanishing points
                vpt_fname = img_fname.replace('/imgs/', '/vpts/')
                vpt_fname = vpt_fname.replace('.jpg', 'vptpre.npz')
                if not os.path.exists(vpt_fname):
                    vpt_fname = 'none'

                # the file name of the detected line segments
                line_fname = img_fname.replace('/imgs/', '/lines/')
                line_fname = line_fname.replace('.jpg', 'nlines.npz')

                # the file name of the semantic segmentation data
                seg_fname = img_fname.replace('/imgs/', '/segs/')
                seg_fname = seg_fname.replace('.jpg', 'segre.npz')

                # the file name of ground truth (default: none)
                zgt_fname = 'none'
                if int(config["GROUND_TRUTH"]["Exist"]):
                    zgt_fname = vpt_fname.replace('/imgs/', '/zgts/')
                    zgt_fname = zgt_fname.replace('.jpg', 'height.npz')

                fname_dict = dict()
                fname_dict["vpt"] = vpt_fname
                fname_dict["img"] = img_fname
                fname_dict["line"] = line_fname
                fname_dict["seg"] = seg_fname
                fname_dict["zgt"] = zgt_fname

                # estimation of building height
                heightCalc(fname_dict, intrins, config, img_size=[640, 640], pitch=25, use_pitch_only=0, use_detected_vpt_only=0, verbose=True)

    # the end
    print('end')
