# -*-encoding:utf-8-*-

import copy
import numpy as np
import itertools
import matplotlib.pyplot as plt
import skimage
import os
from collections import Counter
from lineDrawingConfig import *
from lineRefinement import *
from sklearn.cluster import DBSCAN


def classifyWithVPTs(n1, n2, vpt, config):
    """
    Use the vanishing point to classify the line segments.
    :param n1: end point of the line segment
    :param n2: end point of the line segment
    :param vpt: the vanishing point
    :param config: configuration
    :return:
    """

    flag = False
    t_angle = float(config["LINE_CLASSIFY"]["AngleThres"])  # the threshold of the anlge error (degree)

    # for points n1 and n2, the elements are arranged as [row, column]=[y, x], but for vpt, it's [column, row]=[x, y]
    # therefore, the following line is needed
    p1 = np.array([n1[1], n1[0]])
    p2 = np.array([n2[1], n2[0]])

    # # compare the direction of the line with the directions of the two lines p1-vpt, p2-vpt
    # d1 = p2 - p1
    # d2 = vpt - p1
    # angle1 = np.rad2deg(np.arccos(np.dot(d1, d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))))
    # d3 = p1 - p2
    # d4 = vpt - p2
    # angle2 = np.rad2deg(np.arccos(np.dot(d3, d4)/(np.linalg.norm(d3)*np.linalg.norm(d4))))
    # if (angle1 < t_angle or 180 - angle1 < t_angle) or (angle2 < t_angle or 180 - angle2 < t_angle):
    #     is_vertical = 1

    # compare the direction of the line with the direction of the middle point of the line to vpt
    mpt = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]  # the middle point of the line
    d1 = p2 - p1
    d2 = vpt - mpt
    angle = np.rad2deg(np.arccos(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))))
    if angle < t_angle or 180 - angle < t_angle:
        flag = True

    return flag


def check_if_line_lies_in_building_area(seg_img, a, b, config)->bool:
    """
    check if the line segment lies in building area
    :param seg_img: the semantic segmentation image array
    :param a: end point of the line segment
    :param b: end point of the line segment
    :param config: configuration
    :return:
    """

    middle = (a + b)/2.0  # middle point of the line segment
    norm_direction = (a - b) / np.linalg.norm(a - b)
    ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

    sky_label = int(config["SEGMENTATION"]["SkyLabel"])
    building_label = int(config["SEGMENTATION"]["BuildingLabel"])
    ground_label = np.cast["int"](config["SEGMENTATION"]["GroundLabel"].split(','))

    ratio = 10
    ppd_dir = ratio * ppd_dir
    point_check_list = copy.deepcopy(a)
    point_check_list = np.vstack([point_check_list, a - ppd_dir])
    point_check_list = np.vstack([point_check_list, a + ppd_dir])
    point_check_list = np.vstack([point_check_list, b])
    point_check_list = np.vstack([point_check_list, b - ppd_dir])
    point_check_list = np.vstack([point_check_list, b + ppd_dir])
    point_check_list = np.vstack([point_check_list, middle])
    point_check_list = np.vstack([point_check_list, middle - ppd_dir])
    point_check_list = np.vstack([point_check_list, middle + ppd_dir])

    total_num = 0
    local_num = 0
    rows, cols = seg_img.shape
    flag = True
    for pcl in point_check_list:
        total_num = total_num + 1
        # swap the x,y coordinate
        y_int = np.cast["int"](pcl[0] + 0.5)
        x_int = np.cast["int"](pcl[1] + 0.5)
        if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
            local_num = local_num + 1
            continue
        if seg_img[y_int, x_int] == building_label:
            local_num = local_num + 1
        if np.remainder(total_num, 3) == 0 and local_num == 0:
            flag = False
            break
        else:
            if np.remainder(total_num, 3) == 0:
                local_num = 0
    return flag


def check_if_bottom_lines(seg_img, a, b, config)->bool:
    """
    Check whether a line is on the bottom of a building.
    :param seg_img: semantic segmentation image array
    :param a: end point of the line
    :param b: end point of the line
    :param config: configuration
    :return:
    """

    middle = (a + b)/2.0  # middle point of the line
    norm_direction = (a - b) / np.linalg.norm(a - b)
    ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

    ground_label = np.cast["int"](config["SEGMENTATION"]["GroundLabel"].split(','))

    ratio = 10
    ppd_dir = ratio * ppd_dir
    point_check_list = copy.deepcopy(a)
    point_check_list = np.vstack([point_check_list, a - ppd_dir])
    point_check_list = np.vstack([point_check_list, a + ppd_dir])
    point_check_list = np.vstack([point_check_list, b])
    point_check_list = np.vstack([point_check_list, b - ppd_dir])
    point_check_list = np.vstack([point_check_list, b + ppd_dir])
    point_check_list = np.vstack([point_check_list, middle])
    point_check_list = np.vstack([point_check_list, middle - ppd_dir])
    point_check_list = np.vstack([point_check_list, middle + ppd_dir])
    rows, cols = seg_img.shape

    flag = False
    # count = 0
    for pcl in point_check_list:
        # swap the x,y coordinate
        y_int = np.cast["int"](pcl[0] + 0.5)
        x_int = np.cast["int"](pcl[1] + 0.5)
        if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
            # count = count + 1
            continue
        if seg_img[y_int, x_int] in ground_label:
            flag = True
            break
            # count = count + 1
            # continue
    # if count < 3:
    #     flag = False

    return flag


def check_if_roof_lines(seg_img, a, b, config)->bool:
    """
    Check whether a line is on the roof of a building.
    :param seg_img: semantic segmentation image array
    :param a: end point of the line
    :param b: end point of the line
    :param config: configuration
    :return:
    """

    middle = (a + b)/2.0  # middle point of the line
    norm_direction = (a - b) / np.linalg.norm(a - b)
    ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

    sky_label = int(config["SEGMENTATION"]["SkyLabel"])

    ratio = 10
    ppd_dir = ratio * ppd_dir
    point_check_list = copy.deepcopy(a)
    point_check_list = np.vstack([point_check_list, a - ppd_dir])
    point_check_list = np.vstack([point_check_list, a + ppd_dir])
    point_check_list = np.vstack([point_check_list, b])
    point_check_list = np.vstack([point_check_list, b - ppd_dir])
    point_check_list = np.vstack([point_check_list, b + ppd_dir])
    point_check_list = np.vstack([point_check_list, middle])
    point_check_list = np.vstack([point_check_list, middle - ppd_dir])
    point_check_list = np.vstack([point_check_list, middle + ppd_dir])

    rows, cols = seg_img.shape

    flag = False
    # count = 0
    for pcl in point_check_list:
        # swap the x,y coordinate
        y_int = np.cast["int"](pcl[0] + 0.5)
        x_int = np.cast["int"](pcl[1] + 0.5)
        if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
            # count = count + 1
            continue
        if seg_img[y_int, x_int] == sky_label:
            flag = True
            break
            # count = count + 1
            # continue
    # if count < 3:
    #     flag = False

    return flag


def dist_comparaison(first_line, second_line, thres):
    """
    Compare the distance of two lines
    :param first_line:
    :param second_line:
    :param thres: distance threshold
    :return:
    """
    a_0 = copy.deepcopy(first_line[0])
    b_0 = copy.deepcopy(first_line[1])
    a_1 = copy.deepcopy(second_line[0])
    b_1 = copy.deepcopy(second_line[1])

    pt_0 = pointOnLine(a_0, b_0, (a_1+b_1)/2.0)
    dist_0 = np.linalg.norm(pt_0 - (a_1+b_1)/2.0)

    pt_1 = pointOnLine(a_1, b_1, (a_0 + b_0) / 2.0)
    dist_1 = np.linalg.norm(pt_1 - (a_0 + b_0) / 2.0)

    if dist_0 < thres or dist_1 < thres:
        a_1_refine = pointOnLine(a_0, b_0, a_1)
        b_1_refine = pointOnLine(a_0, b_0, b_1)

        if a_0[0] > a_1_refine[0]:
            a_0 = a_1_refine
        if b_0[0] < b_1_refine[0]:
            b_0 = b_1_refine

        return True, [a_0, b_0]

    return False, first_line


def lineCoeff(p1, p2):
    """
    Get the coefficients of a line from its two points
    :param p1: a point on the line
    :param p2: a point on the line
    :return: three coefficients
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def intersection(L1, L2):
    """
    Get the intersection point of two lines
    :param L1: a line
    :param L2: a line
    :return: the intersection point
    """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

# def fittingVanshingPoints(lines):
#     dist = []
#     for line in lines:
#         a = line[0]
#         b = line[1]
#         dist.append(np.linalg.norm(np.asarray([a-b])))
#     ind = np.argsort(np.asarray(dist))
#     max_num = np.min([5, len(ind)])
#     A = []
#     b = []
#     for i in np.arange(-1, -max_num, -1):
#         if dist[i] < 10:
#             continue
#         l = lineCoeff(lines[i][0], lines[i][1])
#         A.append([l[0], l[1]])
#         b.append(l[2])
#     A=np.asarray(A)
#     b=np.asarray(b)
#     # vpt = np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),\
#     #                 np.matmul(np.transpose(A),b))
#     vpt = np.linalg.lstsq(A, b)[0].T
#     # vpt_verify = intersection([A[0][0],A[0][1], -b[0]], [A[1][0],A[1][1], -b[1]])
#     return vpt


def filter_lines_outof_building_ade20k(imgfile, lines, line_scores, segimg, vpts, config, use_vertical_vpt_only=0, verbose=True):
    """
    Use the semantic segmentation results to filter the line segments and classify them into different groups.
    :param imgfile: the file name of the image
    :param lines: the line segments
    :param line_scores: the scores of the corresponding line segments
    :param segimg: the semantic segmentation image array
    :param vpts: the vanishing points
    :param config: configuration
    :param use_vertical_vpt_only: use only the vertical vanishing point to process the line segments
    :param verbose: when true, show the results
    :return:
    """

    # initialize
    vert_lines = []  # vertical line segments
    hori0_lines = []  # horizontal line segments related to the first vanishing point
    hori1_lines = []  # horizontal line segments related to the second vanishing point

    # ######### filter the line segments out of buildings
    t_score = float(config["LINE_CLASSIFY"]["LineScore"])  # the threshold for the scores of lines
    for (a, b), s in zip(lines, line_scores):
        # only process the lines with scores over the threshold
        if s < t_score:
            continue
        is_in_building = check_if_line_lies_in_building_area(segimg, a, b, config)
        if not is_in_building:
            continue

        # classify line segments into different groups using their directions
        if use_vertical_vpt_only:
            is_vert = classifyWithVPTs(a, b, vpts[2], config)
            if is_vert:
                vert_lines.append([a, b])
        else:
            is_hori0 = classifyWithVPTs(a, b, vpts[0], config)
            if is_hori0:
                hori0_lines.append([a, b])
            is_hori1 = classifyWithVPTs(a, b, vpts[1], config)
            if (not is_hori0) and is_hori1:
                hori1_lines.append([a, b])
            is_vert = classifyWithVPTs(a, b, vpts[2], config)
            if (not is_hori0) and (not is_hori1) and is_vert:
                vert_lines.append([a, b])

        # if verbose:
        #     plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
        #     plt.scatter(a[1], a[0], **PLTOPTS)
        #     plt.scatter(b[1], b[0], **PLTOPTS)

    # ######### refine and merge short vertical lines to long lines
    vert_line_refine = []
    vert_line_merge = []
    vert_lines_copy = copy.deepcopy(vert_lines)

    # use vanishing point to refine the vertical lines
    for line in vert_lines_copy:
        a = line[0]
        b = line[1]
        line = lineRefinementWithVPT([a, b], np.asarray([vpts[2, 1], vpts[2, 0]]))
        vert_line_refine.append(line)

    # merge short lines
    for i in range(len(vert_line_refine)):
        linesi = vert_line_refine[i]
        lens = np.linalg.norm(linesi[0] - linesi[1])
        if linesi[0][0] < 0 and linesi[1][0] < 0 or lens < 10:
            continue
        for j in range(i+1, len(vert_line_refine)):
            # compare the distance between the two lines
            linesj = vert_line_refine[j]
            if linesj[0][0] < 0 and linesj[1][0] < 0:
                continue
            is_merging, vert_line_refine[i] = dist_comparaison(vert_line_refine[i], vert_line_refine[j], 5)
            if is_merging:
               vert_line_refine[j] = [np.asarray([-1, -1]), np.asarray([-1, -1])]

    for line in vert_line_refine:
        a = line[0]
        b = line[1]
        if a[1] < 0:
            continue
        vert_line_merge.append(line)
        if verbose:
            plt.plot([a[1], b[1]], [a[0], b[0]], c='b', linewidth=2)
            plt.scatter(a[1], a[0], **PLTOPTS)
            plt.scatter(b[1], b[0], **PLTOPTS)


    # # ######### classify horizontal lines   ## this may be used in future work
    # bottom_lines = []  # bottom lines of buildings
    # roof_lines =[]  # roof lines of buildings
    #
    # if not use_vertical_vpt_only:
    #     for line in hori0_lines:
    #         a = line[0]
    #         b = line[1]
    #         flag = check_if_roof_lines(segimg, line[0], line[1], config)
    #         if flag:
    #             if verbose:
    #                 plt.plot([a[1], b[1]], [a[0], b[0]], c='r', linewidth=2)
    #                 plt.scatter(a[1], a[0], **PLTOPTS)
    #                 plt.scatter(b[1], b[0], **PLTOPTS)
    #             roof_lines.append(line)
    #             continue
    #         flag = check_if_bottom_lines(segimg, line[0], line[1], config)
    #         if flag:
    #             if verbose:
    #                 plt.plot([a[1], b[1]], [a[0], b[0]], c='g', linewidth=2)
    #                 plt.scatter(a[1], a[0], **PLTOPTS)
    #                 plt.scatter(b[1], b[0], **PLTOPTS)
    #             bottom_lines.append(line)
    #             continue
    #
    #     for line in hori1_lines:
    #         a = line[0]
    #         b = line[1]
    #         flag = check_if_roof_lines(segimg, line[0], line[1], config)
    #         if flag:
    #             if verbose:
    #                 plt.plot([a[1], b[1]], [a[0], b[0]], c='r', linewidth=2)
    #                 plt.scatter(a[1], a[0], **PLTOPTS)
    #                 plt.scatter(b[1], b[0], **PLTOPTS)
    #             roof_lines.append(line)
    #             continue
    #         flag = check_if_bottom_lines(segimg, line[0], line[1], config)
    #         if flag:
    #             if verbose:
    #                 plt.plot([a[1], b[1]], [a[0], b[0]], c='g', linewidth=2)
    #                 plt.scatter(a[1], a[0], **PLTOPTS)
    #                 plt.scatter(b[1], b[0], **PLTOPTS)
    #             bottom_lines.append(line)
    #             continue

    if verbose:
        # plt.show()
        plt.close()

    # return vert_line_merge, bottom_lines, roof_lines
    return vert_line_merge


def clausterLinesWithCenters(ht_set, config, using_height=False):
    """
    Use DBSAN algorithm to divide the line segments and their heights into groups.
    :param ht_set: the list of heights
    :param config: configuration
    :param using_height: if value is '1', use the heights in the grouping
    :return:
    """

    X = []
    if using_height:
        for ht, a, b, *_ in ht_set:
            X.append([(a[0] + b[0])/2, (a[1] + b[1])/2, ht])
    else:
        for ht, a, b, *_ in ht_set:
            X.append([(a[0] + b[0])/2, (a[1] + b[1])/2])
    X = np.asarray(X)

    max_DBSAN_dist = float(config["HEIGHT_MEAS"]["MaxDBSANDist"])

    try:
        clustering = DBSCAN(eps=max_DBSAN_dist, min_samples=1).fit(X)
    except:
        print("!!! error in clustering: Expected 2D array, got 1D array instead.    Return no results")
        clustered_lines = None
        return clustered_lines

    clustered_lines = []
    max_val = np.max(clustering.labels_)+1
    for label in range(max_val):
        new_list = []
        new_ht_list = []
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == label:
                new_list.append(ht_set[i])
                new_ht_list.append(ht_set[i][0])
        medi_val = np.median(np.asarray(new_ht_list))  # the median height of the group
        mean_val = np.mean(np.asarray(new_ht_list))  # the mean height of the group
        new_list.append(medi_val)
        new_list.append(mean_val)
        clustered_lines.append(new_list)

    return clustered_lines
