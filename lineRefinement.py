# -*-encoding:utf-8-*-

import matplotlib.pyplot as plt
from lineDrawingConfig import *
import skimage
import numpy as np


def extendLines(pt1, pt2, segmt, config):
    """
    Extend the vertical line segments by referring to the segmentation image
    :param pt1: end point of the line
    :param pt2: end point of the line
    :param segmt: semantic segmentation image array
    :return:
    """

    sky_label = int(config["SEGMENTATION"]["SkyLabel"])
    building_label = int(config["SEGMENTATION"]["BuildingLabel"])
    ground_label = np.cast["int"](config["SEGMENTATION"]["GroundLabel"].split(','))
    edge_thres = np.cast["int"](config["LINE_REFINE"]["Edge_Thres"].split(','))

    if pt1[0] > pt2[0]:  # 0 is the x axis
        pt_up = pt2
        pt_down = pt1
    else:
        pt_up = pt1
        pt_down = pt2

    if np.linalg.norm(pt_down - pt_up) == 0:
        return [], []
    direction = (pt_down - pt_up) / np.linalg.norm(pt_down - pt_up)
    pt_up_end = pt_up
    pt_down_end = pt_down
    pt_middle = (pt_up + pt_down) / 2.0

    rows, cols = segmt.shape
    if pt_up_end[0] > rows - 2:
        pt_up_end[0] = rows - 2
    if pt_up_end[1] > cols - 2:
        pt_up_end[1] = cols - 2
    if pt_down_end[0] > rows - 2:
        pt_down_end[0] = rows - 2
    if pt_down_end[1] > cols - 2:
        pt_down_end[1] = cols - 2

    if pt_middle[0] >= rows - 1 or pt_middle[1] >= cols - 1:
        return [], []

    if segmt[np.cast['int'](pt_up_end[0] + 0.5)][np.cast['int'](pt_up_end[1] + 0.5)] != building_label or \
       segmt[np.cast['int'](pt_down_end[0] + 0.5)][np.cast['int'](pt_down_end[1] + 0.5)] != building_label or \
       segmt[np.cast['int'](pt_middle[0] + 0.5)][np.cast['int'](pt_middle[1] + 0.5)] != building_label:
        return [], []

    flag = 1
    while flag:
        pt_up_end = pt_up_end - direction
        if pt_up_end[0] < 0 or pt_up_end[1] < 0 or pt_up_end[1] >= rows - 1:
            flag = 0
            pt_up_end = pt_up_end + direction
            continue
        if segmt[np.cast['int'](pt_up_end[0] + 0.5)][np.cast['int'](pt_up_end[1] + 0.5)] == sky_label:
            flag = 0
            pt_up_end = pt_up_end + direction

    flag = 1
    out_of_building = False
    while flag:
        pt_down_end = pt_down_end + direction
        if pt_down_end[0] >= cols - 1 or pt_down_end[1] < 0 or pt_down_end[1] >= rows - 1:
            flag = 0
            continue
        if segmt[np.cast['int'](pt_down_end[0] + 0.5)][np.cast['int'](pt_down_end[1] + 0.5)] != building_label and \
           not segmt[np.cast['int'](pt_down_end[0] + 0.5)][np.cast['int'](pt_down_end[1] + 0.5)] in ground_label:
            out_of_building = True
        else:
            if segmt[np.cast['int'](pt_down_end[0] + 0.5)][np.cast['int'](pt_down_end[1] + 0.5)] == building_label:
                out_of_building = False
        if segmt[np.cast['int'](pt_down_end[0] + 0.5)][np.cast['int'](pt_down_end[1] + 0.5)] in ground_label and \
                not out_of_building:
            flag = 0
            pt_down_end = pt_down_end - direction
        else: # reach the ground and the previous label is not building
            if segmt[np.cast['int'](pt_down_end[0] + 0.5)][np.cast['int'](pt_down_end[1] + 0.5)] in ground_label:
                return [], []
            pass
    if  pt_up_end[0] > cols - 1 - edge_thres or \
        pt_up_end[1] < edge_thres or pt_up_end[1] > rows - edge_thres or \
        pt_down_end[0] < edge_thres or pt_down_end[0] > cols - 1 - edge_thres or \
        pt_down_end[1] < edge_thres or pt_down_end[1] > rows - 1 - edge_thres:
        return [],[]
    return pt_up_end, pt_down_end


def verticalLineExtending(img_name, vertical_lines, segimg, vptz, config, verbose=True):
    """
    Extend the vertical line segments to make their end points on the bottom and the roof of the buildings.
    :param img_name: the file name of the image
    :param vertical_lines: vertical line segments
    :param segimg: semantic segmentation image array
    :param vptz: vertical vanishing point
    :param config: configuration
    :param verbose: when true, show the results
    :return:
    """
    if verbose:
        plt.close()
        org_img = skimage.io.imread(img_name)
        plt.imshow(org_img)

    extd_lines = []
    for line in vertical_lines:
        # when reach the part of sky, stop extending
        # when go down until reach the ground
        line = lineRefinementWithVPT(line, vptz)
        a = line[0]
        b = line[1]
        extd_a, extd_b = extendLines(a, b, segimg, config)
        if len(extd_a) == 0 or len(extd_b) == 0:
            continue
        extd_lines.append([extd_a, extd_b])

        if verbose:
            plt.plot([extd_a[1], extd_b[1]], [extd_a[0], extd_b[0]], c='y', linewidth=2)
            plt.scatter(extd_a[1], extd_a[0], **PLTOPTS)
            plt.scatter(extd_b[1], extd_b[0], **PLTOPTS)

    if verbose:
        # plt.show()
        plt.close()

    return extd_lines


def verticalLineExtendingWithBRLines(img_name, vertical_lines, roof_lines, bottom_lines, segimg, config, verbose=True):
    """
    Extend the vertical line segments using the classified roof and bottom lines
    :param img_name: the file name of the image
    :param vertical_lines: vertical line segments
    :param roof_lines: roof line segments
    :param bottom_lines: bottom line segments
    :param segimg: semantic segmentation image array
    :param config: configuration
    :param verbose: when true, show the results
    :return:
    """

    if verbose:
        org_img = skimage.io.imread(img_name)
        plt.close()
        plt.imshow(org_img)

    rows, cols = segimg.shape
    extd_lines = []
    for vl in vertical_lines:
        pt_rl = []
        for rl in roof_lines:
            # roof lines
            vl_direction = vl[0] - vl[1]
            rl_direction = rl[0] - rl[1]
            A = np.transpose(np.vstack([vl_direction, -rl_direction]))
            b = np.transpose(rl[0] - vl[0])
            x = np.matmul(np.linalg.inv(A), b)
            pt = vl_direction*x[0] + vl[0]

            pt = np.cast['int'](pt + 0.5)

            if x[0] > 2 or x[0] < -2:
                continue

            if pt[0] < 10 or pt[0] > rows - 10 or pt[1] < 10 or pt[1] > cols - 10:
                continue

            # print(segimg[pt[0] - 5: pt[0] + 5, pt[1] - 5:pt[1] + 5])
            if np.std(segimg[pt[0] - 10: pt[0] + 10, pt[1] - 10:pt[1] + 10]) == 0:
                continue

            if len(pt_rl) == 0:
                pt_rl = pt
            else:
                if pt_rl[0] > pt[0]:
                    pt_rl = pt

        if len(pt_rl) == 0:
        #     vl[0] = pt_rl
        # else:
            continue

        pt_bl = []
        for bl in bottom_lines:
            # bottom lines
            vl_direction = vl[0] - vl[1]
            bl_direction = bl[0] - bl[1]
            A = np.transpose(np.vstack([vl_direction, -bl_direction]))
            b = np.transpose(bl[0] - vl[0])
            x = np.matmul(np.linalg.inv(A), b)
            pt = vl_direction*x[0] + vl[0]

            pt = np.cast['int'](pt + 0.5)

            if x[0] > 2 or x[0] < -2:
                continue

            if pt[0] < 10 or pt[0] > rows - 10 or pt[1] < 10 or pt[1] > cols - 10:
                continue

            if np.std(segimg[pt[0] - 10: pt[0] + 10, pt[1] - 10:pt[1] + 10]) == 0:
                continue

            if len(pt_bl) == 0:
                pt_bl = pt
            else:
                if pt_bl[0] < pt[0]:
                    pt_bl = pt

        if len(pt_bl) == 0:
        #     vl[1] = pt_bl
        # else:
            continue

        # if verbose:
        #     plt.figure()
        #     plt.imshow(org_img)
        #     plt.plot([vl[0][1], vl[1][1]], [vl[0][0], vl[1][0]], c='r', linewidth=2)
        #     # plt.plot([rl[0][1], rl[1][1]], [rl[0][0], rl[1][0]], c='r', linewidth=2)
        #     plt.scatter(pt_rl[1], pt_rl[0], **PLTOPTS)
        #     plt.scatter(pt_bl[1], pt_bl[0], **PLTOPTS)
        #     plt.show()

        extd_lines.append([pt_rl, pt_bl])

    return extd_lines


def pointOnLine(a, b, p):
    """
    Project a point (p) onto the line (a-b)
    :param a: end point of the line
    :param b: end point of the line
    :param p: a point
    :return:
    """
    # ap = p - a
    # ab = b - a
    # result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    # return result
    l2 = np.sum((a - b) ** 2)
    if l2 == 0:
        print('p1 and p2 are the same points')
    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2
    # if you need the point to project on line extention connecting p1 and p2
    t = np.sum((p - a) * (b - a)) / l2
    # if you need to ignore if p3 does not project onto line segment
    # if t > 1 or t < 0:
    #     print('p3 does not project onto p1-p2 line segment')
    #
    # # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    # t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))
    projection = a + t * (b - a)
    return projection


def lineRefinementWithVPT(line, vpt):
    """
    Use vanishing point to refine line segments. Slightly rotate the line around its middle point to make its direction
    the same as the one from its middle point to the vanishing point. The projected points of the original end points
    of the line onto the new direction will be the new end points of the refined line.
    :param line: line segment with two end points
    :param vpt: vanishing point
    :return:
    """
    a = line[0]
    b = line[1]
    mpt = (a + b) / 2.0
    line[0] = pointOnLine(vpt, mpt, a)
    line[1] = pointOnLine(vpt, mpt, b)

    return line
