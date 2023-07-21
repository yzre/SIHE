#-*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from lineClassification import *
from lineDrawingConfig import *
from lineRefinement import *
from filesIO import *
import skimage.io
import copy


def gt_measurement(zgt_img, a, b, verbose=False):
    """
    If there is a ground truth image with each pixel value represents the vertical z value, the ground truth height of a
    vertical line can be measured.
    :param zgt_img: the ground truth image with z values
    :param a: the up point of a vertical line segment
    :param b: the bottom point of a vertical line segment
    :param verbose: when true, show the results
    :return:
    """

    if a[1] > b[1]:
        temp = copy.deepcopy(a)
        a = b
        b = temp

    # check the a point & b point
    a = np.cast["int"](a + [0.5, 0.5])
    b = np.cast["int"](b + [0.5, 0.5])

    rows, cols = zgt_img.shape

    row_check = lambda x : min(rows - 1, max(0, x))
    cols_check = lambda x: min(cols - 1, max(0, x))
    pt_check = lambda pt:np.asarray([cols_check(pt[0]), row_check(pt[1])])

    a = pt_check(a)
    b = pt_check(b)

    gt_org = 0
    gt_expd= 0

    if zgt_img[a[1],a[0]] == 0 or zgt_img[b[1],b[0]] == 0:
        gt_org = 0
    else:
        gt_org = abs(zgt_img[a[1],a[0]] - zgt_img[b[1],b[0]])

    direction = (a-b)/np.linalg.norm(a - b)

    b_expd = copy.deepcopy(b)
    count = 1
    while zgt_img[b_expd[1], b_expd[0]] == 0 and a[1] < b_expd[1]:
        b_expd = np.cast["int"](b + count*direction)
        count = count + 1
    a_expd = copy.deepcopy(a)
    count = 1

    if zgt_img[a_expd[1],a_expd[0]] == 0:
        while a_expd[0]>0 and a_expd[0] <= cols -1 and a_expd[1] <=rows - 1 and zgt_img[a_expd[1],a_expd[0]] == 0:
            a_expd = np.cast["int"](a - count*direction)
            count = count + 1
    else:
        while a_expd[0]>0 and a_expd[0] <= cols -1 and a_expd[1] >=0 and zgt_img[a_expd[1],a_expd[0]]!= 0:
            a_expd = np.cast["int"](a + count*direction)
            count = count + 1
        a_expd = np.cast["int"](a + (count-2)*direction)
        pass
    gt_expd = abs(zgt_img[a_expd[1],a_expd[0]] - zgt_img[b_expd[1],b_expd[0]])

    if verbose:
        print("here---------------:")
        print(a_expd)
        print(b_expd)
        print(zgt_img[a_expd[1],a_expd[0]])
        print(gt_org,gt_expd)

    if verbose:
        plt.close()
        plt.figure()
        plt.imshow(zgt_img)
        plt.plot([a[0], b[0]], [a[1], b[1]], c=c(0), linewidth=2)
        plt.scatter(a[0], a[1], **PLTOPTS)
        plt.scatter(b[0], b[1], **PLTOPTS)
        # plt.show()

    return gt_org, gt_expd


def sv_measurement(v1, v2, v3, x1, x2, zc = 2.5):
    """
    Use single-view metrology and three vanishing points to calculate height.
    :param v1: vanishing point on the horizontal vanishing line
    :param v2: vanishing point on the horizontal vanishing line
    :param v3: vertical vanishing point
    :param x1: bottom point of the vertical line segment
    :param x2: top point of the vertical line segment
    :param zc: camera height, unit is meter
    :return: height zx
    """

    vline = np.cross(v1, v2)
    p4 = vline / np.linalg.norm(vline)

    zc = zc * np.linalg.det([v1, v2, v3])
    alpha = -np.linalg.det([v1, v2, p4]) / zc  # the scalar
    p3 = alpha * v3

    # rho = np.dot(x1, p4)/(1 + zc * np.dot(p3, p4))
    # zx = -np.linalg.norm(np.cross(x1, x2))/(rho * np.linalg.norm(np.cross(p3, x2)))

    zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
    zx = abs(zx)  # v1 and v2 may exchange (vanishing line can have two directions)

    return zx


def sv_measurement1(v, vline, x1, x2, zc=2.5):
    """
    Use single-view metrology and vertical vanishing point along with horizontal vanishing line to calculate height.
    :param v: vertical vanishing point
    :param vline: horizontal vanishing line
    :param x1: bottom point of the vertical line segment
    :param x2: top point of the vertical line segment
    :param zc: camera height, unit is meter
    :return: height zx
    """

    p4 = vline / np.linalg.norm(vline)

    alpha = -1 / (np.dot(p4, v) * zc)  # the scalar
    p3 = alpha * v

    # rho = np.dot(x1, p4) / (1 + zc * np.dot(p3, p4))
    # zx = -np.linalg.norm(np.cross(x1, x2)) / (rho * np.linalg.norm(np.cross(p3, x2)))

    zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
    zx = abs(zx)  # vanishing line can have two directions

    return zx


def singleViewMeasWithCrossRatio(hori_v1, hori_v2, vert_v1, pt_top, pt_bottom, zc=2.5):
    """
    Use single-view metrology and three vanishing points to calculate height. The function has the same effect as
    "sv_measurement()", with a different calculation method. Pay attention to x, y order of the input.
    :param hori_v1: image coordinates of a vanishing point on the horizontal vanishing line
    :param hori_v2: image coordinates of a vanishing point on the horizontal vanishing line
    :param vert_v1: image coordinates of the vertical vanishing point
    :param pt_top: bottom point of the vertical line segment
    :param pt_bottom: top point of the vertical line segment
    :param zc: camera height, unit is meter
    :return: height
    """
    line_vl = lineCoeff(hori_v1, hori_v2)
    line_building_vert = lineCoeff(pt_top, pt_bottom)
    C = intersection(line_vl, line_building_vert)

    dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
    dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
    dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
    dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

    height = dist_BD*dist_AC/(dist_CD*dist_AB)*zc
    return height


def singleViewMeasWithCrossRatio_vl(hori_vline, vert_v1, pt_top, pt_bottom, zc=2.5):
    """
    Use single-view metrology and vertical vanishing point along with horizontal vanishing line to calculate height.
    The function has the same effect as "sv_measurement1()", with a different calculation method.
    Pay attention to x, y order of the input.
    :param hori_vline: image coordinates of the horizontal vanishing line
    :param vert_v1: image coordinates of the vertical vanishing point
    :param pt_top: bottom point of the vertical line segment
    :param pt_bottom: top point of the vertical line segment
    :param zc: camera height, unit is meter
    :return: height
    """

    line_vl = hori_vline
    line_building_vert = lineCoeff(pt_top, pt_bottom)
    C = intersection(line_vl, line_building_vert)

    dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
    dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
    dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
    dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

    height = dist_BD*dist_AC/(dist_CD*dist_AB)*zc
    return height


def vp_calculation_with_pitch(w, h, pitch, focal_length):
    """
    Calculate the vertical vanishing point and the horizontal vanishing line through pitch angle. Note: this function is
    specially set for street view images with known rotation angles (pitch, yaw, and roll), e.g. Google street view.
    Normally, the roll angle is zero. The pitch is used for the calculation.
    :param w: image width
    :param h: image height
    :param pitch: pitch angle
    :param focal_length: focal length
    :return: v, the vertical vanishing point, and vline, the horizontal vanishing line
    """

    # initialize
    v = np.array([w / 2, 0.0, 1.0])  # pitch will influence the second element of v, v[1]
    vline = np.array([0.0, 1.0, 0.0])  # pitch will influence the third element of vline, vline[2]

    if pitch == 0:
        v[:] = [0, -1, 0]
        vline[:] = [0, 1, h / 2]
    else:
        v[1] = h / 2 - (focal_length / np.tan(np.deg2rad(pitch)))
        vline[2] = (h / 2 + focal_length * np.tan(np.deg2rad(pitch)))

    # print(v)
    # print(vline)

    return v, vline


def heightCalc(fname_dict, intrins, config, img_size=None, pitch=None, use_pitch_only=0, use_detected_vpt_only=0, verbose=False):
    """
    Estimate the height of buildings.
    :param fname_dict: the dictionary of file names
    :param intrins: the intrinsic matrix
    :param config: the configuration
    :param img_size: the size of the image
    :param pitch: the pitch angle of the image
    :param use_pitch_only: when the value is '1', use only the pitch angle to calculate vanishing line and vertical vanishing point
    :param use_detected_vpt_only: when the value is '1', use only the detected vanishing points
    :param verbose: when true, show the results
    :return:
    """

    if img_size is None:
        img_size = [640, 640]

    try:
        vpt_fname = fname_dict["vpt"]
        img_fname = fname_dict["img"]
        line_fname = fname_dict["line"]
        seg_fname =  fname_dict["seg"]
        zgt_fname = fname_dict["zgt"]

        # ######### get the vanishing points
        w = img_size[0]
        h = img_size[1]
        focal_length = intrins[0, 0]
        if use_pitch_only:
            # initialize the vanishing points
            # note: vps is set to keep the same format as the detected vps
            # and only vps[2] (the vertical vanishing point) is used together with the vanishing line
            vps = np.zeros([3, 2])

            # calculate the vertical vanishing point and the vanishing line
            vertical_v, vline = vp_calculation_with_pitch(w, h, pitch, focal_length)

            # transformation of vps to 2D image coordinates
            if vertical_v[2] == 0:  # a special case
                vertical_v[0] = 320
                vertical_v[1] = -9999999
            vps[2, :] = vertical_v[:2]

        elif '.npz' in vpt_fname:
            vps = load_vps_2d(vpt_fname)

            # if not 'use_detected_vpt_only', use the calculated vertical vanishing point to replace the detected one
            if not use_detected_vpt_only:
                vertical_v, vline = vp_calculation_with_pitch(w, h, pitch, focal_length)

                # transformation of vps to 2D image coordinates
                if vertical_v[2] == 0:  # a special case
                    vertical_v[0] = 320
                    vertical_v[1] = -9999999
                vps[2, :] = vertical_v[:2]

        # ######### get the detected line segments and the semantic segmentation results
        line_segs, scores = load_line_array(line_fname)
        seg_img = load_seg_array(seg_fname)

        # save the visualization of the line/segmentation results
        org_image = skimage.io.imread(img_fname)
        for i, t in enumerate([0.94]):  # lines with different score thresholds ([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            for (a, b), s in zip(line_segs, scores):
                if s < t:
                    continue
                plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
                plt.scatter(a[1], a[0], **PLTOPTS)
                plt.scatter(b[1], b[0], **PLTOPTS)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(org_image)
            plt.imshow(seg_img, alpha=0.5)

            # show the vanishing points and vanishing line
            if use_pitch_only:
                x, y = vertical_v[:2]
                plt.scatter(x, y)
                plt.plot([0, w], [vline[2], vline[2]], c='b', linewidth=5)
            else:
                for i in range(len(vps)):
                    x, y = vps[i]
                    plt.scatter(x, y)

            integrated_save_name = img_fname.replace(".jpg", f"-{t:.02f}_inls.svg")
            integrated_save_name = integrated_save_name.replace("imgs", "inls")
            integrated_save_dir = os.path.dirname(integrated_save_name)
            if not os.path.exists(integrated_save_dir):
                os.makedirs(integrated_save_dir)
            # plt.show()
            plt.close()

        # ######### processing the line segments
        if verbose:
            plt.close()
            org_img = skimage.io.imread(img_fname)
            plt.imshow(org_img)
            plt.imshow(seg_img, alpha=0.5)

        # classify the line segments and extend vertical lines
        verticals = filter_lines_outof_building_ade20k(img_fname, line_segs, scores, seg_img, vps, config, use_pitch_only)
        verticals = verticalLineExtending(img_fname, verticals, seg_img, [vps[2, 1], vps[2, 0]], config)
        # verticals, bottoms, roofs = filter_lines_outof_building_ade20k(img_fname, line_segs, scores, seg_img, vps, config, use_pitch_only)
        # verticals = verticalLineExtendingWithBRLines(img_fname, verticals, roofs, bottoms, seg_img, config)

        # ######### calculate heights of processed vertical line segments
        invK = np.linalg.inv(intrins)
        ht_set = []
        check_list = []

        for line in verticals:
            # only consider a, b as integers
            # a = np.cast["int"](line[0] + 0.5)
            # b = np.cast["int"](line[1] + 0.5)
            a = line[0]
            b = line[1]

            # remove duplicate a,b because of integer
            if len(check_list) !=0 :
                flag = 0
                for a0,a1,b0,b1 in check_list:
                    if a0 == a[0] and a1 == a[1] and b0 == b[0] and b1 == b[1]:
                        flag=1
                        break
                if flag:
                    continue
            check_list.append([a[0], a[1], b[0], b[1]])

            # swap x and y, as coordinates here are expressed in [y, x] order
            a_d3 = np.asarray([a[1], a[0], 1])
            a_d3 = np.matmul(invK, np.transpose(a_d3))

            b_d3 = np.asarray([b[1], b[0], 1])
            b_d3 = np.matmul(invK, np.transpose(b_d3))

            if use_detected_vpt_only:
                vps0 = np.asarray([vps[0, 0], vps[0, 1], 1])
                vps1 = np.asarray([vps[1, 0], vps[1, 1], 1])

                use_horizontal_property_to_refine = 0
                if use_horizontal_property_to_refine:
                    vps0 = np.asarray([vps[0, 0], (vps[0, 1] + vps[1, 1]) / 2.0, 1])
                    vps1 = np.asarray([vps[1, 0], (vps[0, 1] + vps[1, 1]) / 2.0, 1])

                vps0 = np.matmul(invK, np.transpose(vps0))
                vps1 = np.matmul(invK, np.transpose(vps1))

                vps2 = np.asarray([vps[2, 0], vps[2, 1], 1])
                vps2 = np.matmul(invK, np.transpose(vps2))
                ht = sv_measurement(vps0, vps1, vps2, b_d3, a_d3, zc=float(config["STREET_VIEW"]["CameraHeight"]))
            else:
                ht = singleViewMeasWithCrossRatio_vl(vline, vertical_v[:2], np.asarray([a[1], a[0]]),
                                                     np.asarray([b[1], b[0]]),
                                                     zc=float(config["STREET_VIEW"]["CameraHeight"]))

            gt_exist = int(config["GROUND_TRUTH"]["Exist"])
            if gt_exist:
                zgt_img = load_zgts(zgt_fname)
                ht_gt_org, ht_gt_expd = gt_measurement(zgt_img,np.asarray([a[1], a[0]]), np.asarray([b[1], b[0]]))
            else:
                ht_gt_org, ht_gt_expd = ht*0, ht*0
            ht_set.append([ht, a, b, ht_gt_org, ht_gt_expd])

        if verbose:
            plt.close()
            # plt.figure(figsize=(18, 8))
            # plt.subplot(121)
            plt.figure(figsize=(10, 8))
            org_img = skimage.io.imread(img_fname)
            plt.imshow(org_img)
            plt.imshow(seg_img, alpha=0.5)
        print("path:%s" % img_fname)

        # divide vertical line segments into groups using computed heights
        grouped_lines = clausterLinesWithCenters(ht_set, config, using_height=True)
        if grouped_lines is None:
            print('no suitable vertical lines founded in image ' + img_fname)
            return

        list_len = len(grouped_lines)
        heights = []
        ax_legends = []
        if len(colors_tables) < list_len:
            print("warning: lines with the same color might be different groups.")
        for i in range(list_len):
            lines = grouped_lines[i]
            list_len_lines = len(lines)
            # rng = np.random.default_rng()
            # colors = np.cast['float'](rng.integers(255, size=3))
            # colors = colors / (np.linalg.norm(colors) + 0.0001)
            heights.append([lines[-2], lines[-1]])
            for j in range(list_len_lines - 2):
                # plot points
                a = lines[j][1]
                b = lines[j][2]
                if verbose:
                    ax_line, = plt.plot([a[1], b[1]], [a[0], b[0]], c=colors_tables[i % len(colors_tables)], linewidth=2)
                    plt.scatter(a[1], a[0], **PLTOPTS)
                    plt.scatter(b[1], b[0], **PLTOPTS)
            ax_legends.append(ax_line)

        if verbose:
            plt.legend(ax_legends, ['average_height = %.4fm, median_height = %.4fm' % (y, x) for x, y in heights])
            # plt.legend(ax_legends, ['median_height = %.4fm, average_height = %.4fm' % (x, y) for x, y in heights])
            result_save_name = img_fname.replace('imgs', 'ht_results')
            result_save_name = result_save_name.replace('.jpg', '_htre.svg')
            result_save_name2 = result_save_name.replace('.svg', '.png')
            re_save_dir = os.path.dirname(result_save_name)
            if not os.path.exists(re_save_dir):
                os.makedirs(re_save_dir)
            plt.savefig(result_save_name, bbox_inches="tight")
            plt.savefig(result_save_name2, bbox_inches="tight")
            # plt.show()
            plt.close()

            pass

    except IOError:
        print("file does not exist\n")
