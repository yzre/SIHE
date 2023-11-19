# by Yizhen Yan

import numpy as np


# ######################### to transform vpts
def to_pixel_new(v, focal_length):
    x = v[0] / v[2] * focal_length * 256 + 256  # 256 is half the image width
    y = -v[1] / v[2] * focal_length * 256 + 256  # 256 is half the image width
    return x, y


def order_vpt(vps_2D, w=640.0):
    # order the vps_2D again to make v3 the vertical vps, and v1 & v2 the horizontal vps
    # here vps_2D_ordered = vps_2D cannot be used to initialize the vps_2D_ordered
    # because if so, the vps_2D_ordered will change when vps_2D change
    # w is image width, h is image height

    # in neurvps, it only deals with images whose h is equal to w
    # w = 640.0 # image width
    h = w  # image height
    vps_2D_ordered = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    dy = abs(vps_2D[:, 1] - h/2)  # the distance from the principle point in the y direction
    # dy_max_id = np.where(dy == np.max(dy))  # the indexes of the max dy
    dy_max_id = np.where(np.max(dy) - dy < 1)  # the indexes of the vps that have dys very close to max dy
    dy_max_id = dy_max_id[0]  # to get the indexes array from the dy_max_id tuple
    if dy_max_id.size == 1:
        v3 = vps_2D[dy_max_id[0], :]
        v3_id = dy_max_id[0]
    else:
        dx1 = abs(vps_2D[dy_max_id[0], 0] - w/2)  # the distance from the principle point in the x direction
        dx2 = abs(vps_2D[dy_max_id[1], 0] - w/2)  # the distance from the principle point in the x direction
        if dx1 < dx2:
            v3 = vps_2D[dy_max_id[0], :]  # the vertical vps
            v3_id = dy_max_id[0]
        else:
            v3 = vps_2D[dy_max_id[1], :]
            v3_id = dy_max_id[1]

    v_order = np.array([0, 1, 2])
    vh_id = np.where(v_order != v3_id)  # the indexes of the horizontal vps
    vh_id = vh_id[0]  # to get the indexes array from the dy_max_id tuple
    # if the x of one vps larger than the other one, it is the right horizontal vps
    if vps_2D[vh_id[0], 0] > vps_2D[vh_id[1], 0]:
        v1 = vps_2D[vh_id[0], :]  # the right horizontal vps
        v2 = vps_2D[vh_id[1], :]  # the left horizontal vps
    else:
        v1 = vps_2D[vh_id[1], :]  # the right horizontal vps
        v2 = vps_2D[vh_id[0], :]  # the left horizontal vps

    # here vps_2D[i,:]=vi cannot be used because if vps_2D changed, the vi calculated above
    # will change as well, so another variable vps_2D_ordered is used to avoid the problem
    vps_2D_ordered[0, :] = v1
    vps_2D_ordered[1, :] = v2
    vps_2D_ordered[2, :] = v3

    # print('vps_2D ordered: ')
    # print(vps_2D_ordered)
    return vps_2D_ordered


def transform_vpt(vpts, fov=120.0, orgimg_width=640.0):
    # transform 3D vpts to 2D vpts with image coordinates of original resolution
    # when detecting the vpts, the images are resized to 512*512
    # fov is field of view in degrees and used to compute focal length
    # orgimg_width is the width (resolution) of the original image

    v1 = vpts[0]
    v2 = vpts[1]
    v3 = vpts[2]

    # fov = 120.0  # field of view in degree
    f = 1 / np.tan(np.deg2rad(fov/2))
    print('focal length')
    print(f)

    p1 = to_pixel_new(v1, f)
    p2 = to_pixel_new(v2, f)
    p3 = to_pixel_new(v3, f)
    vpts_2d = np.array([p1, p2, p3])
    print('2d vpts')
    print(vpts_2d)

    # orgimg_width = 640.0  # the width (resolution) of the original image
    p1t = np.multiply(p1, orgimg_width / 512)
    p2t = np.multiply(p2, orgimg_width / 512)
    p3t = np.multiply(p3, orgimg_width / 512)
    vpts_2d_t = np.array([p1t, p2t, p3t])
    print('transformed 2d vpts')
    print(vpts_2d_t)

    vpts_2d_ordered = order_vpt(vpts_2d_t, w=orgimg_width)
    print('ordered 2d vpts')
    print(vpts_2d_ordered)

    return vpts_2d_ordered
# ######################### to transform vpts


if __name__ == "__main__":

    # vpts_pd: the direct vpt output from the model
    vpts_pd = np.load('vpt_filename')  # load from files or input the model output
    # show vpts detected
    print("/n the vpts of image is: ")
    print(vpts_pd)

    # transform vpts_pd to 2D pixel coordinates and transform them to coordinates of 640*640 image size
    vpts_re = transform_vpt(vpts_pd, fov=90.0, orgimg_width=512.0)  # for gsv images with size 512x512 and fov 90

    # for saving the predictions
    save_name = './vpts_results_sample.npz'
    np.savez(
        save_name,
        vpts_pd=vpts_pd,
        vpts_re=vpts_re,
        )

