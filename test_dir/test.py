import os, sys, glob, cv2, matplotlib
import tkinter

import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np

from tqdm import tqdm
from itertools import product
from facenet_pytorch import MTCNN

def build_axes(ax_size, rows = 1, cols = 1):
    fig, axes = plt.subplots(rows, cols, figsize = ax_size, constrained_layout = True, squeeze = False)

    return fig, axes

def draw_axes(ax_size):
    fig, axes = build_axes(ax_size, 6, 6)

    for ax in axes.flatten():
        plt.axes(ax)
        plt.axis("off")

    return fig, axes

def draw_axes2(ax_size, rows, cols):
    fig, axes = build_axes(ax_size, rows, cols)

    for ax in axes.flatten():
        plt.axes(ax)

    return fig, axes

def align_images(images, idx_ref, sz_window):
    def norm_sig(sig):
        return np.float32((sig - sig.min()) / (sig.max() - sig.min()))
    
    image_ref = images[idx_ref]

    H, W = image_ref.shape
    template = norm_sig(image_ref)[
        (H // 2) - sz_window : (H // 2) + sz_window, (W // 2) - sz_window : (W // 2) + sz_window
    ]

    rt_mats = []
    for idx, image in enumerate(images):
        rt_mat = np.eye(3, 3)[:2]

        if idx == idx_ref:
            rt_mats.append(rt_mat)
            continue

        image_target = np.float32((image - image.min()) / (image.max() - image.min()))

        cross_correlation = cv2.matchTemplate(image_target, template, method = cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(cross_correlation)

        trans_x, trans_y = max_loc[0] + sz_window - W // 2, max_loc[1] + sz_window - W // 2
        rt_mat[0, -1] = -trans_x
        rt_mat[1, -1] = -trans_y
        rt_mats.append(rt_mat)

    imgs = []
    for idx, (image, rt_mat) in enumerate(zip(images, rt_mats)):
        imgs.append(cv2.warpAffine(image, rt_mat, (image.shape[1], image.shape[0])))

    return imgs

def show_mtcnn(img, boxes, points):
    img_resz = cv2.resize(img[0], fx=0.5, fy=0.5, dsize=(0, 0))
    boxes_resz = (boxes[0] / 2).astype(int)
    boxes_resz[:, 2] = boxes_resz[:, 2] - boxes_resz[:, 0]
    boxes_resz[:, 3] = boxes_resz[:, 3] - boxes_resz[:, 1]
    width, height = np.mean(boxes_resz[:, 2]), np.mean(boxes_resz[:, 3])

    # order sort
    boxes_resz = sorted(boxes_resz, key=lambda pts: pts[1])
    boxes_resz[0:6] = sorted(boxes_resz[0:6], key=lambda pts: pts[0])
    boxes_resz[6:12] = sorted(boxes_resz[6:12], key=lambda pts: pts[0])
    boxes_resz[12:18] = sorted(boxes_resz[12:18], key=lambda pts: pts[0])
    boxes_resz[18:24] = sorted(boxes_resz[18:24], key=lambda pts: pts[0])
    boxes_resz[24:30] = sorted(boxes_resz[24:30], key=lambda pts: pts[0])
    boxes_resz[30:36] = sorted(boxes_resz[30:36], key=lambda pts: pts[0])
    boxes_resz.reverse()

    points_resz = (points[0] / 2).astype(np.float32)
    points_resz = sorted(points_resz, key=lambda pts: pts[0][1])
    points_resz[0:6] = sorted(points_resz[0:6], key=lambda pts: pts[0][0])
    points_resz[6:12] = sorted(points_resz[6:12], key=lambda pts: pts[0][0])
    points_resz[12:18] = sorted(points_resz[12:18], key=lambda pts: pts[0][0])
    points_resz[18:24] = sorted(points_resz[18:24], key=lambda pts: pts[0][0])
    points_resz[24:30] = sorted(points_resz[24:30], key=lambda pts: pts[0][0])
    points_resz[30:36] = sorted(points_resz[30:36], key=lambda pts: pts[0][0])
    points_resz.reverse()

    pts_zero = points_resz[0]
    pts_zero_c = np.mean(pts_zero, axis=0)
    left_top = np.array([pts_zero_c[0] - width / 2, pts_zero_c[1] - height / 2])

    for i in range(len(points_resz)):
        pts = points_resz[i]
        pts_c = np.mean(pts, axis=0)
        left_top_new = left_top + (pts_c - pts_zero_c)

        box = np.array([*left_top_new, width, height], dtype=int)

        cv2.rectangle(img_resz, rec=box, color=(255, 255, 255), thickness=2)
        cv2.putText(img_resz, f"{i+1}", box[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for idx, pt in enumerate(pts):
            cv2.circle(img_resz, pt.astype(int), 1, (255, 0, 255), 1, cv2.LINE_AA)
            if idx == 0:
                cv2.putText(
                    img_resz, f"{i+1}", pt.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )

    cv2.imshow("1", img_resz)
    cv2.waitKey(0)

def mtcnn_landmark(image, boxes, points, new_size):
    #order sort
    boxes_resz = boxes[0].astype(int)
    boxes_resz[:, 2] = boxes_resz[:, 2] - boxes_resz[:, 0]
    boxes_resz[:, 3] = boxes_resz[:, 3] - boxes_resz[:, 1]
    width, height  = np.mean(boxes_resz[:, 2]), np.mean(boxes_resz[:, 3])

    points_resz = points[0].astype(np.float32)
    points_resz = sorted(points_resz, key=lambda pts: pts[0][1])
    points_resz[0:6] = sorted(points_resz[0:6], key=lambda pts: pts[0][0])
    points_resz[6:12] = sorted(points_resz[6:12], key=lambda pts: pts[0][0])
    points_resz[12:18] = sorted(points_resz[12:18], key=lambda pts: pts[0][0])
    points_resz[18:24] = sorted(points_resz[18:24], key=lambda pts: pts[0][0])
    points_resz[24:30] = sorted(points_resz[24:30], key=lambda pts: pts[0][0])
    points_resz[30:36] = sorted(points_resz[30:36], key=lambda pts: pts[0][0])
    points_resz.reverse()

    pts_zero = points_resz[0]
    pts_zero_c = np.mean(pts_zero, axis=0)
    left_top = np.array([pts_zero_c[0] - width / 2, pts_zero_c[1] - height / 2])

    # crop and save
    img_gray = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
    
    st = []
    for i in range(len(points_resz)):

        pts = points_resz[i]
        pts_c = np.mean(pts, axis=0)
        left_top_new = left_top + (pts_c - pts_zero_c)

        box = np.array([*left_top_new, left_top_new[0] + width, left_top_new[1] + height], dtype=int)

        img_gray_cropped = img_gray[max(0, box[1]) : max(0, box[3]), max(0, box[0]) : max(0, box[2])]
        st.append(cv2.resize(img_gray_cropped, new_size))

    return np.stack(st).transpose(1, 2, 0)

def main():
    ## template matching
    image_raw = cv2.imread("test_dir/dataset/02HSI1.bmp")

    # Height, Width, Channel
    H, W, C = image_raw.shape

    fig, axes = build_axes((10, 10))
    plt.axes(axes[0, 0])
    plt.imshow(image_raw)

    plt.show()

    cy, cx = int(H/2), int(W/2)
    image_ = image_raw.copy()

    sz_step = 300
    lt = (cx - 3 * sz_step, cy - 3 * sz_step)

    anch_x = [lt[0] + sz_step * i for i in range(6)]
    anch_y = [lt[1] + sz_step * i for i in range(6)]
    anchors = [(x, y) for (y, x) in product(anch_y, anch_x)]

    for idx, (x, y) in enumerate(anchors, start = 1):
        cv2.circle(image_, (x, y), radius = 5, thickness = -1, color = (0, 255, 0))
        cv2.putText(image_, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    fig, axes = build_axes((10, 10))
    plt.axes(axes[0, 0])
    plt.imshow(image_)

    plt.show()

    fig, axes = draw_axes((10, 10))

    image_crop = []
    for x, y in anchors:
        image_ = image_raw[y : y + sz_step, x : x + sz_step, :]
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        image_crop.append(image_)

    for i, ax in enumerate(axes.flatten()):
        plt.axes(ax)
        plt.imshow(image_crop[i], cmap = "gray")
        
        plt.show()

    ref_idx = 14
    image_aligned = align_images(image_crop, ref_idx, 50)

    sz_cube = 250
    cube_means_tm = np.stack(image_aligned).transpose(1, 2, 0)

    io.savemat("./cube_means_tm.mat", {"cube_means_tm": cube_means_tm})

    fig, axes = draw_axes2((8, 120), 36, 2)
    plt.set_cmap("gray")

    for idx in range(len(image_crop)):
        start = sz_step // 2 - sz_cube // 2
        image1 = image_crop[ref_idx][start : start + sz_cube, start : start + sz_cube]
        image2 = image_crop[idx][start : start + sz_cube, start : start + sz_cube]
        iamge3 = image_aligned[idx][start : start + sz_cube, start : start + sz_cube]

        plt.axes(axes[idx, 0])
        plt.imshow(image1 * 0.5 + image2 * 0.5)

        plt.show()

        plt.axes(axes[idx, 1])
        plt.imshow(image1 * 0.5 + iamge3 * 0.5)

        plt.show()

    ## show mtcnn
    path = "test_dir/dataset"
    mtcnn = MTCNN(post_process = False, device = "cuda:0")
    
    name = "02HSI1.bmp"

    image_path = os.path.join(path, name)
    image = cv2.imread(image_path)

    H, W, C = image.shape
    image = image.reshape(1, H, W, C)

    # boxes: x_min, y_min, x_max, y_max
    boxes, probs, points = mtcnn.detect(image, landmarks = True)

    # show mtcnn face detection result
    show_mtcnn(image, boxes, points)

    ## From raw image to Cube
    new_size = (128, 128)

    for file in tqdm(glob.glob("*.bmp", root_dir = path)):
        image = cv2.imread(os.path.join(path, file))
        H, W, C = image.shape
        image = image.reshape(1, H, W, C)
        boxes, probs, points = mtcnn.detect(image, landmarks = True)
        cube = mtcnn_landmark(image, boxes, points, new_size)
        print(f"{file} : {cube.shape}")

        io.savemat(f"{path}/{file[:-4]}.mat", {"cube_meas": cube})

if __name__ == "__main__":
    matplotlib.use("TkAgg")
    main()
