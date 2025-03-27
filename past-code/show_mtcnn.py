import os, glob

import cv2
import numpy as np

from facenet_pytorch import MTCNN


new_size = (128, 128)

mtcnn = MTCNN(post_process=False, device="cuda:0")


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


def mtcnn_landmark(img, boxes, points):
    # order sort
    boxes_resz = boxes[0].astype(int)
    boxes_resz[:, 2] = boxes_resz[:, 2] - boxes_resz[:, 0]
    boxes_resz[:, 3] = boxes_resz[:, 3] - boxes_resz[:, 1]
    width, height = np.mean(boxes_resz[:, 2]), np.mean(boxes_resz[:, 3])

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
    img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
    st = []
    for i in range(len(points_resz)):

        pts = points_resz[i]
        pts_c = np.mean(pts, axis=0)
        left_top_new = left_top + (pts_c - pts_zero_c)

        box = np.array([*left_top_new, left_top_new[0] + width, left_top_new[1] + height], dtype=int)

        img_gray_cropped = img_gray[max(0, box[1]) : max(0, box[3]), max(0, box[0]) : max(0, box[2])]
        st.append(cv2.resize(img_gray_cropped, new_size))

    st = np.stack(st).transpose(1, 2, 0)
    print(st.shape)


def main():

    # path_base = "F:/gist-grad/research/projects/hsi-face-recog/20240724-demo/01-j_lee/01-live"
    path_base = "./dataset/01-yi_choi/01-live"

    name = "01-live.bmp"
    # name = "02-print.bmp"
    # name = "03-ipad.bmp"
    # name = "04-iphone.bmp"

    path_img = os.path.join(path_base, name)
    img = cv2.imread(path_img)

    H, W, C = img.shape
    print(img.shape)
    img = img.reshape(1, H, W, C)

    # boxes: x_min, y_min, x_max, y_max
    boxes, probs, points = mtcnn.detect(img, landmarks=True)

    # show mtcnn face detection results
    show_mtcnn(img, boxes, points)

    # # mtcnn_box(img, boxes, points)
    # mtcnn_landmark(img, boxes, points)


if __name__ == "__main__":
    main()

# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())