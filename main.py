import os, re, cv2

import numpy as np
import scipy.io as io
# import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product
from facenet_pytorch import MTCNN

def modify_image():
    '''
    bmp: 180도 회전 
    mp4: 6 frame 단위로 정제
    이름 변경
    '''

    exp_dir = {
        "exp": "04",
        "exp0": "02",
        "exp1": "07",
        "exp2": "06",
        "exp3": "05",
        "exp4": "09",
        "exp5": "08",
        "exp6": "03",
        "exp7": "01",
        "exp8": "10",
        "exp9": "11",
        "exp10": "17",
        "exp11": "15",
        "exp12": "12",
        "exp13": "18",
        "exp14": "13",
        "exp15": "16",
        "exp16": "14",
        "exp17": "20",
        "exp18": "19",
        "exp19": "21",
        "exp20": "22",
        "exp22": "26",
        "exp23": "24",
        "exp24": "25",
        "exp25": "23",
        "exp26": "27",
        "exp27": "28",
        "exp28": "30",
        "exp29": "32",
        "exp30": "29",
        "exp31": "31",
        "exp32": "33",
        "exp33": "34",
        "exp34": "39",
        "exp35": "37",
        "exp36": "35",
        "exp37": "38",
        "exp38": "40",
        "exp39": "36",
        "exp40": "41",
        "exp41": "43",
        "exp42": "46",
        "exp43": "45",
        "exp44": "44",
        "exp45": "42",
        "exp46": "47",
    }

    rgb_dir = {
        "WIN_20250320_16_52_53_Pro": "07",
        "WIN_20250320_17_23_35_Pro": "06",
        "WIN_20250320_18_06_51_Pro": "05",
        "WIN_20250321_13_46_47_Pro": "09",
        "WIN_20250321_13_59_08_Pro": "08",
        "WIN_20250321_14_14_08_Pro": "03",
        "WIN_20250321_14_45_22_Pro": "01",
        "WIN_20250321_14_59_43_Pro": "10",
        "WIN_20250321_15_14_34_Pro": "11",
        "WIN_20250323_11_12_21_Pro": "17",
        "WIN_20250323_11_21_33_Pro": "15",
        "WIN_20250323_11_32_18_Pro": "12",
        "WIN_20250323_11_43_49_Pro": "18",
        "WIN_20250323_11_53_04_Pro": "13",
        "WIN_20250323_12_03_59_Pro": "16",
        "WIN_20250323_12_28_30_Pro": "14",
        "WIN_20250324_14_58_45_Pro": "20",
        "WIN_20250324_15_25_20_Pro": "19",
        "WIN_20250325_14_32_12_Pro": "21",
        "WIN_20250326_16_07_56_Pro": "22",
        "WIN_20250330_14_03_59_Pro": "26",
        "WIN_20250330_14_14_26_Pro": "24",
        "WIN_20250330_14_24_06_Pro": "25",
        "WIN_20250330_15_58_59_Pro": "23",
        "WIN_20250330_16_09_47_Pro": "27",
        "WIN_20250331_11_17_44_Pro": "28",
        "WIN_20250401_14_09_34_Pro": "30",
        "WIN_20250401_14_23_24_Pro": "32",
        "WIN_20250401_17_43_34_Pro": "29",
        "WIN_20250401_17_52_23_Pro": "31",
        "WIN_20250403_17_41_32_Pro": "33",
        "WIN_20250403_17_50_30_Pro": "34",
        "WIN_20250404_14_10_36_Pro": "39",
        "WIN_20250404_14_11_59_Pro": "39",
        "WIN_20250404_14_20_41_Pro": "37",
        "WIN_20250404_14_30_11_Pro": "35",
        "WIN_20250404_16_14_47_Pro": "38",
        "WIN_20250404_16_23_53_Pro": "40",
        "WIN_20250404_16_32_56_Pro": "36",
        "WIN_20250404_16_41_45_Pro": "41",
        "WIN_20250404_16_43_39_Pro": "41",
        "WIN_20250406_16_00_28_Pro": "43",
        "WIN_20250406_16_10_06_Pro": "46",
        "WIN_20250406_16_20_05_Pro": "45",
        "WIN_20250406_16_32_43_Pro": "44",
        "WIN_20250406_16_42_04_Pro": "42",
        "WIN_20250407_17_31_51_Pro": "47",
    }

    ## Load image files
    video_and_exp_dir = [i for i in os.listdir("raw-dataset") if ("exp" in i and f"HSI{exp_dir[i]}" not in os.listdir("dataset")) or (("WIN" in i) and (f"RGB{rgb_dir[i[:-4]]}" not in os.listdir("dataset") and f"RGB{rgb_dir[i[:-4]]}-need-clean" not in os.listdir("dataset")))]
    video_and_exp_dir.sort() ## only for RGB39 and RGB41, I don't know other case works either

    done_item = []
    for item in tqdm(video_and_exp_dir):
        ## hsi image인 경우
        if "exp" in item:
            for elem in tqdm(os.listdir(f"raw-dataset/{item}")):
                ## 예외 처리: 해당 디렉토리 내 bmp 형식이 아닌 파일이 있는 경우
                if elem[-3:] != "bmp":
                    continue

                ## 이미지 파일 읽기
                image = cv2.imread(f"raw-dataset/{item}/{elem}")
            
                ## 180도 회전
                h, w = image.shape[:2]
                cX, cY = w // 2, h // 2
                M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
                image = cv2.warpAffine(image, M, (w, h))

                ## 디렉토리 생성 및 이미지 저장
                os.makedirs(f"dataset/HSI{exp_dir[item]}", exist_ok = True)
                cv2.imwrite(f"dataset/HSI{exp_dir[item]}/{exp_dir[item]}HSI{elem[:-4]}.bmp", image)

        ## rgb image인 경우
        elif "WIN" in item:
            ## 디렉토리 생성
            os.makedirs(f"dataset/RGB{rgb_dir[item[:-4]]}-need-clean", exist_ok = True)

            ## 비디오 -> 이미지 변환
            frame = 1

            ## 이어지는 비디오 파일이 있는 경우 frame 조절
            for i in done_item:
                if rgb_dir[i[:-4]] == rgb_dir[item[:-4]]:
                    video = cv2.VideoCapture(f"raw-dataset/{i}")
                    success, image = video.read()

                    while success:
                        success, image = video.read()
                        frame += 1
                    video.release()
                    break
                            
            ## 비디오 파일 읽기
            video = cv2.VideoCapture(f"raw-dataset/{item}")
            success, image = video.read()
            
            while success:
                ## 6 frame 단위로 정제
                if not (frame % 6):
                    cv2.imwrite(f"dataset/RGB{rgb_dir[item[:-4]]}-need-clean/{rgb_dir[item[:-4]]}RGB{frame // 6}.jpg", image)

                success, image = video.read()
                frame += 1

            ## video 해제
            video.release()

        ## 예외 처리: 해당 디렉토리 내 bmp 또는 mp.4 형식이 아닌 파일이 있는 경우
        else:
            continue

        done_item.append(item)

def rgb_rename_crop_resize():
    '''
    image 중 불필요한 내용 제거
    image를 256 x 256으로 크기 조절
    '''

    ## 파일 이름 뒤에 있는 숫자를 추출하는 함수
    def file_number(file_name):
        match = re.search(r'RGB(\d+)\.jpg$', file_name)
        return int(match.group(1)) if match else None
    
    def image_crop_and_resize(file_dir):
        ## 이미지 자르기
        image = cv2.imread(file_dir)
        h, w, _ = image.shape

        crop_image = image[100:855, int(w/2 - w/7):int(w/2 + w/4)]

        ## 이미지 크기 조절
        ressized_image = cv2.resize(crop_image, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)

        ## 이미지 저장
        cv2.imwrite(file_dir, ressized_image)
    
    ## dataset 디렉토리 중 정제된 RGB 이미지 디렉토리만 추출 및 정렬
    directories = [i for i in os.listdir("dataset") if ("RGB" in i and "-need-clean" not in i)]
    directories.sort()

    ## exception handling: 정제한 RGB가 없는 경우
    if not directories:
        print("Nothing to do")
        return
    
    ## 디렉토리 내 이미지 파일 정렬, 자르기, 크기 조절, 이름 변경 수행
    for item in directories:
        ## 디렉토리 내 이미지 파일 정렬
        frames = [i for i in os.listdir(f"dataset/{item}") if ".jpg" in i]
        sorted_frame = sorted(frames, key = file_number)

        item_number = int(item[-2:]) # 디렉토리로부터 숫자 추출
        ## RGB3, RGB5 ~ RGB12, RGB15, RGB18은 형광등 -> LED 정면 -> LED 좌측 순으로 촬영됨
        if (4 < item_number and item_number < 13 and item_number != 7) or item_number == 3 or item_number == 15 or item_number == 18:
            for idx, elem in enumerate(sorted_frame, start = 1):
                os.rename(f"dataset/{item}/{elem}", f"dataset/{item}/{item_number}RGB{idx}.jpg") # 이름 변경
                image_crop_and_resize(f"dataset/{item}/{item_number}RGB{idx}.jpg")


        ## RGB7, RGB13~는 LED 좌측 -> LED 정면 -> 형광등 순으로 촬영됨
        elif item_number == 7 or item_number > 12:
            file_order = 0
            for idx, elem in enumerate(sorted_frame, start = 1):
                if (idx == 1):
                    file_order = 45

                elif (idx == 23):
                    file_order = 23

                elif (idx == 45):
                    file_order = 1

                os.rename(f"dataset/{item}/{elem}", f"dataset/{item}/{item_number}RGB{file_order}.jpg")
                image_crop_and_resize(f"dataset/{item}/{item_number}RGB{file_order}.jpg")
                file_order += 1

def template_matching(hsi_dir : list):
    # def build_axes(ax_size, rows = 1, cols = 1):
    #     fig, axes = plt.subplots(rows, cols, figsize = ax_size, contrained_layout = True, squeeze = False)
    #     return fig, axes
    
    def align_images(images, idx_ref, sz_window):
        def norm_sig(sig):
            return np.float32((sig - sig.min()) / (sig.max() - sig.min()))
        
        img_ref = images[idx_ref]

        H, W = img_ref.shape
        template = norm_sig(img_ref)[
            (H // 2) - sz_window : (H // 2) + sz_window, (W // 2) - sz_window : (W // 2) + sz_window
        ]

        rt_mats = []
        for idx, img in enumerate(images):
            rt_mat = np.eye(3, 3)[:2]

            if idx == idx_ref:
                rt_mats.append(rt_mat)
                continue

            img_tgt = np.float32((img - img.min()) / (img.max() - img.min()))

            cross_correlation = cv2.matchTemplate(img_tgt, template, method = cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(cross_correlation)

            trans_x, trans_y = max_loc[0] + sz_window - W // 2, max_loc[1] + sz_window - H // 2
            rt_mat[0, -1] = -trans_x
            rt_mat[1, -1] = -trans_y
            rt_mats.append(rt_mat)

        imgs = []
        for idx, (img, rt_mat) in enumerate(zip(images, rt_mats)):
            imgs.append(cv2.warpAffine(img, rt_mat, (img.shape[1], img.shape[0])))

        return imgs
    
    for img_dir in hsi_dir:
        for img in os.listdir(f"dataset/hsi/{img_dir}"):
            if ".bmp" not in img:
                continue

            print(img)

            continue

            # Load image
            raw_image = cv2.imread(f"dataset/hsi/{img_dir}/{img}")

            # Height, Width, Channel
            H, W = raw_image.shape[:2]

            cy, cx = int(H / 2), int( W / 2)

            sz_step = 390
            lt = (cx - 3 * sz_step, cy - 3 * sz_step)

            image_ = raw_image.copy()

            anch_x = [lt[0] + sz_step * i for i in range(6)]
            anch_y = [lt[1] + sz_step * i for i in range(6)]
            anchors = [(x, y) for (y, x) in product(anch_y, anch_x)]

            for idx, (x, y) in enumerate(anchors, start = 1):
                cv2.circle(image_, (x, y), radius = 5, thickness = -1, color = (0, 255, 0))
                cv2.putText(image_, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            image_cropped = []
            for x, y in anchors:
                image_ = raw_image[y : y + sz_step, x : x + sz_step, :]
                image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
                image_cropped.append(image_)

            img_aligned = align_images(image_cropped, 14, 50)

            sz_cube = 250
            cube_meas_tm = np.stack(img_aligned).transpose(1, 2, 0)

            io.savemat(f"./dataset/hsi/{img_dir}/{img[:-4]}-cube_meas_tm.mat", {"cube_meas_tm" : cube_meas_tm})

def mtcnn_landmark(img, boxes, points):
    # 박스 변환 및 너비 / 높이 계산
    boxes_resz = boxes[0].astype(int)
    boxes_resz[:, 2] = boxes_resz[:, 2] - boxes_resz[:, 0]
    boxes_resz[:, 3] = boxes_resz[:, 3] - boxes_resz[:, 1]
    width, height = np.mean(boxes_resz[:, 2]), np.mean(boxes_resz[:, 3])

    # 랜드마크 좌표 처리
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
        st.append(cv2.resize(img_gray_cropped, (128, 128)))

    return np.stack(st).transpose(1, 2, 0)

def image_to_cube(hsi_dir : list):
    # post_process: 후처리를 하지 않고 원본 결과를 그대로 반환
    # device: GPU 사용
    mtcnn = MTCNN(post_process=False, device = "cuda:0")


    for file in hsi_dir:
        # Load image
        img = cv2.imread(f"dataset/hsi/{file}")

        # Height, Width, Channel
        H, W, C = img.shape

        # MTCNN은 (1, H, W, C) 형태의 이미지를 입력으로 받기 때문에 reshape
        img = img.reshape(1, H, W, C)

        # Detect faces
        # boxes: 얼굴을 감싸는 박스 좌표
        # probs: 얼굴 감지 확률
        # points: 얼굴 랜드마크 좌표
        boxes, probs, points = mtcnn.detect(img, landmarks=True)

        # cube: 128x128x36 크기의 이미지 큐브
        cube = mtcnn_landmark(img, boxes, points)

        # Save cube as .mat file
        io.savemat(f"./dataset/hsi/{file[:-4]}-cube_meas.mat", {"cube_meas": cube})

def modality_analysis():
    data = {}
    for subject_name in os.listdir("dataset/hsi"):
        pass

def main():
    # if len(os.listdir("dataset/hsi")) < 50 and len(os.listdir("dataset/rgb")) < 50:
    #     modify_image()
    #     rgb_rename_crop_resize()

    hsi_dir = [i for i in os.listdir("dataset/hsi") if "HSI" in i]
    template_matching(hsi_dir)

    

if __name__ == "__main__":
    main()