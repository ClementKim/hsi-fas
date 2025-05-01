import os, re, cv2, torch

import numpy as np
import scipy.io as io
# import matplotlib.pyplot as plt

from tqdm import tqdm
# from ultralytics import YOLO
from itertools import product
from facenet_pytorch import MTCNN

### Junsung
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
        "exp47": "55",
        "exp48": "53",
        "exp49": "49",
        "exp50": "52",
        "exp51": "50",
        "exp52": "48",
        "exp53": "51",
        "exp54": "54",
        "exp55": "56",
        "exp56": "3",
        "exp57": "2",
        "exp58": "1"
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
        "WIN_20250412_14_09_42_Pro": "55",
        "WIN_20250412_14_19_25_Pro": "53",
        "WIN_20250412_14_30_34_Pro": "49",
        "WIN_20250412_14_39_45_Pro": "52",
        "WIN_20250412_15_25_37_Pro": "50",
        "WIN_20250412_15_35_19_Pro": "48",
        "WIN_20250412_15_44_27_Pro": "51",
        "WIN_20250412_15_57_14_Pro": "54",
        "WIN_20250412_16_06_01_Pro": "56",
        "WIN_20250417_18_02_17_Pro": "3",
        "WIN_20250417_18_03_16_Pro": "2",
        "WIN_20250417_18_04_11_Pro": "1",
        "WIN_20250417_18_05_11_Pro": "3",
        "WIN_20250417_18_06_14_Pro": "2",
        "WIN_20250417_18_07_15_Pro": "1",
        "WIN_20250417_18_08_14_Pro": "1",
        "WIN_20250417_18_09_06_Pro": "2",
        "WIN_20250417_18_10_03_Pro": "3"
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

### Junsung -- need edit
def rgb_rename_crop_resize():
    '''
    image 중 불필요한 내용 제거
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
    for item in tqdm(directories):
        if item == "RGB05":
            continue

        ## 디렉토리 내 이미지 파일 정렬
        frames = [i for i in os.listdir(f"dataset/{item}") if ".jpg" in i]
        sorted_frame = sorted(frames, key = file_number)

        item_number = int(item[-2:]) # 디렉토리로부터 숫자 추출
        ## RGB3, RGB5 ~ RGB12, RGB15, RGB18은 형광등 -> LED 정면 -> LED 좌측 순으로 촬영됨
        if (5 < item_number and item_number < 13 and item_number != 7) or item_number == 3 or item_number == 15 or item_number == 18 or (item_number > 0 and item_number < 4):
            for idx, elem in enumerate(sorted_frame, start = 1):
                os.rename(f"dataset/{item}/{elem}", f"dataset/{item}/{item_number}RGB{idx}.jpg") # 이름 변경
                # image_crop_and_resize(f"dataset/{item}/{item_number}RGB{idx}.jpg")


        ## RGB7, RGB13~는 LED 좌측 -> LED 정면 -> 형광등 순으로 촬영됨
        elif item_number == 7 or item_number > 12 or item_number == 4:
            file_order = 0
            for idx, elem in enumerate(sorted_frame, start = 1):
                if (idx == 1):
                    file_order = 45

                elif (idx == 23):
                    file_order = 23

                elif (idx == 45):
                    file_order = 1

                os.rename(f"dataset/{item}/{elem}", f"dataset/{item}/{item_number}RGB{file_order}.jpg")
                # image_crop_and_resize(f"dataset/{item}/{item_number}RGB{file_order}.jpg")
                file_order += 1

### Junsung
def image_crop_with_mtcnn():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    mtcnn = MTCNN(
        image_size = 256, margin = 0, min_face_size = 20,
        thresholds = [0.6, 0.7, 0.7], factor = 0.709,
        post_process = True, device = device
    )

    for directory in os.listdir("dataset/"):
        if "RGB" not in directory:
            continue

        for file in os.listdir(f"dataset/{directory}"):
            if ".jpg" not in file:
                continue

            image = cv2.imread(f"dataset/{directory}/{file}")

            mtcnn(image, save_path = f"dataset/rgb/{file}")

def main():
    # if len(os.listdir("dataset/hsi")) < 50 and len(os.listdir("dataset/rgb")) < 50:
    # modify_image()
    # rgb_rename_crop_resize()

    # hsi_dir = [i for i in os.listdir("dataset/hsi") if "HSI" in i]
    # template_matching(hsi_dir)
    pass    

if __name__ == "__main__":
    for directory in os.listdir("dataset/spoof"):
        if 'hsi' in directory:
            for img in os.listdir(f"dataset/spoof/{directory}"):
                if ".bmp" not in img:
                    continue

                image = cv2.imread(f"dataset/spoof/{directory}/{img}")

                ## 180도 회전
                h, w = image.shape[:2]
                cX, cY = w // 2, h // 2
                M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
                image = cv2.warpAffine(image, M, (w, h))

                cv2.imwrite(f"dataset/spoof/{directory}/{img}", image)