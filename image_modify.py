import os, re, json, cv2

from tqdm import tqdm

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
    }

    ## Load image files
    video_and_exp_dir = [i for i in os.listdir("raw-dataset") if f"HSI{exp_dir[i]}" not in os.listdir("dataset") or f"RGB{rgb_dir[i]}" not in os.listdir("dataset") or f"RGB{rgb_dir[i]}-need-edit" not in os.listdir("dataset")]

    for item in tqdm(video_and_exp_dir):
        ## hsi image인 경우
        if "exp" in item:
            for elem in tqdm(f"raw-dataset/{item}"):
                ## 예외 처리: 해당 디렉토리 내 bmp 형식이 아닌 파일이 있는 경우
                if elem[-3:] != "bmp":
                    continue

                ## 이미지 파일 읽기
                image = cv2.imread(f"raw-dataest/{item}/{elem}")

                ## 180도 회전
                h, w, _ = image.shape
                cX, cY = w // 2, h // 2
                M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
                image = cv2.warpAffine(image, M, (w, h))

                ## 디렉토리 생성 및 이미지 저장
                os.makedirs(f"dataset/HSI{exp_dir[item]}", exist_ok = True)
                cv2.imwrite(f"dataset/HSI{exp_dir[item]}/{exp_dir[item]}HSI{elem[:-4]}.bmp")

        ## rgb image인 경우
        elif "WIN" in item:
            ## 디렉토리 생성
            os.makedirs(f"dataset/RGB{rgb_dir[item]}-need-clean", exist_ok = True)

            ## 비디오 파일 읽기
            video = cv2.VideoCapture(f"raw-dataset/{item}")
            success, image = video.read()

            ## 비디오 -> 이미지 변환
            frame = 1
            while success:
                ## 6 frame 단위로 정제
                if not (frame % 6):
                    cv2.imwrite(f"dataset/RGB{rgb_dir[item]}/{rgb_dir[item]}RGB{frame // 6}.jpg", image)

                success, image = video.read()
                frame += 1

            ## video 해제
            video.release()

        ## 예외 처리: 해당 디렉토리 내 bmp 또는 mp.4 형식이 아닌 파일이 있는 경우
        else:
            continue

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
    directories = [i for i in os.listdir("dataset") if ("RGB" in i and "-need-edit" not in i)]
    directories.sort()

    ## 디렉토리 내 이미지 파일 정렬, 자르기, 크기 조절, 이름 변경 수행
    for item in directories:
        ## 디렉토리 내 이미지 파일 정렬
        frames = [i for i in os.listdir(f"dataset/{item}")]
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

def text_file_generate():
    pass