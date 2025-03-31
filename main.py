import os
import json
import cv2

from tqdm import tqdm

def rename_dir():
    exp_dir = {
        "exp_rotated": "04",
        "exp0_rotated": "02",
        "exp1_rotated": "07",
        "exp2_rotated": "06",
        "exp3_rotated": "05",
        "exp4_rotated": "09",
        "exp5_rotated": "08",
        "exp6_rotated": "03",
        "exp7_rotated": "01",
        "exp8_rotated": "10",
        "exp9_rotated": "11",
        "exp10_rotated": "17",
        "exp11_rotated": "15",
        "exp12_rotated": "12",
        "exp13_rotated": "18",
        "exp14_rotated": "13",
        "exp15_rotated": "16",
        "exp16_rotated": "14",
        "exp17_rotated": "20",
        "exp18_rotated": "19",
        "exp19_rotated": "21",
        "exp20_rotated": "22",
        "exp22_rotated": "26",
        "exp23_rotated": "24",
        "exp24_rotated": "25",
        "exp25_rotated": "23",
        "exp26_rotated": "27",
        "exp27_rotated": "28",
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
    }

    lst = [i for i in os.listdir("dataset") if (("exp" in i) and (f"{exp_dir[i]}HSI" not in os.listdir(f"dataset/{i}")[0])) or (("WIN" in i) and (f"RGB{rgb_dir[i]}" not in os.listdir(f"dataset/{i}")[0]))]

    for i in lst:
        if ("exp" in i):
            elems = [i for i in os.listdir("dataset/" + i) if ".bmp" in i]

            for hsi in elems:
                if exp_dir[i] + "HSI" in hsi:
                    continue

                os.rename(f"dataset/{i}/{hsi}", f"dataset/{i}/{exp_dir[i]}HSI{hsi}")

        elif ("WIN" in i):
            os.rename(f"dataset/{i}", f"dataset/RGB{rgb_dir[i]}")

def old_video_to_image():
    lst = [i for i in os.listdir("raw-dataset")]
    
    for directory in lst:
        dir_path = os.path.join("dataset", directory)
        os.makedirs(dir_path, exist_ok=True)

        for file_type in os.listdir("raw-dataset/" + directory):
            file_type_path = os.path.join(dir_path, file_type)
            os.makedirs(file_type_path, exist_ok=True)

            source_path = os.path.join("raw-dataset", directory, file_type)
            target_path = os.path.join("dataset", directory, file_type)

            for elem in tqdm(os.listdir(source_path)):
                video = cv2.VideoCapture(f"{source_path}/{elem}")
                success, image = video.read()

                count = 1
                while success:
                    if not (count % 3):
                        if elem[-4:] in [".avi", ".mp4"]:
                            img_ext = ".bmp" if elem[-4:] == ".avi" else ".jpg"

                            if img_ext == ".bmp":
                                h, w = image.shape[:2]
                                cX, cY = w // 2, h // 2
                                M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
                                image = cv2.warpAffine(image, M, (w, h))
                                
                            cv2.imwrite(f"{target_path}/{elem[:-4]}_{count}{img_ext}", image)

                    success, image = video.read()
                    count += 1

                video.release()

def new_hsi_rotate():
    lst = [i for i in os.listdir("raw-dataset") if "exp" in i and i + "_rotated" not in os.listdir("dataset")]

    for source in lst:
        source_path = os.path.join("raw-dataset", source)
        target_path = os.path.join("dataset", source + "_rotated")

        os.makedirs(target_path, exist_ok=True)

        for elem in tqdm(os.listdir(source_path)):
            image = cv2.imread(f"{source_path}/{elem}")
            h, w = image.shape[:2]
            cX, cY = w // 2, h // 2
            M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            cv2.imwrite(f"{target_path}/{elem}", image)

def new_video_to_image():
    lst = [i for i in os.listdir("raw-dataset") if "WIN" in i and i[:-4] not in os.listdir("dataset")]
    
    for source in tqdm(lst):
        source_path = os.path.join("raw-dataset", source)
        target_path = os.path.join("dataset", source[:-4])

        os.makedirs(target_path, exist_ok=True)

        video = cv2.VideoCapture(source_path)
        success, image = video.read()

        count = 1
        while success:
            if not (count % 6):
                cv2.imwrite(f"{target_path}/{source[:-4]}_{count // 6}.jpg", image)

            success, image = video.read()
            count += 1

        video.release()

def real_text_gen():
    additional_info_dictionary = {
        # id : [light, pose, Expression, Acc.]
        "01" : ["Fluorescent", "Straight", "Neutral", "N/A"],
        "02" : ["Fluorescent", "Left", "Neutral", "N/A"],
        "03" : ["Fluorescent", "Up", "Neutral", "N/A"],
        "04" : ["Fluorescent", "Right", "Neutral", "N/A"],
        "05" : ["Fluorescent", "Down", "Neutral", "N/A"],
        "06" : ["Fluorescent", "Straight", "Closed Eyes", "N/A"],
        "07" : ["Fluorescent", "Straight", "Smile", "N/A"],
        "08" : ["Fluorescent", "Straight", "Neutral", "Mask"],
        "09" : ["Fluorescent", "Left", "Neutral", "Mask"],
        "10" : ["Fluorescent", "Up", "Neutral", "Mask"],
        "11" : ["Fluorescent", "Right", "Neutral", "Mask"],
        "12" : ["Fluorescent", "Down", "Neutral", "Mask"],
        "13" : ["Fluorescent", "Straight", "Neutral", "Hat"],
        "14" : ["Fluorescent", "Left", "Neutral", "Hat"],
        "15" : ["Fluorescent", "Up", "Neutral", "Hat"],
        "16" : ["Fluorescent", "Right", "Neutral", "Hat"],
        "17" : ["Fluorescent", "Down", "Neutral", "Hat"],
        "18" : ["Fluorescent", "Straight", "Neutral", "Sunglasses"],
        "19" : ["Fluorescent", "Left", "Neutral", "Sunglasses"],
        "20" : ["Fluorescent", "Up", "Neutral", "Sunglasses"],
        "21" : ["Fluorescent", "Right", "Neutral", "Sunglasses"],
        "22" : ["Fluorescent", "Down", "Neutral", "Sunglasses"],
        "23" : ["LED", "Straight", "Neutral", "N/A"],
        "24" : ["LED", "Left", "Neutral", "N/A"],
        "25" : ["LED", "Up", "Neutral", "N/A"],
        "26" : ["LED", "Right", "Neutral", "N/A"],
        "27" : ["LED", "Down", "Neutral", "N/A"],
        "28" : ["LED", "Straight", "Closed Eyes", "N/A"],
        "29" : ["LED", "Straight", "Smile", "N/A"],
        "30" : ["LED", "Straight", "Neutral", "Mask"],
        "31" : ["LED", "Left", "Neutral", "Mask"],
        "32" : ["LED", "Up", "Neutral", "Mask"],
        "33" : ["LED", "Right", "Neutral", "Mask"],
        "34" : ["LED", "Down", "Neutral", "Mask"],
        "35" : ["LED", "Straight", "Neutral", "Hat"],
        "36" : ["LED", "Left", "Neutral", "Hat"],
        "37" : ["LED", "Up", "Neutral", "Hat"],
        "38" : ["LED", "Right", "Neutral", "Hat"],
        "39" : ["LED", "Down", "Neutral", "Hat"],
        "40" : ["LED", "Straight", "Neutral", "Sunglasses"],
        "41" : ["LED", "Left", "Neutral", "Sunglasses"],
        "42" : ["LED", "Up", "Neutral", "Sunglasses"],
        "43" : ["LED", "Right", "Neutral", "Sunglasses"],
        "44" : ["LED", "Down", "Neutral", "Sunglasses"],
        "45" : ["LED", "Straight", "Neutral", "N/A"],
        "46" : ["LED", "Left", "Neutral", "N/A"],
        "47" : ["LED", "Up", "Neutral", "N/A"],
        "48" : ["LED", "Right", "Neutral", "N/A"],
        "49" : ["LED", "Down", "Neutral", "N/A"],
        "50" : ["LED", "Straight", "Closed Eyes", "N/A"],
        "51" : ["LED", "Straight", "Smile", "N/A"],
        "52" : ["LED", "Straight", "Neutral", "Mask"],
        "53" : ["LED", "Left", "Neutral", "Mask"],
        "54" : ["LED", "Up", "Neutral", "Mask"],
        "55" : ["LED", "Right", "Neutral", "Mask"],
        "56" : ["LED", "Down", "Neutral", "Mask"],
        "57" : ["LED", "Straight", "Neutral", "Hat"],
        "58" : ["LED", "Left", "Neutral", "Hat"],
        "59" : ["LED", "Up", "Neutral", "Hat"],
        "60" : ["LED", "Right", "Neutral", "Hat"],
        "61" : ["LED", "Down", "Neutral", "Hat"],
        "62" : ["LED", "Straight", "Neutral", "Sunglasses"],
        "63" : ["LED", "Left", "Neutral", "Sunglasses"],
        "64" : ["LED", "Up", "Neutral", "Sunglasses"],
        "65" : ["LED", "Right", "Neutral", "Sunglasses"],
        "66" : ["LED", "Down", "Neutral", "Sunglasses"],
    }
    
    os.makedirs("dataset/text/real", exist_ok=True)

    target_list = [i[-2:] for i in os.listdir("dataset/hsi") if i not in os.listdir("dataset/text/real")]
    target_list.sort()

    assert all([len(os.listdir(f"dataset/rgb/{i}")) == 66 for i in os.listdir("dataset/rgb")]), "run this function later"

    for identification in target_list:
        os.mkdir(f"dataset/text/real/{identification}")
        
        for file_order in range(1, 67):
            file_dictionary = {
                "id" : identification,
                "label" : "real",
                "RGB" : f"dataset/rgb/RGB{identification}/{identification}RGB{file_order}.jpg",
                "HSI" : f"dataset/hsi/HSI{identification}/{identification}HSI{file_order}.bmp",
                # "Signal" : f"dataset/signal/{identification}/", ## 충분한 논의 필요
                "Lighting" : additional_info_dictionary[file_order][0],
                "Pose" : additional_info_dictionary[file_order][1],
                "Expression" : additional_info_dictionary[file_dictionary][2],
                "Accessory" : additional_info_dictionary[file_dictionary][3],
                "Spoof Type" : "N/A"
            }

            # 우선 txt로 저장하는 방식을 선택했지만 json으로 변환 가능하며 dictionary 자체 저장 파일로 변환 가능함
            with open(f"{identification}.txt", 'w', encoding = 'UTF-8') as f:
                f.write("{")
                for key, val in additional_info_dictionary.items():
                    f.write(f"    {key} : {val}")
                f.write("}")
                f.close()


def main():
    # these functions will be removed after dataset build is done
    new_hsi_rotate() ## hsi bmp 파일 회전하는 함수
    new_video_to_image() ## rgb video 파일 jpg로 변환하는 함수
    rename_dir() ## bmp, jpg 파일 이름 변경하는 함수
    # old_video_to_image() ## hsi video 파일 bmp 파일로 변환하는 함수
    # real_text_gen() ## text 데이터를 생성하는 함수, Test this function after completing data clean
    

if __name__ == "__main__":
    main()
