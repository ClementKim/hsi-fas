import os
import cv2

from tqdm import tqdm

# will remove this function after the dataset is fixed
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
    }

    lst = [i for i in os.listdir("dataset") if (("exp" in i) and (f"{exp_dir[i]}HSI" not in os.listdir(f"dataset/{i}")[0])) or (("WIN" in i) and (f"RGB{rgb_dir[i]}" not in os.listdir("dataset/{i}")[0]))]

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
            if not (count % 3):
                cv2.imwrite(f"{target_path}/{source[:-4]}_{count}.jpg", image)

            success, image = video.read()
            count += 1

        video.release()

def main():
    #new_hsi_rotate()
    #new_video_to_image()
    rename_dir()
    #old_video_to_image()
    

if __name__ == "__main__":
    main()