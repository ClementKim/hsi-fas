import os
import cv2

from tqdm import tqdm

def rename_dir():
    dir_name = {
        "exp0_rotated": "02",
        "exp1_rotated": "07",
        "exp2_rotated": "06",
        "exp3_rotated": "05",
        "exp4_rotated": "09",
        "exp5_rotated": "08",
        "exp6_rotated": "03",
        "exp7_rotated": "01",
        "exp8_rotated": "10",
        "exp9_rotated": "11"
    }

    os.chdir("dataset")
    lst = [i for i in os.listdir() if "exp" in i]

    for i in lst:
        elems = os.listdir(i)

        for hsi in elems:
            if dir_name[i] + "HSI" in hsi:
                continue

            os.rename(f"{i}/{hsi}", f"{i}/{dir_name[i]}HSI{hsi}")

    os.chdir("..")

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
            cv2.imread(f"{target_path}/{elem}", image)

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
    #rename_dir()
    #old_video_to_image()
    new_video_to_image()
    new_hsi_rotate()

if __name__ == "__main__":
    main()