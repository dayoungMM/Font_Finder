#-*- coding: utf-8 -*- 
import os
import pathlib
import glob
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 Path Pattern에 일치하는 파일 얻기
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # 파일 경로
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대경로):{fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지파일(파일명):{filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        
        
        if image is None:
            print(f"이미지파일({fullpath})을 읽을 수 없습니다.")
            continue
        name_images.append((filename, image))
        
    return name_images

def detect_image_face(file_path, image):
    # image = np.resize(np.array(image), (76, 76))
    image = cv2.resize(image, (76,76))
    # image = image.resize((76,76), refcheck=False)
    sobelx, sobely, laplacian = extract_edge(image)
    path = pathlib.Path(file_path)
    directory = str(path.parent.resolve())
    filename = path.stem
    extension = path.suffix

    # save_name_sobelx = f"{filename}_test_sobelx{extension}"
    # save_name_sobely = f"{filename}_test_sobely{extension}"
    save_name_laplacian = f"{filename}_test_laplacian{extension}"

    # output_path_sobelx = os.path.join(directory,save_name_sobelx)
    # output_path_sobely = os.path.join(directory,save_name_sobely)
    output_path_laplacian = os.path.join(directory,save_name_laplacian)

    # cv2.imwrite(output_path_sobelx, sobelx)
    # cv2.imwrite(output_path_sobely, sobely)
    cv2.imwrite(output_path_laplacian, laplacian)

    return RETURN_SUCCESS

# def delete_dir(dir_path, is_delete_top_dir=True):
#     for root, dirs, files in os.walk(dir_path, topdown=False):
#         for name in files:
#             os.remove(os.path.join(root, name))
#         for name in dirs:
#             os.rmdir(os.path.join(root, name))
#     if is_delete_top_dir:
#         os.rmdir(dir_path)
        
def extract_edge(img):
    sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    laplacian = cv2.Laplacian(img, cv2.CV_8U)
    return sobelx, sobely, laplacian
    

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./testset/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./training/laplacian"

def main():
    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내의 파일 제거
    # delete_dir(OUTPUT_IMAGE_DIR, False)

    # 이미지 파일 읽기
    name_images =load_name_images(IMAGE_PATH_PATTERN)
    
    for name_image in name_images:
        file_path = os.path.join(OUTPUT_IMAGE_DIR,f"{name_image[0]}")
        image = name_image[1]
      
        detect_image_face(file_path, image)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()