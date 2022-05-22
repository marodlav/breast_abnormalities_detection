import os
import pydicom as dicom
import cv2
import numpy as np
from skimage.util import img_as_ubyte
import numpy as np
from aux_functions.aux_functions import change_extension, delete_file, get_extension


def change_to_png(upload_file_path):
    dicom_img = dicom.dcmread(upload_file_path).pixel_array
    new_file_path = change_extension(upload_file_path)

    cv2.imwrite(new_file_path, dicom_img)
    #delete_file(upload_file_path)

    img = cv2.imread(new_file_path)

    return new_file_path, img


def open_image(upload_file_path, extension):
    if extension == "dcm":
        upload_file_path, img = change_to_png(upload_file_path)
    else:
        img = cv2.imread(upload_file_path)

    img_original_shape = img.shape

    return upload_file_path, img, img_original_shape


def resize_image(img):
    return cv2.resize(img, (320, 640))


def normalizate_image(img_array):
    norm_img_array = (img_array - img_array.min()) / \
        (img_array.max() - img_array.min())
    return norm_img_array


def check_rigth_or_left(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 150])
    upper = np.array([150, 255, 255])
    mask = cv2.inRange(hsv_image, lower, upper)

    x, y, w, h = 0, 0, img.shape[1]//2, img.shape[0]
    left = mask[y:y+h, x:x+w]
    right = mask[y:y+h, x+w:x+w+w]

    left_sum = cv2.countNonZero(left)
    right_sum = cv2.countNonZero(right)

    if left_sum > right_sum:
        left_image = True
    else:
        left_image = False

    return left_image


def padding_image(img):
    is_left = check_rigth_or_left(img)
    black_image = np.zeros(img.shape, dtype="uint8")
    if is_left:
        result = np.concatenate((img, black_image), axis=1)
    else:
        result = np.concatenate((black_image, img), axis=1)

    return result


def min_max_normalise(img):
    norm_img = (img - img.min()) / (img.max() - img.min())

    return norm_img


def clahe(img, clip=2.0, tile=(8, 8)):
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)

    R, G, B = cv2.split(img)
    output1_R = clahe_create.apply(R)
    output1_G = clahe_create.apply(G)
    output1_B = clahe_create.apply(B)

    clahe_img = cv2.merge((output1_R, output1_G, output1_B))

    return clahe_img


def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion_image = cv2.erode(img, kernel)
    return erosion_image


def remove_noise(img_array):
    norm = img_as_ubyte(min_max_normalise(img_array))
    gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_array)

    cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), -1)

    new_image = cv2.bitwise_and(img_array, mask)

    return new_image


def process(upload_file_path):
    extension = get_extension(upload_file_path)

    upload_file_path, img, img_original_shape = open_image(
        upload_file_path, extension)
    img_resize = resize_image(img)
    no_noise_img = remove_noise(img_resize)
    clahe_image = clahe(no_noise_img)
    erosion_image = erosion(clahe_image)
    pad_image = padding_image(erosion_image)
    pad_image = padding_image(erosion_image)
    pad_original_image = padding_image(img_resize)

    return pad_image, pad_original_image, img_original_shape
