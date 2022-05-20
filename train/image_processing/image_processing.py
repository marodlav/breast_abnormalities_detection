from PIL import Image
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from skimage.util import img_as_ubyte

target_dir = "Resize640"


def cropImage(img):
    width, height = img.size   # Get dimensions
    left = width*0.01
    top = height*0.04
    right = width - width*0.01
    bottom = height - height*0.04
    cropped_image = img.crop((left, top, right, bottom))

    return cropped_image


def normalizate_image(img_array):
    norm_img_array = (img_array - img_array.min()) / \
        (img_array.max() - img_array.min())
    return norm_img_array


def write_image(img, new_dir, f):
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    img.save(os.path.join(new_dir, f))


def check_rigth_or_left(img):
    # Get number of rows and columns in the image.
    width, height = img.size
    x_center = width // 2

    img_array = normalizate_image(np.asarray(img))
    col_sum = img_array.sum(axis=0)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    if left_sum > right_sum:
        left_image = True
    else:
        left_image = False

    return left_image


def get_max_contour(contours):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours[0]


def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion_image = cv2.erode(img, kernel)
    return erosion_image


def removeNoise(dir):
    img_array = cv2.imread(dir)
    norm = img_as_ubyte(minMaxNormalise(img_array))
    gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_array)
    cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), -1)
    new_image = cv2.bitwise_and(img_array, mask)
    if "FULL" in dir:
        clahe_image = clahe(new_image)
        new_image = erosion(clahe_image)
    cv2.imwrite(dir, new_image)


def get_full_image(curdir, f):
    f_full = f[0:len(f)-10] + "FULL.png"
    full_dir = os.path.join(curdir, f_full)
    full = Image.open(full_dir)
    return full


def padding_image(img, curdir, f):
    width, height = img.size
    background_color = 0
    if "FULL" in f:
        is_left = check_rigth_or_left(img)
    else:
        full = get_full_image(curdir, f)
        is_left = check_rigth_or_left(full)
    result = Image.new(img.mode, (height, height), background_color)
    if is_left:
        result.paste(img, (0, 0))
        return result
    else:
        result.paste(img, (height - width, 0))
        return result


def minMaxNormalise(img):

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
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)

    R, G, B = cv2.split(img_uint8)
    output1_R = clahe_create.apply(R)
    output1_G = clahe_create.apply(G)
    output1_B = clahe_create.apply(B)

    clahe_img = cv2.merge((output1_R, output1_G, output1_B))

    return clahe_img


def normalize_image_yolo(max_w, max_h, x, y, w, h):
    x = x/max_w
    w = w/max_w
    y = y/max_h
    h = h/max_h

    return (x, y, w, h)


def getSizeRectangle(dir):
    img = cv2.imread(dir)
    width, height, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # not copying here will throw an error
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    # basically you can feed this rect into your classifier
    rect = cv2.boundingRect(max_contour)
    (x, y, w, h) = rect
    x_center = (x + (x + w)) / 2
    y_center = (y + (y + h)) / 2
    rect_norm = normalize_image_yolo(width, height, x_center, y_center, w, h)
    return rect_norm


def resizeImages(dir="", dest_dir=""):
    for (curdir, _, files) in os.walk(dir, topdown=False):
        for f in files:
            split_dir = os.path.split(curdir)
            new_dir = os.path.join(dest_dir, split_dir[-1])

            image_dir = os.path.join(curdir, f)
            image = Image.open(image_dir)
            cropped_image = cropImage(image)
            new_image = cropped_image.resize((320, 640))  # Divisible by 32
            pad_image = padding_image(new_image, curdir, f)
            write_image(pad_image, new_dir, f)
            removeNoise(os.path.join(new_dir, f))


def save_image(curdir, dest_dir, f, image_set):
    image = Image.open(os.path.join(curdir, f))
    new_dir = os.path.join(dest_dir, image_set.lower())
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    image.save(os.path.join(new_dir, f))


def move_images(dir, output_dir):
    train = "Train"
    test = "Test"
    val = "Val"
    output_dir = os.path.join(output_dir,"images")
    for (curdir, _, files) in os.walk(dir, topdown=False):
        for f in files:
            if "FULL" in f and "Mass" in curdir:
                if train in curdir:
                    save_image(curdir, output_dir, f, train)
                elif test in curdir:
                    save_image(curdir, output_dir, f, test)
                elif val in curdir:
                    save_image(curdir, output_dir, f, val)


def generate_dir(file, set, mask_files, curdir):
    prefix = file.replace("_FULL.png", "")
    rename_files = [x for x in mask_files if x.startswith(prefix)]
    rename_files.append(file)
    new_dir = os.path.join(curdir, set)
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    for r_file in rename_files:
        new_file = r_file.replace("Training", set).replace("Test", set)
        os.rename(os.path.join(curdir, r_file),
                  os.path.join(new_dir, new_file))


def create_val(dir):
    train = "Train"
    test = "Test"
    val = "Val"
    types_object = ["Calc", "Mass"]
    for type in types_object:
        dir_images = os.path.join(dir, type)
        for (curdir, _, files) in os.walk(dir_images, topdown=False):
            full_files = [f for f in files if "FULL" in f]
            mask_files = [m for m in files if "MASK" in m]
            #train_files, val_files = np.split(full_files, [int(len(full_files)*0.7)])
            train_files, val_files, test_files = np.split(
                np.array(full_files), [int(len(full_files)*0.7), int(len(full_files)*0.9)])
            for file in train_files:
                generate_dir(file, train, mask_files, curdir)
            for file in val_files:
                generate_dir(file, val, mask_files, curdir)
            for file in test_files:
                generate_dir(file, test, mask_files, curdir)


def flip_v(curdir, aug_file):
    if "MASK" in aug_file:
        num_char = -11
    else:
        num_char = -9
    image = Image.open(os.path.join(curdir, aug_file))
    type = aug_file[num_char:]
    new_file = aug_file[:num_char] + "_FLIPV" + type
    new_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    new_image.save(os.path.join(curdir, new_file))


def flip_h(curdir, aug_file):
    if "MASK" in aug_file:
        num_char = -11
    else:
        num_char = -9
    image = Image.open(os.path.join(curdir, aug_file))
    type = aug_file[num_char:]
    new_file = aug_file[:num_char] + "_FLIPH" + type
    new_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    new_image.save(os.path.join(curdir, new_file))


def rotate_90(curdir, aug_file):
    if "MASK" in aug_file:
        num_char = -11
    else:
        num_char = -9
    image = Image.open(os.path.join(curdir, aug_file))
    type = aug_file[num_char:]
    new_file = aug_file[:num_char] + "_ROTATE90" + type
    new_image = image.transpose(Image.ROTATE_90)
    new_image.save(os.path.join(curdir, new_file))


def augmentation(dir):
    train = "Train"
    dir_images = os.path.join(dir, train)
    for (curdir, _, files) in os.walk(dir_images, topdown=False):
        full_files = [f for f in files if "FULL" in f]
        mask_files = [m for m in files if "MASK" in m]
        for file in full_files:
            prefix = file.replace("_FULL.png", "")
            augmentation_files = [
                x for x in mask_files if x.startswith(prefix)]
            augmentation_files.append(file)
            for aug_file in augmentation_files:
                flip_h(curdir, aug_file)
                flip_v(curdir, aug_file)
                rotate_90(curdir, aug_file)


def getAnnotations(dir_images,output_dir):
    dir_labels = os.path.join(output_dir,"labels")
    for (curdir, _, files) in os.walk(dir_images, topdown=False):
        for f in files:
            label = '0'
            image_set = "Train"
            if "_MASK" in f:
                if "Mass" in curdir:
                    label = '0'
                if "Test" in curdir:
                    image_set = "Test"
                elif "Val" in curdir:
                    image_set = "Val"
                (x, y, w, h) = getSizeRectangle(os.path.join(curdir, f))
                full_file = f[0:len(f)-10] + "FULL.txt"
                new_dir = os.path.join(dir_labels, image_set.lower())
                Path(new_dir).mkdir(parents=True, exist_ok=True)
                if "Mass" in curdir:
                    with open(os.path.join(new_dir, full_file), "a", encoding="utf-8") as fw:
                        fw.write(label + " " + str(x) + " " + str(y) +
                                 " " + str(w) + " " + str(h) + "\n")


if __name__ == '__main__':
    dir = ""
    output_dir = ""
    yolo_images_dir = ""

    if len(sys.argv) > 4:
        raise Exception("Too many arguments")
    if len(sys.argv) == 1:
        raise Exception("At least the YOLO's image directory is needed")

    output_yolo, *args = sys.argv[1:]
    resizeImages(*args)
    create_val(args[0])
    augmentation(args[0])
    move_images(args[0],output_yolo)
    getAnnotations(args[0],output_yolo)
