
from PIL import Image
import os
from pathlib import Path
import cv2
import numpy as np
from skimage.util import img_as_ubyte
import sys


def open_image(curdir, f):
    # Opens an image with cv2 in array format
    image_dir = os.path.join(curdir, f)
    img = cv2.imread(image_dir)

    return img


def crop_image(img):
    # Crop the image by 1% on the sides and 4% on high and low.
    width, height, _ = img.shape
    left = round(width*0.01)
    top = round(height*0.04)
    right = round(width - width*0.01)
    bottom = round(height - height*0.04)

    cropped_image = img[left:right, top:bottom]

    return cropped_image


def resize_image(img):
    # Resizes the image to 320x640
    return cv2.resize(img, (320, 640))


def get_full_image(curdir, f):
    # From a mask the function finds its corresponding mammogram
    f_full = f[0:len(f)-10] + "FULL.png"
    full_dir = os.path.join(curdir, f_full)
    full = cv2.imread(full_dir)
    return full


def check_rigth_or_left(img):
    # Check whether the breast is on the left or right side of the mammogram.
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 100])
    upper = np.array([100, 255, 255])
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


def padding_image(img, curdir, f):
    # Depending on whether the breast is on the left or right side,
    # the image is filled with black until it is converted into a picture
    # with dimension 640x640
    if "FULL" in f:
        is_left = check_rigth_or_left(img)
    else:
        full = get_full_image(curdir, f)
        is_left = check_rigth_or_left(full)

    black_image = np.zeros(img.shape, dtype="uint8")

    if is_left:
        result = np.concatenate((img, black_image), axis=1)
    else:
        result = np.concatenate((black_image, img), axis=1)

    return result


def min_max_normalise(img):
    # Normalises the values of an image
    norm_img = (img - img.min()) / (img.max() - img.min())

    return norm_img


def clahe(img, clip=2.0, tile=(8, 8)):
    # Applies the CLAHE function to the picture
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
    # Apply the morphological erosion function to the image
    kernel = np.ones((5, 5), np.uint8)
    erosion_image = cv2.erode(img, kernel)

    return erosion_image


def write_image(img, new_dir, f):
    # Write the image in the indicated place
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite((os.path.join(new_dir, f)), img)


def remove_noise(img_array):
    # Remove everything from the image that is not the breast and erase the isolated dots
    norm = img_as_ubyte(min_max_normalise(img_array))
    gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_array)

    cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), -1)

    new_image = cv2.bitwise_and(img_array, mask)

    return new_image


def generate_dir(file, set, mask_files, curdir):
    # Create the directories for the separation of data in training, validation and test
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
    # Main function that is responsible for creating training, validation and test data
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
    # Performs a vertical flip of the image
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
    # Performs a horizontal flip of the image
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
    # Performs a 90-degree rotation of the picture
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
    # Main function to create more images for training.
    mass = "Calc"
    train = "Train"
    dir_images = os.path.join(os.path.join(dir, mass), train)
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


def save_image_enviroment(curdir, dest_dir, f, image_set):
    # Save images according to whether they are training, validation or test images.
    image = Image.open(os.path.join(curdir, f))
    new_dir = os.path.join(dest_dir, image_set.lower())
    Path(new_dir).mkdir(parents=True, exist_ok=True)
    image.save(os.path.join(new_dir, f))


def move_images(dir, output_dir):
    # Move all images that will be useful for model training to
    # the final folder where they will be processed
    train = "Train"
    test = "Test"
    val = "Val"
    output_dir = os.path.join(output_dir, "images")
    for (curdir, _, files) in os.walk(dir, topdown=False):
        for f in files:
            if "FULL" in f:
                if train in curdir:
                    save_image_enviroment(curdir, output_dir, f, train)
                elif test in curdir:
                    save_image_enviroment(curdir, output_dir, f, test)
                elif val in curdir:
                    save_image_enviroment(curdir, output_dir, f, val)


def normalize_image_yolo(max_w, max_h, x, y, w, h):
    # Performs the necessary normalisations to the YOLOv5 annotations.
    x = x/max_w
    w = w/max_w
    y = y/max_h
    h = h/max_h

    return (x, y, w, h)


def getSizeRectangle(dir):
    # Collects the position of the minimum rectangle containing the anomaly.
    img = cv2.imread(dir)
    width, height, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.boundingRect(max_contour)

    (x, y, w, h) = rect
    x_center = (x + (x + w)) / 2
    y_center = (y + (y + h)) / 2
    rect_norm = normalize_image_yolo(width, height, x_center, y_center, w, h)

    return rect_norm


def getAnnotations(dir_images, output_dir):
    # Create annotations for YOLO
    dir_labels = os.path.join(output_dir, "labels")
    for (curdir, _, files) in os.walk(dir_images, topdown=False):
        for f in files:
            label = '0'
            image_set = "Train"
            if "_MASK" in f:
                if "Mass" in curdir:
                    label = '1'
                if "Test" in curdir:
                    image_set = "Test"
                elif "Val" in curdir:
                    image_set = "Val"

                try:
                    (x, y, w, h) = getSizeRectangle(os.path.join(curdir, f))
                    full_file = f[0:len(f)-10] + "FULL.txt"
                    new_dir = os.path.join(dir_labels, image_set.lower())
                    Path(new_dir).mkdir(parents=True, exist_ok=True)
                    # if "Mass" in curdir:
                    with open(os.path.join(new_dir, full_file), "a", encoding="utf-8") as fw:
                        fw.write(label + " " + str(x) + " " + str(y) +
                                 " " + str(w) + " " + str(h) + "\n")
                except Exception:
                    print("Doesn't exit abnormalities in " + f)


def process(dir="", dest_dir=""):
    # Principal function that transform all the images
    for (curdir, _, files) in os.walk(dir, topdown=False):
        for f in files:
            split_dir = os.path.split(curdir)
            new_dir = os.path.join(dest_dir, split_dir[-1])

            img = open_image(curdir, f)
            cropped_img = crop_image(img)
            new_image = resize_image(cropped_img)

            if "FULL" in f:
                no_noise_img = remove_noise(new_image)
                clahe_image = clahe(no_noise_img)
                new_image = erosion(clahe_image)

            pad_image = padding_image(new_image, curdir, f)
            write_image(pad_image, new_dir, f)


if __name__ == '__main__':
    dir = ""
    output_dir = ""
    yolo_images_dir = ""

    if len(sys.argv) > 4:
        raise Exception("Too many arguments")
    if len(sys.argv) == 1:
        raise Exception("At least the YOLO's image directory is needed")

    output_yolo, *args = sys.argv[1:]
    process(*args)
    create_val(args[1])
    augmentation(args[1])
    move_images(args[1], output_yolo)
    getAnnotations(args[1], output_yolo)
