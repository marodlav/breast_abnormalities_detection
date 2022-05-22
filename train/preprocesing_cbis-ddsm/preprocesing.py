import os
import sys
import numpy as np
import pydicom as dicom
from utils import constants
import cv2
from pathlib import Path


def get_new_name(old_path):
    try:
        image = dicom.dcmread(old_path)
        patient_id = image.PatientID.replace(constants.DCM_EXTENSION, "")
        if patient_id[0] == 'P':  # Some test patiend_id haven't this prefix
            patient_id = "Calc-Test_" + patient_id

        # If the word "full" is in the path, it means that it's the complete mammography
        if "full" in old_path:
            new_name = patient_id + "_FULL" + constants.PNG_EXTENSION
        else:
            # Mask images only have two colors
            image_arr = image.pixel_array
            num_colors = len(np.unique(image_arr).tolist())
            if num_colors == 2:
                new_name = patient_id + "_MASK" + constants.PNG_EXTENSION
            else:
                # Discard crop images
                new_name = "discard"
        return new_name

    except Exception as e:
        print("Can't read dcm image")
        print(e)


def save_to_png(new_dir, new_name, source_file):
    try:
        new_dir = os.path.join(
            new_dir, "Calc") if "Calc" in source_file else os.path.join(new_dir, "Mass")
        final_dir = os.path.join(new_dir, new_name)
        Path(new_dir).mkdir(parents=True, exist_ok=True)

        image = dicom.read_file(source_file)
        image_arr = image.pixel_array
        cv2.imwrite(final_dir, image_arr)

    except Exception as e:
        print("Can't save the image as png.")
        print(e)


def main_preprocesing(dir="", new_dir=""):
    # Get the path to the folder that contains all the nested .dcm files.
    if dir == "":
        dir = os.getcwd()
    if new_dir == "":
        new_dir = os.path.join(os.path.dirname(
            dir), constants.PREPROCESSING_PNG_DIR)

    for (curdir, _, files) in os.walk(dir, topdown=False):
        for f in files:
            if f.endswith(constants.DCM_EXTENSION):
                old_name_path = os.path.join(curdir, f)
                new_name = get_new_name(old_name_path)
                if new_name != "discard":
                    save_to_png(new_dir, new_name, old_name_path)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        raise Exception("Too many arguments")

    print("Loading images...")
    main_preprocesing(*sys.argv)
    print("Finish")
