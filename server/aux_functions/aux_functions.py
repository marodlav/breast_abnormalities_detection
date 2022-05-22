import os
import errors
import glob

def delete_file(path):
    # Delete uploaded and resulted file
    try:
        path_no_extension = os.path.splitext(path)[0]
        for file in glob.glob(path_no_extension + ".*"):
            os.remove(file)
    except OSError as e:
        print(errors.ERROR_DELETING_FILE, e)


def change_extension(file_path):
    return file_path.replace(".dcm", ".png")


def get_extension(file_path):
    filename = os.path.split(file_path)[-1]
    extension = filename.split(".")[1]

    return extension
