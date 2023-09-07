"""
Useful raw functions for processing and validation
"""
import zipfile
import os
import shutil


def characters() -> list:
    letters = [chr(i) for i in list(range(97, 123)) + list(range(65, 91))]
    underscore = ["_"]
    digits = [str(k) for k in list(range(10))]
    return letters + underscore + digits


allowed_characters = characters()


def check_allowed_characters(value):
    for letter in value:
        if letter not in allowed_characters:
            return False

    return True


is_valid_character = check_allowed_characters


def extract_zip(save_as_name, zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        folder_name = str(zip_path).split("/")[-1].split(".")[0] + "/"
        if not os.path.exists(extract_to + save_as_name):
            zip_ref.extractall(extract_to)
            os.rename(extract_to + folder_name, extract_to + save_as_name)
            return True
        else:
            return False


def remove_directory(folder_path):
    shutil.rmtree(folder_path)
    if not os.path.exists(folder_path):
        return True
    return False


if __name__ == "__main__":
    extract_zip(
        save_as_name="Another",
        zip_path="file.zip",
        extract_to="../media/",
    )
