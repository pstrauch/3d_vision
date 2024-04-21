import zipfile

# Replace 'path_to_zip_file.zip' with the path to your zip file
# Replace 'npy_file.npy' with the file name inside the zip you want to extract
with zipfile.ZipFile('/Users/dennisbaumann/3d_vision/Dataset.zip', 'r') as zip_ref:
    zip_ref.extract('Dataset.npy', '/Users/dennisbaumann/3d_vision/')

