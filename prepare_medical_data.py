import os
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread


def get_img_files(
    parent_folder,
    extensions=(
        ".bmp",
        ".dib",
        ".png",
        ".jpg",
        ".jpeg",
        ".pbm",
        ".pgm",
        ".ppm",
        ".tif",
        ".tiff",
        ".npy",
    ),
):
    """Retrieve all image files from a given directory and its subdirectories."""
    img_files = []
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(extensions):
                img_files.append(os.path.join(root, file))
    return img_files


def rgb2y(img):
    y = (
        img[0:1, :, :] * 0.299000
        + img[1:2, :, :] * 0.587000
        + img[2:3, :, :] * 0.114000
    )
    return y


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0 : endw - win + 0 + 1 : stride, 0 : endh - win + 0 + 1 : stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[
                :, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride
            ]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def is_low_contrast(
    image, fraction_threshold=0.1, lower_percentile=10, upper_percentile=90
):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


data_name = "Data4Enhance"
img_size = 128  # patch size
stride = 200  # patch stride

# Define folders
MRI_folders = [
    "MedicalImages/MRI_CT/MRI",
    "MedicalImages/MRI_PET/MRI",
    "MedicalImages/MRI_SPECT/MRI",
]

IR_folders = [
    "MedicalImages/MRI_CT/CT",
    "MedicalImages/MRI_PET/PET",
    "MedicalImages/MRI_SPECT/SPECT",
]

# Get file lists
MRI_files = sorted([f for folder in MRI_folders for f in get_img_files(folder)])
IR_files = sorted([f for folder in IR_folders for f in get_img_files(folder)])

assert len(MRI_files) == len(IR_files)

# Create HDF5 file
h5f = h5py.File(
    os.path.join(
        "./data",
        data_name + "_imgsize_" + str(img_size) + "_stride_" + str(stride) + ".h5",
    ),
    "w",
)
h5_vis = h5f.create_group("vis_patchs")
h5_ir = h5f.create_group("ir_patchs")
train_num = 0

for i in tqdm(range(len(MRI_files))):
    try:
        I_MRI = (
            imread(MRI_files[i], as_gray=True).astype(np.float32)[None, :, :] / 255.0
        )  # [1, H, W] Uint8->float32
        I_IR = (
            imread(IR_files[i], as_gray=True).astype(np.float32)[None, :, :] / 255.0
        )  # [1, H, W] Float32
        # crop
        I_MRI_Patch_Group = Im2Patch(I_MRI, img_size, stride)
        I_IR_Patch_Group = Im2Patch(I_IR, img_size, stride)
    except Exception as e:
        print(f"Error processing file {MRI_files[i]}: {e}")
        continue

    for ii in range(I_MRI_Patch_Group.shape[-1]):
        bad_MRI = is_low_contrast(I_MRI_Patch_Group[0, :, :, ii])
        bad_IR = is_low_contrast(I_IR_Patch_Group[0, :, :, ii])
        # Determine if the contrast is low
        if not (bad_MRI or bad_IR):
            avl_MRI = I_MRI_Patch_Group[0, :, :, ii]  # available MRI (for vi)
            avl_IR = I_IR_Patch_Group[0, :, :, ii]  # available IR
            avl_MRI = avl_MRI[None, ...]
            avl_IR = avl_IR[None, ...]

            h5_vis.create_dataset(
                str(train_num), data=avl_MRI, dtype=avl_MRI.dtype, shape=avl_MRI.shape
            )
            h5_ir.create_dataset(
                str(train_num), data=avl_IR, dtype=avl_IR.dtype, shape=avl_IR.shape
            )
            train_num += 1

print(f"Success, total = {train_num}")
h5f.close()

with h5py.File(
    os.path.join(
        "data",
        data_name + "_imgsize_" + str(img_size) + "_stride_" + str(stride) + ".h5",
    ),
    "r",
) as f:
    for key in f.keys():
        print(f[key], key, f[key].name)
