import os
import numpy as np
import torch
import SimpleITK as sitk
import cv2
from PIL import Image
from skimage import morphology, filters
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from monai.transforms import (
    ScaleIntensityRangePercentiles,
    ScaleIntensityRange,
    CropForeground,
    ResizeWithPadOrCrop,
    Resize,
)

def mk_dir(directory):
    """
    Creates a directory if it doesn't exist.
    :param directory: The path of the directory to create.
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)
        print(f"Success in making {directory}!")
    else:
        if os.listdir(directory):
            raise UserWarning(f"[ERROR] '{directory}' already exists and is not empty.")
        else:
            print(f"[WARNING] {directory} already exists but is empty.")

def show_images_horizontally(images, main_title=None, titles=['CT', 'MR', 'CT Label'], data_path=None):
    """
    A function to display 3 images horizontally.
    :param images: A list of image arrays (3 items).
    :param titles: Titles for each image (in list format), default is None.
    :param data_path: The path where the image will be saved.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmap = ['gray', 'gray', 'viridis']
    
    if main_title:
        fig.suptitle(main_title, fontsize=13)
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap=cmap[i])
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])

    fig.tight_layout()
    if data_path:
        fig.savefig(os.path.join(data_path, f'{main_title}.png'))
    plt.close(fig)

def load_data(ct, mri, ct_label, mr_label=None):
    """
    Load and preprocess data (CT, MRI, and their labels).
    """
    ct_img = sitk.GetArrayFromImage(sitk.ReadImage(ct)).astype(np.float32)
    mr_img = sitk.GetArrayFromImage(sitk.ReadImage(mri)).astype(np.float32)
    ct_lab_img = sitk.GetArrayFromImage(sitk.ReadImage(ct_label)).astype(np.float32)
    
    bounding_box = CropForeground(allow_smaller=True).compute_bounding_box(torch.Tensor(np.expand_dims(ct_img, axis=0)))
    ct_img = CropForeground(allow_smaller=True).crop_pad(torch.Tensor(np.expand_dims(ct_img, axis=0)), bounding_box[0], bounding_box[1])
    ct_img = ResizeWithPadOrCrop([ct_img.shape[1], 416, 416], mode="minimum")(ct_img)
    mr_img = CropForeground(allow_smaller=True).crop_pad(torch.Tensor(np.expand_dims(mr_img, axis=0)), bounding_box[0], bounding_box[1])
    mr_img = ResizeWithPadOrCrop([mr_img.shape[1], 416, 416], mode="minimum")(mr_img)
    ct_lab_img = CropForeground(allow_smaller=True).crop_pad(torch.Tensor(np.expand_dims(ct_lab_img, axis=0)), bounding_box[0], bounding_box[1])
    ct_lab_img = ResizeWithPadOrCrop([ct_lab_img.shape[1], 416, 416], mode="minimum")(ct_lab_img)
    
    ct_img = Resize([ct_img.shape[1], 512, 512], mode="trilinear")(ct_img)[0]
    mr_img = Resize([mr_img.shape[1], 512, 512], mode="trilinear")(mr_img)[0]
    ct_lab_img = Resize([ct_lab_img.shape[1], 512, 512], mode="nearest")(ct_lab_img)[0]
    
    if mr_label:
        mr_lab_img = sitk.GetArrayFromImage(sitk.ReadImage(mr_label)).astype(np.float32)
        mr_lab_img = CropForeground(allow_smaller=True).crop_pad(torch.Tensor(np.expand_dims(mr_lab_img, axis=0)), bounding_box[0], bounding_box[1])
        mr_lab_img = ResizeWithPadOrCrop([mr_lab_img.shape[1], 416, 416], mode="minimum")(mr_lab_img)
        mr_lab_img = Resize([mr_lab_img.shape[1], 512, 512], mode="nearest")(mr_lab_img)[0]
        return np.array(ct_img), np.array(mr_img), np.array(ct_lab_img), np.array(mr_lab_img)
    else:
        return np.array(ct_img), np.array(mr_img), np.array(ct_lab_img)

def preprocess_slices(slice_i, ct_img, ct_norm, ct_lab_img, threshold_ct):
    """
    Preprocess slices for CT and MR images.
    """
    for i in range(ct_img.shape[0]):
        if np.max(ct_lab_img[i,:,:]) == 0 or np.sum(ct_norm[i,:,:]<-0.99) > threshold_ct:
            continue
        slice_i.append(i)
    
    slice_i.sort()
    
    # Add surrounding slices(±10) where the lung is not present
    for i in range(slice_i[0] - 10, slice_i[0]):
        if i < 0 or np.sum(ct_norm[i,:,:]<-0.99) > threshold_ct:
            continue
        slice_i.append(i)
    for i in range(slice_i[-1] + 1, slice_i[-1] + 11):
        if i >= ct_img.shape[0] or np.sum(ct_norm[i,:,:]<-0.99) > threshold_ct:
            continue
        slice_i.append(i)
    
    return list(set(slice_i))

def save_images(ct_img, mr_img, ct_lab_img, mr_lab_img, slice_i, data_path, folder, type):
    """
    Save the processed images and slices.
    """
    mr_volnorm_3d = ScaleIntensityRangePercentiles(0, 100, 0, 1, clip=True)(mr_img) 
    ct_norm_3d = ScaleIntensityRange(-1024, 3071, -1, 1, clip=True)(ct_img)
    ct_lungnorm_3d = ScaleIntensityRange(-950, 350, -1, 1, clip=True)(ct_img)
    ct_lungnorm1024_3d = ScaleIntensityRange(-1024, 350, -1, 1, clip=True)(ct_img)
    for i in slice_i:
        ct_2d = ct_img[i,:,:]
        mr_2d = mr_img[i,:,:]
        ct_lab_2d = ct_lab_img[i,:,:]
        mr_volnorm = mr_volnorm_3d[i,:,:]
        ct_norm = ct_norm_3d[i,:,:]
        ct_lungnorm = ct_lungnorm_3d[i,:,:]
        ct_lungnorm1024 = ct_lungnorm1024_3d[i,:,:]

        mr_slicenorm = ScaleIntensityRangePercentiles(0, 100, 0, 1, clip=True)(mr_2d)

        # Create edge map
        image = (np.array(mr_slicenorm).copy() * 255).astype(np.uint8)
        mask = morphology.area_closing(morphology.area_opening(image > 50, area_threshold=700), area_threshold=20000)
        masked = filters.gaussian(image * mask, sigma=(1.2, 1.2), channel_axis=1)
        masked = (masked - np.min(masked)) / (np.max(masked) - np.min(masked))
        edge = cv2.Canny((masked * 255).astype(np.uint8), 40, 255)

        # Save the images
        sitk.WriteImage(sitk.GetImageFromArray(ct_2d), f'{data_path}/CT/{folder}_CT_{type}_s{i:03}.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(mr_2d), f'{data_path}/MR/{folder}_MR_{type}_s{i:03}.nii.gz')
        # Save CT norm
        sitk.WriteImage(sitk.GetImageFromArray(ct_norm), f'{data_path}/CT_norm/{folder}_CT_norm_{type}_s{i:03}.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(ct_lungnorm), f'{data_path}/CT_norm_lung/{folder}_CT_norm_lung_{type}_s{i:03}.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(ct_lungnorm1024), f'{data_path}/CT_norm_lung1024/{folder}_CT_norm_lung1024_{type}_s{i:03}.nii.gz')
        # Save MR norm
        sitk.WriteImage(sitk.GetImageFromArray(mr_volnorm), f'{data_path}/MR_norm_img/{folder}_MR_norm_img_{type}_s{i:03}.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(mr_slicenorm), f'{data_path}/MR_norm_slice/{folder}_MR_norm_slice_{type}_s{i:03}.nii.gz')
        # Save MR edge
        Image.fromarray(edge, 'L').save(f'{data_path}/MR_CannyEdgeMap/{folder}_MR_edge_{type}_s{i:03}.png')
        # Save label
        sitk.WriteImage(sitk.GetImageFromArray(ct_lab_2d), f'{data_path}/CT_label/{folder}_lung_{type}_s{i:03}.nii.gz')
        if mr_lab_img is not None:
            sitk.WriteImage(sitk.GetImageFromArray(mr_lab_img[i,:,:]), f'{data_path}/MR_label/{folder}_lung_mr_{type}_s{i:03}.nii.gz')

def process_dataset(root_path, data_path):
    """
    Main function to process the dataset and save the results.
    """
    mk_dir(data_path)
    for subfolder in ['CT', 'MR', 'CT_norm', 'CT_norm_lung', 'CT_norm_lung1024', 'MR_norm_img', 'MR_norm_slice', 'MR_CannyEdgeMap', 'CT_label', 'MR_label', 'first_last_img']:
        mk_dir(os.path.join(data_path, subfolder))

    bl_counts = 0
    fu_counts = 0
    counts = 0  # Total number of pairs where both CT and MR are present

    for folder in tqdm(os.listdir(root_path), position=0):
        for type in ['bl', 'fu']:
            path = os.path.join(root_path, folder)
            ct = f'{path}/{type}_img_CT_registered.nii.gz'
            mri = f'{path}/{type}_img_MRI.nii.gz'
            ct_lab = f'{path}/{type}_label_CT_registered.nii.gz'
            mr_lab = f'{path}/{type}_label_MRI.nii.gz'
            
            if not (os.path.isfile(ct) & os.path.isfile(mri)):
                continue
            
            if os.path.isfile(mr_lab):
                ct_img, mr_img, ct_lab_img, mr_lab_img = load_data(ct, mri, ct_lab, mr_lab)
            else:
                ct_img, mr_img, ct_lab_img = load_data(ct, mri, ct_lab)
                mr_lab_img = None

            # Normalization
            ct_norm = ScaleIntensityRange(-1024, 3071, -1, 1, clip=True)(ct_img)

            threshold_ct = np.sum(ct_norm < -0.99) / ct_norm.shape[0] * 1
            
            slice_i = preprocess_slices([], ct_img, ct_norm, ct_lab_img, threshold_ct)

            if not slice_i:
                print(f'[WARNING] No valid slices found in {folder} ({type})')
                continue

            if type == 'bl':
                bl_counts += len(slice_i)
            elif type == 'fu':
                fu_counts += len(slice_i)
            counts += 1

            # Save images
            save_images(ct_img, mr_img, ct_lab_img, mr_lab_img, slice_i, data_path, folder, type)

    print(f'Baseline Total 2D slice: {bl_counts}, Followup Total 2D slice: {fu_counts}')
    print(f'Total CT-MR: {counts}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess lung MR and CT data for segmentation.")
    parser.add_argument('--root_path', type=str, required=True, help='Root path of the dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to save the processed data')
    
    args = parser.parse_args()
    process_dataset(args.root_path, args.data_path)
