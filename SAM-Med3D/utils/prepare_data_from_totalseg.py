import os.path as osp
import os
import json
import shutil
import nibabel as nib
from tqdm import tqdm
import torchio as tio

def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1.5, 1.5, 1.5), n=None, reference_image=None, mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    
    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)
    
    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        if(isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img
    
    save_image.save(output_path)


# dataset_root = "/playpen/soumitri/data/TotalSegMRI_test/test"
# target_dir = "./data_preproc/TotalSegMRI_test"

# os.makedirs(target_dir, exist_ok=True)
# class_list = os.listdir(osp.join(dataset_root, "s0002", "segmentations"))
# for cls in class_list:
#     dirname = cls.split(".nii.gz")[0]
#     os.makedirs(osp.join(target_dir, dirname, "totalsegmri", "imagesTs"))
#     os.makedirs(osp.join(target_dir, dirname, "totalsegmri", "labelsTs"))
    
# for pat_id in os.listdir(dataset_root):
#     orig_img_path = osp.join(dataset_root, pat_id, "mri.nii.gz")
#     target_img_path = osp.join(target_dir, '{}', "totalsegmri", "imagesTs", pat_id+".nii.gz")
#     orig_seg_path = osp.join(dataset_root, pat_id, "segmentations", '{}' + ".nii.gz")
#     target_seg_path = osp.join(target_dir, '{}', "totalsegmri", "labelsTs", pat_id+".nii.gz")
     
#     for cls in class_list:
#         dirname = cls.split(".nii.gz")[0]
#         shutil.copy(orig_img_path, target_img_path.format(dirname))
#         shutil.copy(orig_seg_path.format(dirname), target_seg_path.format(dirname))
             
             
import glob
imageslist = sorted(glob.glob("/playpen/soumitri/SAM-Med3D/data_preproc/TotalSegMRI_test/*/totalsegmri/imagesTs/*.nii.gz"))
labelslist = sorted(glob.glob("/playpen/soumitri/SAM-Med3D/data_preproc/TotalSegMRI_test/*/totalsegmri/labelsTs/*.nii.gz"))

for imgpath, labpath in zip(imageslist, labelslist):
    resample_nii(imgpath, imgpath)
    gt_img = nib.load(labpath)    
    spacing = tuple(gt_img.header['pixdim'][1:4])
    spacing_voxel = spacing[0] * spacing[1] * spacing[2]
    reference_image = tio.ScalarImage(imgpath)
    resample_nii(labpath, labpath, n=1, reference_image=reference_image, mode='nearest')