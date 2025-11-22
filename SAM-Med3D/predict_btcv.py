import json
from tqdm import tqdm
import os
from os.path import join
from glob import glob
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry
import torch

device = "cuda:0"

checkpoint_path = "./ckpt/sam_med3d_turbo.pth"
sam_model_tune = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device)
model_dict = torch.load(checkpoint_path, map_location=device)
state_dict = model_dict['model_state_dict']
sam_model_tune.load_state_dict(state_dict)
sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

sam_model_tune.eval()

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def dilate_mask_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Dilate a 3D binary mask by 'radius' voxels (cubic structuring element).
    Args:
        mask: torch.Tensor of shape (D, H, W), values 0/1.
        radius: number of voxels to dilate.
    Returns:
        torch.Tensor of shape (D, H, W), dilated mask.
    """
    m = mask.unsqueeze(0).unsqueeze(0).float()       # (1,1,D,H,W)
    k = 2 * radius + 1
    dilated = F.max_pool3d(m, kernel_size=k, stride=1, padding=radius)
    return (dilated > 0).squeeze().to(mask.dtype)

def erode_mask_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Erode a 3D binary mask by 'radius' voxels (cubic structuring element).
    Args:
        mask: torch.Tensor of shape (D, H, W), values 0/1.
        radius: number of voxels to erode.
    Returns:
        torch.Tensor of shape (D, H, W), eroded mask.
    """
    m_inv = (1 - mask).unsqueeze(0).unsqueeze(0).float()
    k = 2 * radius + 1
    eroded_inv = F.max_pool3d(m_inv, kernel_size=k, stride=1, padding=radius)
    eroded = 1 - eroded_inv
    return (eroded > 0).squeeze().to(mask.dtype)

def hollow_mask_from_center_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Create a hollow region in the center of a 3D mask by carving out a cube of given radius.
    Supports both shape (D, H, W) and batched (B, C, D, H, W) masks.
    """
    m_inv = (1 - mask).unsqueeze(0).unsqueeze(0).float()
    k = 2 * radius + 1
    eroded_inv = F.max_pool3d(m_inv, kernel_size=k, stride=1, padding=radius)
    eroded = 1 - eroded_inv
    holed_mask = mask - eroded
    result = (holed_mask > 0).squeeze().to(mask.dtype)
    return result

def translate_mask_3d(mask: torch.Tensor, shift=None) -> torch.Tensor:
    """
    Translate a 3D mask by (dz, dy, dx), filling empty voxels with 0.
    Args:
        mask: torch.Tensor of shape (D, H, W).
    Returns:
        torch.Tensor of shape (D, H, W), translated mask.
    """
    # augmentation: shift
    dz = np.random.randint(-3, 3)
    dy = np.random.randint(-3, 3)
    dx = np.random.randint(-3, 3)
    
    D, H, W = mask.shape
    out = torch.zeros_like(mask)

    def get_slices(n, s):
        if s >= 0:
            return slice(0, n - s), slice(s, n)
        else:
            return slice(-s, n), slice(0, n + s)

    sz, dz_dst = get_slices(D, dz)
    sy, dy_dst = get_slices(H, dy)
    sx, dx_dst = get_slices(W, dx)

    out[dz_dst, dy_dst, dx_dst] = mask[sz, sy, sx]
    return out

def no_aug(mask, r=None):
    return mask

from utils.click_method import *
import torchio as tio

click_methods = {
    'default': get_next_click3D_torch,
    'uncertainty': next_click_from_logits_uncertainty_no_gt
}


def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum


def sammed_predict3D(img3D, prompt3D, gt3D, sam_model_tune, device='cuda'):
    # '''both prompt3D and gt3D are in one-hot format'''
    
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)
    prev_masks = prompt3D.clone().float()
    
    # remaining is based on SAM-Med3D inference
    low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size//4, crop_size//4, crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device))
        
    with torch.no_grad():

        # ONLY MASKS AS PROMPTS
        _, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=None,
            masks=low_res_masks.to(device),
        )
        
        low_res_masks, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), 
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=None, 
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    
        # logits
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

        medsam_seg_prob = torch.sigmoid(prev_masks)
        # convert prob to mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        # pred_list.append(medsam_seg_prob)

        dice = round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4)

    return medsam_seg, dice

# get organ-wise datalist

if __name__ == "__main__":
    
    test_data_path = "/playpen/soumitri/SAM-Med3D/data_testbed/BTCV"
    
    all_dataset_paths = glob(join(test_data_path, "*", "*"))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")
    
    crop_size = 128

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(crop_size, crop_size, crop_size)),
    ]

    from utils.data_loader import Dataset_Union_ALL_Val

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type="Tr", 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=1,
        split_idx=0,
        pcc=False,
    )

    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )

    #### TENSORBOARD LOGGING ####

    from torch.utils.tensorboard import SummaryWriter
    import datetime

    # Create a log directory with a timestamp
    log_dir = f"/playpen/soumitri/SAM-Med3D/expts_oracle_ISBI/BTCV"

    organs = [os.path.basename(os.path.dirname(path)) for path in all_dataset_paths if "background" not in path]
    tasks = ["no-aug"] # ["dilate", "erode", "hole", "shift", "no-aug"]
    strengths = [1,2,3,4,5,6,7,8,9,10]
    rad=None

    pred_save_dir = os.path.join(log_dir, "preds")

    # writers = {}
    for organ in organs:
        os.makedirs(f"/playpen/soumitri/SAM-Med3D/expts_oracle_ISBI/BTCV/img_npz/{organ}", exist_ok=True)
        for task in tasks:
            # for rad in strengths:
            run_name = f"{organ}/{task}/radius={rad}"
            # writers[(organ, task, rad)] = SummaryWriter(os.path.join(log_dir, "tensorboard", run_name))
            os.makedirs(os.path.join(pred_save_dir, run_name), exist_ok=True)     

    print("organs:", str(organs))
    # print(f"TensorBoard SummaryWriter initialized at {log_dir}")

    from collections import defaultdict
    def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)
    def convert_defaultdict(d):
        if isinstance(d, defaultdict):
            d = {k: convert_defaultdict(v) for k, v in d.items()}
        return d

    all_logs = recursive_defaultdict()
    for organ in organs:
        for task in tasks:
            # for rad in strengths:
            all_logs[organ][task] = []

    task_mapper = {
        "dilate": dilate_mask_3d,
        "erode": erode_mask_3d,
        "hole": hollow_mask_from_center_3d,
        "shift": translate_mask_3d,
        "no-aug": no_aug    
    }  

    # loop begins
    for task in tasks:
    # for rad in strengths:
        with torch.no_grad():
            for i, test_data in enumerate(tqdm(test_dataloader)):
                
                image3D, gt3D, img_name = test_data   
                organ = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
                dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))  
                
                # extract tensorboard writer for current organ and radius
                # writer = writers[(organ, task, rad)]            
            
                # dilation
                aug_mask3D = task_mapper[task](gt3D[0][0], rad).unsqueeze(0).unsqueeze(0).float().to(device)
                
                # convert one-hot mask to logits to make it compatible with SAM-Med3D
                aug_mask3D[aug_mask3D > 0] = 10.0
                aug_mask3D[aug_mask3D == 0] = -20.0
                                
                image3D = F.interpolate(image3D.float(), size=(128,128,128), mode="trilinear", align_corners=False)
                aug_mask3D = F.interpolate(aug_mask3D.float(), size=(128,128,128), mode="nearest")
                gt3D = F.interpolate(gt3D.float(), size=(128,128,128), mode="nearest")
                
                # call the prediction function
                pred, dice = sammed_predict3D(
                    image3D, aug_mask3D, gt3D, sam_model_tune, device=device
                )
                
                # add dice scores to the logging dictionary
                all_logs[organ][task].append(dice)
                
                # save image + gt in one npz, and pred + prompt in another npz
                # img_npz_save_path = os.path.join(f"/playpen/soumitri/SAM-Med3D/expts_oracle_ISBI/BTCV/img_npz/{organ}", os.path.basename(img_name[0]).replace('.nii.gz', '_img+gt.npz'))
                # if not os.path.exists(img_npz_save_path):
                #     np.savez_compressed(img_npz_save_path, imgs=image3D.cpu().numpy().astype(np.float32), gts=gt3D.cpu().numpy().astype(np.uint8))
                npz_save_path = os.path.join(pred_save_dir, f"{organ}/{task}/radius={rad}", os.path.basename(img_name[0]).replace('.nii.gz', '_pred+prompt.npz'))
                np.savez_compressed(npz_save_path, preds=pred.astype(np.uint8), prompts=aug_mask3D.cpu().numpy().astype(np.uint8), dice=np.array([dice], dtype=np.float32))

            log_dict = convert_defaultdict(all_logs)
            json.dump(log_dict, open(os.path.join(log_dir, "evals/all_logs_no-aug.json"), 'w'), indent=2)
        
    log_dict = convert_defaultdict(all_logs)
    json.dump(log_dict, open(os.path.join(log_dir, "evals/all_logs_no-aug.json"), 'w'), indent=2)