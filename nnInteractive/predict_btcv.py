from __future__ import annotations
import argparse
import os
import json
import glob
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache


def mask3D_to_dense_prompt(gt3D, spacing, augment, radius=3, to_logit=False):
    """3D mask --> augment --> convert to logit space --> return dense prompt"""
    if augment == "dilate":
        if radius is None: radius = np.random.randint(1, 5)
        gt3D = dilate_mask_3d(gt3D, radius)
    elif augment == "erode":
        if radius is None: radius = np.random.randint(1, 5)
        gt3D = erode_mask_3d(gt3D, radius)
    elif augment == "hole":
        if radius is None: radius = np.random.randint(1, 5)
        gt3D = hollow_mask_from_center_3d(gt3D, radius)
    elif augment == "shift":
        dz = np.random.randint(-3, 3)
        dy = np.random.randint(-3, 3)
        dx = np.random.randint(-3, 3)
        gt3D = shift_mask_3d(gt3D, (dz, dy, dx))
    elif augment == "random-aug":
        p = np.random.random()
        if p < 0.25:
            # augmentation: dilate
            radius = np.random.randint(1, 5)
            gt3D = dilate_mask_3d(gt3D, radius)
        elif p < 0.50:
            # augmentation: erode
            radius = np.random.randint(1, 5)
            gt3D = erode_mask_3d(gt3D, radius)
        elif p < 0.75:
            # augmentation: hollow out from center  
            radius = np.random.randint(1, 5)
            gt3D = hollow_mask_from_center_3d(gt3D, radius)
        else:
            # augmentation: shift
            dz = np.random.randint(-3, 3)
            dy = np.random.randint(-3, 3)
            dx = np.random.randint(-3, 3)
            gt3D = shift_mask_3d(gt3D, (dz, dy, dx))
    # else: if augment == "no-aug": eat 5-star, do nothing
    
    if to_logit:
        # convert to logit space if required (NOT required for nnInteractive)
        gt3D = binary_mask_to_logit(gt3D, spacing, use_spacing=True, sigma=1.0, logit_clip=10.0, eps=1e-6)
        
    return gt3D 


def binary_mask_to_logit(mask, spacing, use_spacing, sigma=1.0, logit_clip=10.0, eps=1e-6):
    """
    Converts binary segmentation masks (0 and 1) in batch format into smooth logit maps.
    
    Args:
        mask (torch.Tensor): Batch of binary masks with shape [B, C, D, H, W] or [C, D, H, W].
        sigma (float): Gaussian blur strength.
        logit_clip (float): Maximum absolute logit value for numerical stability.
        eps (float): Small value to ensure probabilities are in [eps, 1-eps].
    
    Returns:
        logit_map (torch.Tensor): Logit map(s) with same shape as input.
    """
    # If mask does not have batch dimension, add one.
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        mask = mask.unsqueeze(0)
    elif mask.dim() != 5:
        raise ValueError("Input mask must have 3/4/5 dimensions")
    
    B, C, D, H, W = mask.shape
    output = []
    
    if use_spacing:
        sigma_vox = [sigma / s for s in spacing]   # per-axis sigma in voxels
    else:
        sigma_vox = sigma
    
    for b in range(B):
        channel_outputs = []
        for c in range(C):
            # Get single mask (D, H, W) and convert to numpy array
            mask_np = mask[b, c].detach().cpu().numpy().astype(float)
            # Apply Gaussian filter
            if use_spacing:
                prob_smoothed = gaussian_filter(mask_np, sigma=sigma_vox, mode='constant', cval=0.0)
            else:
                prob_smoothed = gaussian_filter(mask_np, sigma=sigma)
            prob_smoothed_tensor = torch.tensor(prob_smoothed, device=mask.device)
            # Clamp to avoid extreme values
            prob_clipped = torch.clamp(prob_smoothed_tensor, eps, 1 - eps)
            # Convert to logit space
            logit = torch.log(prob_clipped / (1 - prob_clipped))
            logit = torch.clamp(logit, -logit_clip, logit_clip)
            channel_outputs.append(logit)
        output.append(torch.stack(channel_outputs, dim=0))
    
    return torch.stack(output, dim=0)

def dilate_mask_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Dilate a 3D binary mask by 'radius' voxels using a cubic structuring element.
    Supports both single masks of shape (D, H, W) and batched masks of shape (B, C, D, H, W).
    """
    original_ndim = mask.ndim
    if original_ndim == 4:
        m = mask.unsqueeze(0).float()  # shape (1,1,D,H,W)
    elif original_ndim == 5:
        m = mask.float()
    else:
        raise ValueError(f"mask must be 4D or 5D, found {mask.shape}")
    k = 2 * radius + 1
    dilated = F.max_pool3d(m, kernel_size=k, stride=1, padding=radius)
    result = (dilated > 0).to(mask.dtype)
    if original_ndim == 3:
        result = result.squeeze(0).squeeze(0)
    return result

def erode_mask_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Erode a 3D binary mask by 'radius' voxels using a cubic structuring element.
    Supports both single masks of shape (D, H, W) and batched masks of shape (B, C, D, H, W).
    """
    # The above code is performing erosion operation on a binary mask tensor. Here is a breakdown of
    # the code:
    original_ndim = mask.ndim
    if original_ndim == 4:
        m_inv = (1 - mask).unsqueeze(0).float()  # (1,1,D,H,W)
    elif original_ndim == 5:
        m_inv = (1 - mask).float()
    else:
        raise ValueError(f"mask must be 4D or 5D, found {mask.shape}")
    k = 2 * radius + 1
    eroded_inv = F.max_pool3d(m_inv, kernel_size=k, stride=1, padding=radius)
    eroded = 1 - eroded_inv
    result = (eroded > 0).to(mask.dtype)
    if original_ndim == 3:
        result = result.squeeze(0).squeeze(0)
    return result

def hollow_mask_from_center_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Create a hollow region in the center of a 3D mask by carving out a cube of given radius.
    Supports both shape (D, H, W) and batched (B, C, D, H, W) masks.
    """
    original_ndim = mask.ndim
    if original_ndim == 4:
        m_inv = (1 - mask).unsqueeze(0).float()  # (1,1,D,H,W)
    elif original_ndim == 5:
        m_inv = (1 - mask).float()
    else:
        raise ValueError(f"mask must be 4D or 5D, found {mask.shape}")
    k = 2 * radius + 1
    eroded_inv = F.max_pool3d(m_inv, kernel_size=k, stride=1, padding=radius)
    eroded = 1 - eroded_inv
    holed_mask = mask - eroded
    result = (holed_mask > 0).to(mask.dtype)
    if original_ndim == 3:
        result = result.squeeze(0).squeeze(0)
    return result

def shift_mask_3d(mask: torch.Tensor, shift: tuple[int, int, int]) -> torch.Tensor:
    """
    Translate a 3D mask by (dz, dy, dx) along the last three dimensions,
    filling empty voxels with 0.
    Works with masks of shape (D,H,W) or batched masks of shape (B,C,D,H,W).
    """
    dz, dy, dx = shift
    if mask.ndim == 3:
        D, H, W = mask.shape
        out = torch.zeros_like(mask)
    elif mask.ndim == 4:
        D, H, W = mask.shape[-3:]
        out = torch.zeros_like(mask)
    elif mask.ndim == 5:
        mask = mask.squeeze(0)
        D, H, W = mask.shape[-3:]
        out = torch.zeros_like(mask)
    else:
        raise ValueError(f"mask must be 3D or 4D or 5D, found {mask.shape}")

    def get_slices(n, s):
        if s >= 0:
            return slice(0, n - s), slice(s, n)
        else:
            return slice(-s, n), slice(0, n + s)

    sz, dz_dst = get_slices(D, dz)
    sy, dy_dst = get_slices(H, dy)
    sx, dx_dst = get_slices(W, dx)

    if mask.ndim == 3:
        out[dz_dst, dy_dst, dx_dst] = mask[sz, sy, sx]
    else:
        out[:, dz_dst, dy_dst, dx_dst] = mask[:, sz, sy, sx]
        if mask.ndim == 5: out = out.unsqueeze(0)
    return out


def compute_multiclass_dice(gt: np.ndarray, pred: np.ndarray):
    """Compute per-label Dice and mean Dice for non-zero labels in gt.

    Returns (per_label_dict, mean_dice, fg_dice)
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f'GT shape {gt.shape} and pred shape {pred.shape} do not match')

    # Ensure integer label maps
    if not np.issubdtype(gt.dtype, np.integer):
        # if gt is float but contains integers, cast
        if np.all(np.mod(gt, 1) == 0):
            gt = gt.astype(np.int32)
        else:
            raise ValueError('GT must be integer label map for multiclass Dice')

    if not np.issubdtype(pred.dtype, np.integer):
        if np.all(np.mod(pred, 1) == 0):
            pred = pred.astype(np.int32)
        else:
            # if pred is probabilistic or binary float, threshold at 0.5
            pred = (pred > 0.5).astype(np.int32)

    labels = np.unique(gt)
    labels = labels[labels != 0]
    per_label = {}
    for lab in labels:
        gt_bin = (gt == lab)
        pred_bin = (pred == lab)
        inter = float(np.logical_and(gt_bin, pred_bin).sum())
        gt_sum = float(gt_bin.sum())
        pred_sum = float(pred_bin.sum())
        if gt_sum + pred_sum == 0:
            dice = 1.0
        else:
            dice = 2.0 * inter / (gt_sum + pred_sum)
        per_label[int(lab)] = float(dice)

    mean_dice = float(np.mean(list(per_label.values()))) if len(per_label) > 0 else 1.0

    # foreground (any label >0)
    gt_fg = (gt > 0)
    pred_fg = (pred > 0)
    inter_fg = float(np.logical_and(gt_fg, pred_fg).sum())
    gt_fg_sum = float(gt_fg.sum())
    pred_fg_sum = float(pred_fg.sum())
    if gt_fg_sum + pred_fg_sum == 0:
        fg_dice = 1.0
    else:
        fg_dice = 2.0 * inter_fg / (gt_fg_sum + pred_fg_sum)

    return per_label, mean_dice, float(fg_dice)


def run_inference(
    image: np.ndarray,
    spacing: tuple[float, float, float],
    bbox: list[dict] | None,
    clicks: list[dict] | None,
    clicks_order: list[list[str]] | None,
    prev_pred: np.ndarray | None,
    mask_aug: str = "no-aug",
    radius: int = None
) -> np.ndarray:
    """
    Stub performing **one** forward pass of your model.

    Parameters
    ----------
    image : (D, H, W) np.ndarray
        Raw image volume (usually float32).  *No preprocessing applied*.
    spacing : (3,) tuple of float
        Physical voxel spacing (z, y, x) in millimetres.
    bbox : list of dict | None
        Bounding‑box prompt(s).  The dict structure is shown in the challenge
        description; may be absent in refinement iterations.
    clicks : list of dict | None
        Fore‑ and background click dictionaries for every class.
    prev_pred : (D, H, W) np.ndarray | None
        Segmentation from the previous iteration.  May be `None` for the first
        call.
    mask_aug : str = "no-aug" | choices = ["dilate", "erode", "shift", "hole", "random-aug", "no-aug"]
        Type of mask augmentation to apply when refining from previous prediction (a la dense-prommpted-segmentation).
        Default is "none" (no augmentation).
    radius: int = None |
        Radius of dilation/erosion. not for shift
    Returns
    -------
    seg : (D, H, W) np.ndarray, dtype=uint8
        Multi‑class segmentation mask.  Background **must** be 0;
        classes start from 1 … N.  Make sure dtype is `np.uint8`.
    """
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=os.path.join(CHECKPOINT_DIR, MODEL_NAME),
    )
    session.set_image(image[None].astype(np.float32))
    target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
    session.set_target_buffer(target_buffer)
    
    # If only a dense previous segmentation is provided (no bboxes or clicks),
    # iterate over all labeled objects in that mask and refine each as a separate
    # dense prompt. This preserves original label values in the composed output.
    if (bbox is None or len(bbox) == 0) and (clicks is None or len(clicks) == 0) and prev_pred is not None:
        pp = prev_pred
        if isinstance(pp, torch.Tensor):
            pp = pp.cpu().numpy()

        if pp.ndim != 3:
            raise ValueError(f'Unsupported prev_pred shape {pp.shape}. Expected 3D (D,H,W).')

        # If pp contains integer labels (multi-object), iterate unique non-zero labels
        if np.issubdtype(pp.dtype, np.integer) or np.all(np.mod(pp, 1) == 0):
            label_map = pp.astype(np.int32)
            unique_labels = np.unique(label_map)
            unique_labels = unique_labels[unique_labels != 0]

            result = np.zeros(image.shape, dtype=np.uint8)
            aug_prompt = np.zeros_like(prev_pred)
            if len(unique_labels) == 0:
                del session
                empty_cache(torch.device('cuda', 0))
                return result

            # Refine each label separately. We fill only empty voxels in `result` to avoid overwriting
            # earlier refined objects. If you prefer a different tie-breaking, we can change this.
                    
            for lab in unique_labels:
                try:
                    mask = (label_map == lab).astype(np.uint8)
                    # now that we have a binary mask, we need to convert it into a dense prompt // nnInteractive does not take logit map, it takes binary map
                    mask = mask3D_to_dense_prompt(
                        gt3D=torch.tensor(mask, device='cpu').unsqueeze(0).unsqueeze(0), 
                        spacing=spacing, augment=mask_aug, radius=radius,
                        to_logit=False
                    )
                    mask = mask.squeeze(0).squeeze(0).numpy()
                    
                    # Ensure mask shape matches expected (D,H,W)
                    if mask.shape != image.shape:
                        raise ValueError(f'Prev mask for label {lab} has shape {mask.shape}, expected {image.shape}.')
                    session.add_initial_seg_interaction(mask, run_prediction=True)
                    pred_mask = session.target_buffer.cpu().numpy() if isinstance(session.target_buffer, torch.Tensor) else session.target_buffer.copy()
                    # assign only to currently unlabeled voxels to preserve earlier assignments
                    to_assign = (pred_mask > 0) & (result == 0)
                    result[to_assign] = int(lab)
                    to_assign_prompt = (mask > 0) & (aug_prompt == 0)
                    aug_prompt[to_assign_prompt] = int(lab)
                except Exception as e:
                    print(f'Warning: could not segment for label {lab}: {e}')
                    continue

            del session
            empty_cache(torch.device('cuda', 0))
            return result, aug_prompt
        
        else:
            # Treat as single soft/dense mask: threshold >0 becomes foreground
            pp_input = pp.astype(np.float32)
            session.add_initial_seg_interaction(pp_input, run_prediction=True)
            pred_out = session.target_buffer.cpu().numpy() if isinstance(session.target_buffer, torch.Tensor) else session.target_buffer.copy()
            result = (pred_out > 0).astype(np.uint8)
            del session
            empty_cache(torch.device('cuda', 0))
            return result

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mask_aug", type=str, choices=["dilate", "erode", "shift", "hole", "random-aug", "no-aug"], default="no-aug", help="type of mask augmentation to apply")
    p.add_argument("--radius", type=int, default=None, help="radius for dilation/erosion. shift is in [-3, 3]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    file_paths = sorted(glob.glob("BTCV_npz/*.npz"))

    pred_save_path = f"BTCV_preds/{args.mask_aug}"
    if args.mask_aug in ['dilate', 'erode', 'hole']: pred_save_path = pred_save_path + f"_radius={args.radius}"
    os.makedirs(pred_save_path, exist_ok=True)
    
    dice_save_path = f"BTCV_evals/{args.mask_aug}"
    if args.mask_aug in ['dilate', 'erode', 'hole']: dice_save_path = dice_save_path + f"_radius={args.radius}"
    dice_save_path = dice_save_path + ".json"
    
    if os.path.exists(dice_save_path):
        with open(dice_save_path, 'r') as f:
            all_dices = json.load(f)
        print(f"[predict_btcv.py] Loaded existing Dice results from {dice_save_path} ==> {len(all_dices)} images.")
    else:
        all_dices = {}
        print(f"[predict_btcv.py] No existing Dice results found at {dice_save_path} ==> starting fresh.")
    
    for img_path in tqdm(file_paths, desc="Processing BTCV images"):
        
        # check if image has been previously processed
        if img_path in all_dices.keys():
            print(f"[predict_btcv.py] Skipping {img_path}, already processed.")
            continue
        
        # ---------------------- Load input & prompts -------------------------- #
        npz_data = np.load(img_path, allow_pickle=True)
        
        image        = npz_data["imgs"]
        spacing      = tuple(npz_data["spacing"])
        gt_mask      = npz_data["gts"]
        
        # bbox         = npz_data.get("boxes")         # bounding boxes
        # clicks       = data.get("clicks")        # fg/bg clicks per class
        # clicks_order = data.get("clicks_order")  # order of click types
        # prev_pred    = data.get("prev_pred")     # from last iteration

        # --------------------------- Inference -------------------------------- #
        seg, prompt = run_inference(image, spacing=spacing, prev_pred=gt_mask, 
                                    mask_aug=args.mask_aug, radius=args.radius, # only inputs, rest are None
                                    bbox=None, clicks=None, clicks_order=None)
        
        
        # --------------------------- Dice Evaluation -------------------------- #
        dicelist = {}
        try:
            per_label_dice, mean_dice, _ = compute_multiclass_dice(gt_mask, seg)
            for lab, d in sorted(per_label_dice.items()):
                dicelist[lab] = d
            dicelist['mean_dice'] = mean_dice
            all_dices[img_path] = dicelist
        except Exception as e:
            print(f'[predict_btcv.py] Warning: could not compute Dice: {e}')

        # save all dices
        with open(dice_save_path, 'w') as f:
            json.dump(all_dices, f, indent=4)
            
        # ------------------------- Save prediction ---------------------------- #
        save_path = pred_save_path + "/" + os.path.basename(img_path)
        np.savez_compressed(save_path, segs=seg.astype(np.uint8), prompts=prompt.astype(np.uint8))
        print(f"[predict_btcv.py] Saved prediction to {save_path}")
    
    # final save of all dices
    with open(dice_save_path, 'w') as f:
        json.dump(all_dices, f, indent=4)
    print(f"[predict_btcv.py] Saved all Dice results to {dice_save_path}")


if __name__ == "__main__":
     # --- Download Trained Model Weights (~400MB) ---

    CHECKPOINT_DIR = './ckpt'
    REPO_ID = "nnInteractive/nnInteractive"
    MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
    
    if not os.path.exists(CHECKPOINT_DIR) or not any(os.scandir(CHECKPOINT_DIR)):
        print(f"[predict_btcv.py] Checkpoint directory {CHECKPOINT_DIR} is empty. Downloading model weights...")
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=CHECKPOINT_DIR
        )
        
    main()