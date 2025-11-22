import os
import argparse
import numpy as np
from glob import glob
from torch.utils.tensorboard import SummaryWriter

#!/usr/bin/env python3

"""
visualize.py

Scan NPZ files under ./dilate, derive corresponding "original" filenames by swapping a filename prefix,
load arrays (keys: img, gts, segs, prompt), and write TensorBoard image series where each global_step
is a slice index so the TensorBoard time slider scrolls through slices. Each logged image is 4 columns:
[image | prompt | pred | gt].

Usage:
    python visualize.py --aug-dir ./dilate --aug-prefix dilate --orig-prefix orig --out-logdir runs/vis
"""


def find_slice_axis(arrs):
    # Return an axis index that is common across all arrays and >1 (assume that's the slice axis).
    shapes = [a.shape for a in arrs]
    ndim = min(len(s) for s in shapes)
    # look for an axis index (from 0..max_ndim-1) that exists for all and has same size
    for axis in range(max(len(s) for s in shapes)):
        sizes = []
        valid = True
        for s in shapes:
            if axis >= len(s):
                valid = False
                break
            sizes.append(s[axis])
        if not valid:
            continue
        if len(set(sizes)) == 1 and sizes[0] > 1:
            return axis
    # fallback: try first axis for all arrays (if they match)
    for axis in range(max(len(s) for s in shapes)):
        try:
            sizes = [s[axis] for s in shapes]
            if len(set(sizes)) == 1 and sizes[0] > 1:
                return axis
        except Exception:
            continue
    raise ValueError("Could not determine common slice axis for arrays: %s" % (shapes,))


def as_uint8_rgb(x):
    """
    Convert a 2D or 3D single-channel/float array into HWC uint8 RGB image (0-255).
    If already 3-channel, will try to leave channels as-is (convert to uint8).
    """
    if x is None:
        return None
    x = np.asarray(x)
    # If has channel-first CxHxW, transpose to HWC
    if x.ndim == 3 and x.shape[0] <= 4 and x.shape[0] != x.shape[2]:
        # c,h,w -> h,w,c
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.expand_dims(x, -1)
    # normalize floats to 0-255
    if np.issubdtype(x.dtype, np.floating):
        mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    elif np.issubdtype(x.dtype, np.bool_):
        x = (x.astype(np.uint8) * 255)
    else:
        # integer types: scale to 0-255 if max>255
        if x.dtype != np.uint8:
            mx = x.max() if x.size else 255
            if mx > 255:
                x = (x.astype(np.float32) / float(mx) * 255.0).astype(np.uint8)
            else:
                x = x.astype(np.uint8)
    # ensure 3 channels
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.shape[-1] == 2:
        # pad to 3
        pad = np.zeros((x.shape[0], x.shape[1], 1), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=-1)
    elif x.shape[-1] > 3:
        x = x[:, :, :3]
    return x


def concat_columns(images):
    # images: list of HWC uint8 arrays with same H
    heights = [im.shape[0] for im in images]
    if len(set(heights)) != 1:
        # resize by simple pad/crop to max height (center)
        H = max(heights)
        resized = []
        for im in images:
            h, w = im.shape[:2]
            if h == H:
                resized.append(im)
            elif h < H:
                pad_top = (H - h) // 2
                pad_bottom = H - h - pad_top
                pad = np.zeros((pad_top, w, 3), dtype=np.uint8)
                pad2 = np.zeros((pad_bottom, w, 3), dtype=np.uint8)
                resized.append(np.concatenate([pad, im, pad2], axis=0))
            else:
                # crop center
                start = (h - H) // 2
                resized.append(im[start:start + H, :, :])
        images = resized
    return np.concatenate(images, axis=1)


def derive_original_paths(dilate_path, aug_dir, dilate_prefix, orig_prefix):
    base = os.path.basename(dilate_path)
    # replace prefix in filename if present
    if base.startswith(dilate_prefix):
        candidate = base.replace(dilate_prefix, orig_prefix, 1)
    elif base.startswith(dilate_prefix):
        candidate = base.replace(dilate_prefix, orig_prefix, 1)
    else:
        candidate = orig_prefix + base
    parent = os.path.dirname(aug_dir)
    # search in parent first, then same dir, then current working dir
    candidates = [
        os.path.join(parent, candidate),
        os.path.join(aug_dir, candidate),
        os.path.join(os.getcwd(), candidate),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: same name with orig_prefix appended prefix (in parent)
    return candidates[0]  # may not exist; upstream will handle missing


def process_pair(aug_npz, orig_img_npz, writer, tag_prefix):
    
    img = np.load(orig_img_npz, allow_pickle=True)['imgs']
    gts = np.load(orig_img_npz, allow_pickle=True)['gts']
    segs = np.load(aug_npz, allow_pickle=True)['segs']
    prompt = np.load(aug_npz, allow_pickle=True)['prompts']
    # Transpose from HWD to DHW -- specific to BTCV data layout
    img, gts, segs, prompt = [np.rot90(np.transpose(arr, (2, 0, 1)), k=1, axes=(1, 2)) for arr in [img, gts, segs, prompt]]
    # print(img.shape, segs.shape, prompt.shape, gts.shape)

    arrays = [a for a in (img, prompt, segs, gts) if a is not None]
    slice_axis = find_slice_axis(arrays)

    nslices = arrays[0].shape[slice_axis]
    base_tag = tag_prefix
    for s in range(nslices):
        # extract slices
        def take(a):
            if a is None:
                return None
            return np.take(a, indices=s, axis=slice_axis)

        # raw slices (may be label maps or floats); don't convert labels with as_uint8_rgb directly
        im_slice = take(img)
        pr_slice = take(prompt)
        pd_slice = take(segs)
        gt_slice = take(gts)

        # normalize image (grayscale/float) to RGB uint8
        im = as_uint8_rgb(im_slice)

        # helper: convert integer label map (H,W) or one-hot (H,W,C) to colored HWC uint8
        def label_to_color(lbl):
            if lbl is None:
                return None
            lbl = np.asarray(lbl)
            # handle channel-first CxHxW
            if lbl.ndim == 3 and lbl.shape[0] <= 8 and lbl.shape[0] != lbl.shape[2]:
                lbl = np.transpose(lbl, (1, 2, 0))
            # one-hot or multi-channel -> argmax
            if lbl.ndim == 3 and lbl.shape[2] > 1:
                try:
                    lbl = np.argmax(lbl, axis=-1)
                except Exception:
                    lbl = lbl[..., 0]
            # if float probabilities, threshold to get discrete labels for visualization
            if np.issubdtype(lbl.dtype, np.floating):
                # if binary/prob map, threshold at 0.5 otherwise cast
                lbl = (lbl > 0.5).astype(np.int32)
            lbl = lbl.astype(np.int32)
            if lbl.ndim == 2:
                h, w = lbl.shape
            else:
                # unexpected, try to collapse
                lbl = lbl.squeeze()
                if lbl.ndim != 2:
                    return None
                h, w = lbl.shape
            # simple repeating palette (index 0 -> background black)
            palette = np.array([
                [0, 0, 0],
                [220, 20, 60],
                [34, 139, 34],
                [30, 144, 255],
                [255, 165, 0],
                [148, 0, 211],
                [255, 105, 180],
                [0, 206, 209],
                [128, 128, 128],
                [255, 215, 0],
            ], dtype=np.uint8)
            ncol = palette.shape[0]
            out = np.zeros((h, w, 3), dtype=np.uint8)
            uniq = np.unique(lbl)
            for v in uniq:
                if v < 0:
                    continue
                color = palette[int(v) % ncol]
                out[lbl == v] = color
            return out

        # convert preds and gts (class labels) to colored maps
        pd_color = label_to_color(pd_slice)
        gt_color = label_to_color(gt_slice)

        # create overlays: blend colored map over image if image exists, otherwise show color map
        def make_overlay(color_map, src_img, alpha=0.5):
            if color_map is None:
                return None
            if src_img is None:
                return color_map
            cm = color_map
            im_local = src_img
            # ensure same spatial size (quick center-crop/pad)
            if cm.shape[:2] != im_local.shape[:2]:
                h, w = im_local.shape[:2]
                ch, cw = cm.shape[:2]
                if ch < h or cw < w:
                    pad_top = max((h - ch) // 2, 0)
                    pad_bottom = h - ch - pad_top
                    pad_left = max((w - cw) // 2, 0)
                    pad_right = w - cw - pad_left
                    padded = np.zeros((h, w, 3), dtype=np.uint8)
                    padded[pad_top:pad_top + ch, pad_left:pad_left + cw] = cm
                    cm = padded
                else:
                    start_h = (ch - h) // 2
                    start_w = (cw - w) // 2
                    cm = cm[start_h:start_h + h, start_w:start_w + w]
            # mask where color map indicates a label (non-black)
            mask = (cm.sum(axis=-1) > 0)
            if not mask.any():
                return im_local.copy()
            overlay = im_local.copy().astype(np.float32)
            overlay[mask] = overlay[mask] * (1.0 - alpha) + cm[mask].astype(np.float32) * alpha
            return overlay.astype(np.uint8)

        # build prompt colored map and overlay it like preds/gts so different prompt classes get colors
        pr_color = label_to_color(pr_slice)
        pd = make_overlay(pd_color, im, alpha=0.5)
        gt = make_overlay(gt_color, im, alpha=0.5)
        pr = make_overlay(pr_color, im, alpha=0.5) if pr_color is not None else None

        # ensure all panes are HWC uint8 and same spatial size (use first available as reference)
        ref = im if im is not None else (pr if pr is not None else (pd if pd is not None else gt))
        if ref is None:
            # nothing to show for this slice
            continue
        H, W = ref.shape[:2]

        def ensure_rgb(x):
            if x is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            x = as_uint8_rgb(x) if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3 else x
            h, w = x.shape[:2]
            if (h, w) == (H, W):
                return x
            # center-crop or pad to match H,W
            if h < H or w < W:
                pad_top = max((H - h) // 2, 0)
                pad_bottom = max(H - h - pad_top, 0)
                pad_left = max((W - w) // 2, 0)
                pad_right = max(W - w - pad_left, 0)
                padded = np.zeros((h + pad_top + pad_bottom, w + pad_left + pad_right, 3), dtype=np.uint8)
                padded[pad_top:pad_top + h, pad_left:pad_left + w] = x
                return padded
            else:
                start_h = (h - H) // 2
                start_w = (w - W) // 2
                return x[start_h:start_h + H, start_w:start_w + W]

        im, pr, pd, gt = ensure_rgb(im), ensure_rgb(pr), ensure_rgb(pd), ensure_rgb(gt)

        quad = concat_columns([im, pr, pd, gt])
        # tag per-file: base_tag/slices
        tag = f"{base_tag}"
        writer.add_image(tag, quad, global_step=s, dataformats='HWC')
    writer.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aug-dir", default="./BTCV_preds", help="Directory containing augmented npz files")
    p.add_argument("--aug-prefix", help="dilate/erode/hole etc prefix under aug-dir")
    p.add_argument("--orig-prefix", default="./BTCV_npz", help="Prefix to use for original filenames")
    p.add_argument("--out-logdir", default="./BTCV_tboard", help="TensorBoard logdir output")
    p.add_argument("--ext", default="npz", help="File extension to search (default npz)")
    args = p.parse_args()

    for radius in range(1,11):
        aug_dir = os.path.join(args.aug_dir, f"{args.aug_prefix}_radius={radius}")
        log_dir = os.path.join(args.out_logdir, f"{args.aug_prefix}/radius={radius}")
        if args.aug_prefix in ['no-aug', 'shift']:
            aug_dir = os.path.join(args.aug_dir, f"{args.aug_prefix}")
            log_dir = os.path.join(args.out_logdir, f"{args.aug_prefix}")
        writer = SummaryWriter(log_dir=log_dir)
        pattern = os.path.join(aug_dir, f"*.{args.ext}")
        files = sorted(glob(pattern))
        if not files:
            print("No files found in", pattern)
            continue

        for aug_path in files:
            f = os.path.basename(aug_path) # XXXX.npz
            orig_path_img = os.path.join(args.orig_prefix, f)
            # print("Processing:", aug_path, " -> original:", orig_path_img)
            process_pair(aug_path, orig_path_img, writer, tag_prefix=f)

        print("Done. TensorBoard logs written to", args.out_logdir)
        writer.close()

if __name__ == "__main__":
    main()