"""
limb_tracking.py
================
Animal Kingdom Pose Estimation Pipeline

Trains and evaluates three models on the Animal Kingdom pose-estimation split:
  - HRNet   (top-down, high-resolution parallel branches)
  - ViTPose (top-down, Vision Transformer backbone)
  - DEKR    (bottom-up, no bounding-box dependency at inference)

GPU detection
-------------
Automatically uses CUDA if available, then MPS (Apple Silicon), then CPU.
Printed at startup: "Training on cuda:0" / "Training on mps" / "Training on cpu".

Expected directory layout
--------------------------
<data_root>/
  annotations/
    ak_P1/
      train.json
      test.json
    ak_P2/
      train.json
      test.json
    ak_P3_amphibian/
      train.json
      test.json
    ak_P3_bird/
      train.json
      test.json
    ak_P3_fish/
      train.json
      test.json
    ak_P3_mammal/
      train.json
      test.json
    ak_P3_reptile/
      train.json
      test.json
  images/
    AAACXZTV/
      AAACXZTV_f000059.jpg
      ...
    AAAUILHH/
      AAAUILHH_f000098.jpg
      ...

Usage
-----
# Train on ALL annotation directories at once
python limb_tracking.py \\
    --data_root  /path/to/ak \\
    --ann_dirs   all

# Train on a single split
python limb_tracking.py \\
    --data_root  /path/to/ak \\
    --ann_dirs   ak_P3_mammal

# Train on specific splits (space-separated)
python limb_tracking.py \\
    --data_root  /path/to/ak \\
    --ann_dirs   ak_P3_mammal ak_P3_bird ak_P3_reptile

# Use specific models only
python limb_tracking.py \\
    --data_root  /path/to/ak \\
    --ann_dirs   all \\
    --models     hrnet vitpose

# Inference on a single image with a saved checkpoint
python limb_tracking.py infer \\
    --model      hrnet \\
    --checkpoint results/hrnet_best.pth \\
    --image      my_animal.jpg

Outputs (written to --results_dir, default: ./results/)
--------------------------------------------------------
  hrnet_train.log          per-epoch training log
  vitpose_train.log
  dekr_train.log
  hrnet_loss_acc.png       loss + accuracy curves
  vitpose_loss_acc.png
  dekr_loss_acc.png
  hrnet_confusion.png      per-joint detection confusion matrix
  vitpose_confusion.png
  dekr_confusion.png
  hrnet_best.pth           best checkpoint (lowest val loss)
  vitpose_best.pth
  dekr_best.pth
  model_comparison.png     combined test-set curves for all models

Note on loss and accuracy
--------------------------
Pose estimation is a regression task — ground truth is a continuous heatmap,
not a discrete class label.  Therefore:
  Loss     = MSE between predicted and ground-truth Gaussian heatmaps,
             computed per visible joint (visibility-masked).
  Accuracy = PCK@0.2 (Percentage of Correct Keypoints within 20% of the
             torso size), the standard metric for animal pose estimation.
             Reported as a percentage (0-100) to match the training log format.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_JOINTS = 23

JOINT_NAMES = [
    "Head_Mid_Top",    # 0
    "Head_Left",       # 1
    "Head_Right",      # 2
    "Neck_Top",        # 3
    "Shoulder_Left",   # 4
    "Shoulder_Right",  # 5
    "Elbow_Left",      # 6
    "Elbow_Right",     # 7
    "Wrist_Left",      # 8
    "Wrist_Right",     # 9
    "Hip_Left",        # 10
    "Hip_Right",       # 11
    "Knee_Left",       # 12
    "Knee_Right",      # 13
    "Ankle_Left",      # 14
    "Ankle_Right",     # 15
    "Tail_Base",       # 16
    "Tail_Mid",        # 17
    "Tail_End_Back",   # 18
    "Spine_Mid",       # 19
    "Spine_Top",       # 20
    "Paw_Front_Left",  # 21
    "Paw_Front_Right", # 22
]

SKELETON = [
    (0, 3), (3, 4), (3, 5),
    (4, 6), (6, 8), (8, 21),
    (5, 7), (7, 9), (9, 22),
    (3, 19), (19, 20),
    (19, 10), (10, 12), (12, 14),
    (19, 11), (11, 13), (13, 15),
    (19, 16), (16, 17), (17, 18),
]

# All annotation subdirectory names in the Animal Kingdom dataset
ALL_ANN_DIRS = [
    "ak_P1",
    "ak_P2",
    "ak_P3_amphibian",
    "ak_P3_bird",
    "ak_P3_fish",
    "ak_P3_mammal",
    "ak_P3_reptile",
]

INPUT_SIZE   = (256, 256)
HEATMAP_SIZE = (64, 64)
SIGMA        = 2.0

# ---------------------------------------------------------------------------
# Training hyperparameters — change these constants to adjust defaults
# ---------------------------------------------------------------------------
EPOCHS     = 10
BATCH_SIZE = 16
LR         = 1e-3

# ---------------------------------------------------------------------------
# Model complexity — edit these to make models larger or smaller
# ---------------------------------------------------------------------------
# HRNet: channel widths for each of the 4 parallel resolution branches.
# Default  (small)  : [32,  64,  128, 256]  ~  1.2M params
# Medium            : [48,  96,  192, 384]  ~  2.7M params
# Large  (HRNet-W48): [48,  96,  192, 384] with more blocks — increase HRNET_BLOCKS
# Increasing channels has the biggest impact on accuracy and memory usage.
HRNET_CHANNELS = [32, 64, 128, 256]

# Number of BasicBlocks in each HRNet branch (default 4 per branch).
# More blocks = deeper network, slower training, potentially better accuracy.
# Range: 2 (very fast) to 8 (close to published HRNet depth)
HRNET_BLOCKS = 4

# ViTPose: transformer embedding dimension.
# Must be divisible by VITPOSE_HEADS.
# Small  (default) : 384  dim, 6  heads  ~  6M  params
# Medium           : 512  dim, 8  heads  ~  10M params
# Large  (ViT-B)   : 768  dim, 12 heads  ~  22M params
VITPOSE_DIM   = 384
VITPOSE_HEADS = 6

# Number of transformer blocks in ViTPose.
# Default 6. Published ViTPose-B uses 12.
# More blocks = more capacity but slower and more memory.
# Range: 2 (tiny/fast) to 12 (full ViT-Base depth)
VITPOSE_DEPTH = 6

# DEKR: base channel width of the backbone.
# The backbone goes: 3->BASE, BASE->BASE*2, then residual blocks at BASE*2 and BASE*4.
# Default  (small)  : 64   ~  1.5M params
# Medium            : 96   ~  3.3M params
# Large             : 128  ~  5.8M params (close to published DEKR with ResNet-50)
DEKR_BASE_CHANNELS = 64

# Number of ResBlocks at each stage of the DEKR backbone (default 2 per stage).
# Range: 1 (very fast) to 4 (deeper backbone)
DEKR_BLOCKS = 2

# ---------------------------------------------------------------------------
# Device — auto-detects GPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ===========================================================================
# 1.  Dataset
# ===========================================================================

def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _build_samples(
    coco: dict,
    image_base: Path,
    source_label: str = "",
    species_filter: Optional[List[str]] = None,
) -> List[dict]:
    """
    Parse an Animal Kingdom COCO-format annotation dict into sample dicts.

    Expected format:
      {
        "images":      [{"id": 0, "file_name": "VIDEOID/VIDEOID_fXXXXXX.jpg", ...}, ...],
        "annotations": [{"image_id": 0, "keypoints": [x,y,v,...], "bbox": [x,y,w,h],
                         "category_id": N, ...}, ...],
        "categories":  [{"id": N, "name": "horse", ...}, ...]
      }

    species_filter : optional list of species name strings to keep, e.g.
                     ["horse", "lion", "frog"].  Matching is case-insensitive
                     and substring-based, so "cat" matches "cat", "wildcat" etc.
                     If None or empty, all species are included.
    """
    # Build id -> image info lookup
    id2info: Dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Normalise species filter to lowercase for case-insensitive matching
    species_lower: Optional[List[str]] = (
        [s.lower() for s in species_filter]
        if species_filter else None
    )

    samples = []
    skipped_reasons: Dict[str, int] = {}

    def _skip(reason: str):
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    # Pre-collect all animal names present for informative warnings
    if species_lower:
        all_animals = sorted({
            ann.get("animal", "").lower()
            for ann in coco.get("annotations", [])
            if ann.get("animal")
        })
        matched_animals = [a for a in all_animals
                           if any(f in a for f in species_lower)]
        if not matched_animals:
            print(f"  WARNING [{source_label}]: no animals matched {species_filter}. "
                  f"Animals in this split: {all_animals}")
        else:
            print(f"  [{source_label}] Filtering to animals: {matched_animals}")

    for ann in coco.get("annotations", []):
        # ── Species filter — match against "animal" field on annotation ─────
        if species_lower:
            animal_name = ann.get("animal", "").lower()
            if not any(f in animal_name for f in species_lower):
                continue   # silently skip — filtered, not malformed

        # ── Image path ──────────────────────────────────────────────────────
        img_info = id2info.get(ann.get("image_id"))
        if img_info is None:
            _skip("image_id not found"); continue

        file_name = img_info["file_name"]   # e.g. "AAJYPNPL/AAJYPNPL_f000011.jpg"
        img_path  = image_base / file_name
        if not img_path.exists():
            img_path = image_base / Path(file_name).name
        if not img_path.exists():
            _skip("image file missing"); continue

        # ── Keypoints — COCO flat format [x, y, v, ...] ────────────────────
        raw_kps = ann.get("keypoints", [])
        if not raw_kps:
            _skip("no keypoints"); continue
        kps = np.array(raw_kps, dtype=np.float32).reshape(-1, 3)
        if kps.shape[0] != NUM_JOINTS:
            _skip(f"wrong joint count ({kps.shape[0]})"); continue

        # ── Bounding box [x, y, w, h] ───────────────────────────────────────
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            _skip("missing bbox"); continue

        samples.append(dict(
            img_path=str(img_path),
            keypoints=kps,
            bbox=list(bbox),
            animal=ann.get("animal", "Unknown"),
        ))

    total_skipped = sum(skipped_reasons.values())
    if total_skipped:
        label = f"[{source_label}] " if source_label else ""
        reasons = ", ".join(f"{v} {k}" for k, v in skipped_reasons.items())
        print(f"  {label}Skipped {total_skipped} annotations ({reasons}).")
    return samples

def _default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class AnimalKingdomDataset(Dataset):
    """
    Loads one annotation JSON (train.json or test.json) from one split.

    Parameters
    ----------
    ann_file   : full path to a train.json or test.json
    image_base : path to the images/ directory
    label      : human-readable label used in progress messages
    """

    def __init__(self, ann_file: str, image_base: str, label: str = "",
                 species_filter: Optional[List[str]] = None, transform=None):
        coco = _load_json(ann_file)
        self.samples   = _build_samples(coco, Path(image_base), label,
                                        species_filter=species_filter)
        self.transform = transform or _default_transform()
        print(f"  [{label or Path(ann_file).parent.name}] "
              f"{len(self.samples)} samples loaded from {Path(ann_file).name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = cv2.imread(s["img_path"])
        if img is None:
            raise FileNotFoundError(f"Image not found: {s['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x, y, w, h = [int(v) for v in s["bbox"]]
        x, y = max(0, x), max(0, y)
        crop = img[y: y + h, x: x + w]
        if crop.size == 0:
            crop = img
            x, y = 0, 0
            h, w = img.shape[:2]

        ih, iw = crop.shape[:2]
        crop_resized = cv2.resize(crop, (INPUT_SIZE[1], INPUT_SIZE[0]))

        kps = s["keypoints"].copy()
        kps[:, 0] = (kps[:, 0] - x) / max(iw, 1) * HEATMAP_SIZE[1]
        kps[:, 1] = (kps[:, 1] - y) / max(ih, 1) * HEATMAP_SIZE[0]

        heatmaps   = generate_heatmaps(kps, HEATMAP_SIZE, SIGMA)
        tensor_img = self.transform(crop_resized)
        # animal is a plain string — DataLoader collates these into a list of strings
        return tensor_img, torch.from_numpy(heatmaps), torch.from_numpy(kps), s["animal"]


def build_datasets(
    data_root: str,
    ann_dir_names: List[str],
    species_filter: Optional[List[str]] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Build combined train and test datasets from one or more annotation dirs.

    If only one directory is requested a plain AnimalKingdomDataset is returned.
    If multiple directories are requested their datasets are merged with
    ConcatDataset so the model sees all annotations in every epoch.

    Parameters
    ----------
    data_root      : root folder that contains both annotations/ and images/
    ann_dir_names  : list of annotation subdirectory names to include,
                     e.g. ["ak_P3_mammal", "ak_P3_bird"] or all of ALL_ANN_DIRS
    species_filter : optional list of species names to keep, e.g.
                     ["horse", "lion"].  Case-insensitive substring match
                     against the category name in the annotation JSON.
                     None means keep all species.
    """
    data_root  = Path(data_root)
    image_base = str(data_root / "images")
    ann_base   = data_root / "annotations"

    # Validate requested directories
    missing = [d for d in ann_dir_names if not (ann_base / d).exists()]
    if missing:
        available = [p.name for p in ann_base.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"Annotation dir(s) not found: {missing}\n"
            f"Available under {ann_base}: {available}"
        )

    train_datasets = []
    test_datasets  = []

    for name in ann_dir_names:
        ann_dir   = ann_base / name
        train_ann = str(ann_dir / "train.json")
        test_ann  = str(ann_dir / "test.json")

        if not Path(train_ann).exists():
            print(f"  WARNING: {train_ann} not found — skipping {name} train split.")
        else:
            train_datasets.append(AnimalKingdomDataset(
                train_ann, image_base, label=name, species_filter=species_filter))

        if not Path(test_ann).exists():
            print(f"  WARNING: {test_ann} not found — skipping {name} test split.")
        else:
            test_datasets.append(AnimalKingdomDataset(
                test_ann, image_base, label=name, species_filter=species_filter))

    if not train_datasets:
        raise RuntimeError("No valid training annotation files found.")
    if not test_datasets:
        raise RuntimeError("No valid test annotation files found.")

    train_ds = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    test_ds  = test_datasets[0]  if len(test_datasets)  == 1 else ConcatDataset(test_datasets)

    total_train = sum(len(d) for d in train_datasets)
    total_test  = sum(len(d) for d in test_datasets)
    print(f"\n  Combined train samples : {total_train}")
    print(f"  Combined test  samples : {total_test}\n")

    return train_ds, test_ds


# ===========================================================================
# 2.  Heatmap utilities
# ===========================================================================

def generate_heatmaps(keypoints: np.ndarray, size: Tuple[int, int], sigma: float) -> np.ndarray:
    H, W = size
    heatmaps = np.zeros((NUM_JOINTS, H, W), dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    for j, (kx, ky, vis) in enumerate(keypoints):
        if vis < 1:
            continue
        heatmaps[j] = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * sigma ** 2))
    return heatmaps


def heatmaps_to_coords(heatmaps: np.ndarray) -> np.ndarray:
    N, H, W = heatmaps.shape
    flat     = heatmaps.reshape(N, -1)
    idx      = flat.argmax(axis=1)
    ys, xs   = np.unravel_index(idx, (H, W))
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ===========================================================================
# 3.  Models
# ===========================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x)
        )


class HRNetBranch(nn.Module):
    def __init__(self, channels, num_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(*[BasicBlock(channels, channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


class HRNet(nn.Module):
    def __init__(self, num_joints=NUM_JOINTS,
                 channels=None, num_blocks=None):
        super().__init__()
        C = channels   or HRNET_CHANNELS
        B = num_blocks or HRNET_BLOCKS
        self.stem    = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1   = nn.Sequential(BasicBlock(64, C[0]), BasicBlock(C[0], C[0]))
        self.branch2  = HRNetBranch(C[1], num_blocks=B)
        self.branch3  = HRNetBranch(C[2], num_blocks=B)
        self.branch4  = HRNetBranch(C[3], num_blocks=B)
        self.down12   = nn.Sequential(nn.Conv2d(C[0], C[1], 3, 2, 1, bias=False), nn.BatchNorm2d(C[1]), nn.ReLU(inplace=True))
        self.down23   = nn.Sequential(nn.Conv2d(C[1], C[2], 3, 2, 1, bias=False), nn.BatchNorm2d(C[2]), nn.ReLU(inplace=True))
        self.down34   = nn.Sequential(nn.Conv2d(C[2], C[3], 3, 2, 1, bias=False), nn.BatchNorm2d(C[3]), nn.ReLU(inplace=True))
        self.fuse_up2 = nn.Sequential(nn.Conv2d(C[1], C[0], 1, bias=False), nn.BatchNorm2d(C[0]))
        self.fuse_up3 = nn.Sequential(nn.Conv2d(C[2], C[0], 1, bias=False), nn.BatchNorm2d(C[0]))
        self.fuse_up4 = nn.Sequential(nn.Conv2d(C[3], C[0], 1, bias=False), nn.BatchNorm2d(C[0]))
        self.head     = nn.Conv2d(C[0], num_joints, 1)

    def forward(self, x):
        s  = self.stem(x)
        b1 = self.layer1(s)
        b2 = self.branch2(self.down12(b1))
        b3 = self.branch3(self.down23(b2))
        b4 = self.branch4(self.down34(b3))
        H, W = b1.shape[2:]
        b2u = nn.functional.interpolate(self.fuse_up2(b2), (H, W), mode="bilinear", align_corners=False)
        b3u = nn.functional.interpolate(self.fuse_up3(b3), (H, W), mode="bilinear", align_corners=False)
        b4u = nn.functional.interpolate(self.fuse_up4(b4), (H, W), mode="bilinear", align_corners=False)
        return self.head(nn.functional.relu(b1 + b2u + b3u + b4u))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x  = x + self.attn(x2, x2, x2)[0]
        return x + self.mlp(self.norm2(x))


class ViTPose(nn.Module):
    def __init__(self, img_size=256, patch_size=16, num_joints=NUM_JOINTS,
                 embed_dim=None, depth=None, num_heads=None):
        embed_dim = embed_dim or VITPOSE_DIM
        depth     = depth     or VITPOSE_DEPTH
        num_heads = num_heads or VITPOSE_HEADS
        super().__init__()
        n = (img_size // patch_size) ** 2
        self.patch_embed      = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token        = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed        = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks           = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm             = nn.LayerNorm(embed_dim)
        self.feature_map_size = img_size // patch_size
        self.decode_proj      = nn.Conv2d(embed_dim, 256, 1)
        self.upsample         = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_joints, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B   = x.shape[0]
        tok = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed
        tok = self.norm(self.blocks(tok))[:, 1:]
        S   = self.feature_map_size
        feat = tok.transpose(1, 2).reshape(B, -1, S, S)
        return self.head(self.upsample(self.decode_proj(feat)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class DEKR(nn.Module):
    def __init__(self, num_joints=NUM_JOINTS,
                 base_channels=None, num_blocks=None):
        super().__init__()
        B  = base_channels or DEKR_BASE_CHANNELS
        NB = num_blocks    or DEKR_BLOCKS
        self.backbone = nn.Sequential(
            nn.Conv2d(3, B,    7, 2, 3, bias=False), nn.BatchNorm2d(B),    nn.ReLU(inplace=True),
            nn.Conv2d(B, B*2,  3, 2, 1, bias=False), nn.BatchNorm2d(B*2),  nn.ReLU(inplace=True),
            *[ResBlock(B*2) for _ in range(NB)],
            nn.Conv2d(B*2, B*4, 3, 1, 1, bias=False), nn.BatchNorm2d(B*4), nn.ReLU(inplace=True),
            *[ResBlock(B*4) for _ in range(NB)],
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(B*4, B*2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(B*2, num_joints, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(B*4, B*2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(B*2, num_joints * 2, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.heatmap_head(feat), self.offset_head(feat)


# ===========================================================================
# 4.  Loss, factory, metrics
# ===========================================================================

class MSEHeatmapLoss(nn.Module):
    def forward(self, pred, target, vis=None):
        diff = (pred - target) ** 2
        if vis is not None:
            diff = diff * vis.unsqueeze(-1).unsqueeze(-1).float()
        return diff.mean()


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "hrnet":
        model = HRNet()
        print(f"  HRNet  | channels={HRNET_CHANNELS}  blocks_per_branch={HRNET_BLOCKS}")
        return model
    if name == "vitpose":
        model = ViTPose()
        print(f"  ViTPose| dim={VITPOSE_DIM}  heads={VITPOSE_HEADS}  depth={VITPOSE_DEPTH}")
        return model
    if name == "dekr":
        model = DEKR()
        print(f"  DEKR   | base_channels={DEKR_BASE_CHANNELS}  blocks_per_stage={DEKR_BLOCKS}")
        return model
    raise ValueError(f"Unknown model '{name}'. Choose: hrnet | vitpose | dekr")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _forward(model_name: str, model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    out = model(imgs)
    return out[0] if model_name == "dekr" else out


def _align_gt(hm_pred: torch.Tensor, hm_gt: torch.Tensor) -> torch.Tensor:
    if hm_pred.shape[-2:] != hm_gt.shape[-2:]:
        hm_gt = nn.functional.interpolate(
            hm_gt, hm_pred.shape[-2:], mode="bilinear", align_corners=False
        )
    return hm_gt


def compute_pck_batch(hm_pred: np.ndarray, kps_gt: np.ndarray,
                      threshold: float = 0.2) -> Tuple[int, int]:
    B, J, H, W = hm_pred.shape
    correct = total = 0
    for b in range(B):
        pred_coords = heatmaps_to_coords(hm_pred[b])
        torso = (
            np.linalg.norm(kps_gt[b, 4, :2] - kps_gt[b, 10, :2]) +
            np.linalg.norm(kps_gt[b, 5, :2] - kps_gt[b, 11, :2])
        ) / 2 + 1e-6
        thresh_px = threshold * torso
        for j in range(J):
            if kps_gt[b, j, 2] < 1:
                continue
            correct += int(np.linalg.norm(pred_coords[j] - kps_gt[b, j, :2]) <= thresh_px)
            total   += 1
    return correct, total


def compute_per_joint_confusion(hm_pred: np.ndarray, kps_gt: np.ndarray,
                                threshold: float = 0.2) -> np.ndarray:
    B, J, H, W = hm_pred.shape
    conf = np.zeros((J, J), dtype=np.int64)
    for b in range(B):
        pred_coords = heatmaps_to_coords(hm_pred[b])
        torso = (
            np.linalg.norm(kps_gt[b, 4, :2] - kps_gt[b, 10, :2]) +
            np.linalg.norm(kps_gt[b, 5, :2] - kps_gt[b, 11, :2])
        ) / 2 + 1e-6
        thresh_px = threshold * torso
        for true_j in range(J):
            if kps_gt[b, true_j, 2] < 1:
                continue
            gt_pt  = kps_gt[b, true_j, :2]
            dists  = np.linalg.norm(pred_coords - gt_pt, axis=1)
            pred_j = int(dists.argmin())
            if dists[pred_j] <= thresh_px:
                conf[true_j, pred_j] += 1
    return conf


# ===========================================================================
# 5.  Training + evaluation loop
# ===========================================================================

def run_epoch(model_name, model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = total_joints = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, hm_gt, kps, animals in loader:
            # animals is a list of strings e.g. ["Horse", "Lion", "Frog", ...]
            imgs, hm_gt = imgs.to(device), hm_gt.to(device)
            vis = (kps[:, :, 2] > 0).to(device)

            if is_train:
                optimizer.zero_grad()

            hm_pred = _forward(model_name, model, imgs)
            hm_gt   = _align_gt(hm_pred, hm_gt)
            loss    = criterion(hm_pred, hm_gt, vis)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            c, t = compute_pck_batch(hm_pred.detach().cpu().numpy(), kps.numpy())
            total_correct += c
            total_joints  += t

    avg_loss = total_loss / len(loader)
    pck_pct  = 100.0 * total_correct / max(total_joints, 1)
    return avg_loss, pck_pct


def train_and_evaluate(
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    results_dir: Path,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float   = LR,
) -> Dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    model     = build_model(model_name).to(device)
    criterion = MSEHeatmapLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_params  = count_parameters(model)

    # Logger
    log_path = results_dir / f"{model_name}_train.log"
    logger   = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(
        f"Model: {model_name.upper()}\n"
        f"Model parameters: {n_params:,}\n"
        f"Training {model_name.upper()} on {device}...\n"
        f"{'-' * 60}"
    )

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model_name, model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = run_epoch(model_name, model, test_loader,  criterion, None,      device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        logger.info(
            f"Epoch [{epoch}/{epochs}]  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  "
            f"Test Loss: {te_loss:.4f}  Test Acc: {te_acc:.2f}%"
        )

        if te_loss < best_val_loss:
            best_val_loss = te_loss
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_loss": te_loss, "num_joints": NUM_JOINTS,
            }, results_dir / f"{model_name}_best.pth")

    elapsed = time.time() - t0
    logger.info(
        f"\nTraining completed in {elapsed:.1f} seconds\n"
        f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%"
    )

    _plot_curves(model_name, history, epochs, results_dir)
    _plot_confusion(model_name, model, test_loader, device, results_dir)

    fh.close()
    logger.removeHandler(fh)
    return history


# ===========================================================================
# 6.  Plotting
# ===========================================================================

def _plot_curves(model_name, history, epochs, out_dir):
    ep = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name.upper()} — Training Curves", fontsize=13, fontweight="bold")

    ax1.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["test_loss"],  "r-o", markersize=4, label="Test")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["test_acc"],  "r-o", markersize=4, label="Test")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("PCK@0.2 (%)"); ax2.set_title("Accuracy (PCK@0.2)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"{model_name}_loss_acc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved         -> {path}")


def _plot_confusion(model_name, model, test_loader, device, out_dir):
    conf = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for imgs, hm_gt, kps, animals in test_loader:
            hm_pred = _forward(model_name, model, imgs.to(device))
            conf   += compute_per_joint_confusion(hm_pred.cpu().numpy(), kps.numpy())

    row_sums  = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = conf.astype(float) / row_sums
    short     = [n.replace("_", "\n") for n in JOINT_NAMES]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction detected")
    ax.set_xticks(range(NUM_JOINTS)); ax.set_yticks(range(NUM_JOINTS))
    ax.set_xticklabels(short, fontsize=6, rotation=90)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted Joint", fontsize=10)
    ax.set_ylabel("True Joint",      fontsize=10)
    ax.set_title(
        f"{model_name.upper()} — Per-joint Detection Confusion Matrix\n"
        "(rows = true joint, cols = nearest predicted joint within PCK threshold)",
        fontsize=10, fontweight="bold"
    )
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            v = conf_norm[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if v > 0.6 else "black")
    fig.tight_layout()
    path = out_dir / f"{model_name}_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


# ===========================================================================
# 7.  Inference
# ===========================================================================

def load_model(model_name: str, checkpoint: str) -> nn.Module:
    state = torch.load(checkpoint, map_location="cpu")
    model = build_model(model_name)
    model.load_state_dict(state["model"])
    return model.eval().to(DEVICE)


def infer(model_name: str, checkpoint: str, image_path: str,
          output_path: Optional[str] = None):
    if output_path is None:
        output_path = f"{Path(image_path).stem}_{model_name}_pose.png"

    model   = load_model(model_name, checkpoint)
    img     = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (INPUT_SIZE[1], INPUT_SIZE[0]))
    tensor  = _default_transform()(resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)
    hmaps  = (out[0] if model_name == "dekr" else out).squeeze(0).cpu().numpy()
    coords = heatmaps_to_coords(hmaps)

    print(f"\n{'#':<4} {'Joint':<20} {'x':>6} {'y':>6}")
    print("-" * 40)
    for j, (x, y) in enumerate(coords):
        print(f"{j:<4} {JOINT_NAMES[j]:<20} {x:6.1f} {y:6.1f}")

    h, w       = resized.shape[:2]
    hm_h, hm_w = HEATMAP_SIZE
    pts = coords.copy()
    pts[:, 0] *= w / hm_w
    pts[:, 1] *= h / hm_h

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(resized)
    axes[0].set_title(f"{model_name.upper()} — Pose", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    cmap_sk = plt.cm.get_cmap("hsv", len(SKELETON))
    for k, (i, j) in enumerate(SKELETON):
        axes[0].plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                     "-", color=cmap_sk(k), linewidth=2.0, alpha=0.85)
    cmap_jt = plt.cm.get_cmap("tab20", NUM_JOINTS)
    for j in range(NUM_JOINTS):
        axes[0].scatter(pts[j, 0], pts[j, 1], c=[cmap_jt(j)],
                        s=45, zorder=5, edgecolors="white", linewidths=0.6)
    axes[1].axis("off")
    axes[1].legend(
        handles=[mpatches.Patch(color=cmap_jt(j), label=f"{j}: {JOINT_NAMES[j]}")
                 for j in range(NUM_JOINTS)],
        loc="upper left", fontsize=7, ncol=2, frameon=True, title="Joints"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pose overlay -> {output_path}")
    return coords


# ===========================================================================
# 8.  CLI
# ===========================================================================

def _resolve_ann_dirs(data_root: str, ann_dirs_arg: List[str]) -> List[str]:
    """
    Resolve the --ann_dirs argument into a concrete list of directory names.

    Passing "all" auto-discovers every subdirectory under annotations/ that
    contains at least a train.json, so it works even if you only downloaded
    a subset of the splits.
    """
    ann_base = Path(data_root) / "annotations"

    if ann_dirs_arg == ["all"]:
        discovered = sorted(
            d.name for d in ann_base.iterdir()
            if d.is_dir() and (d / "train.json").exists()
        )
        if not discovered:
            raise FileNotFoundError(
                f"No annotation subdirectories with train.json found under {ann_base}"
            )
        print(f"  Discovered annotation dirs: {discovered}")
        return discovered

    return ann_dirs_arg


def parse_args():
    p = argparse.ArgumentParser(
        description="Animal Kingdom limb tracking — train/evaluate/infer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ── train + evaluate (default) ──────────────────────────────────────────
    p.add_argument(
        "--data_root", default=None,
        help="Root folder containing images/ and annotations/",
    )
    p.add_argument(
        "--ann_dirs", nargs="+", default=["all"],
        metavar="DIR",
        help=(
            "Which annotation subdirectories to use.\n"
            "  all                     -> every dir found under annotations/\n"
            "  ak_P3_mammal            -> single split\n"
            "  ak_P3_mammal ak_P3_bird -> specific splits\n"
            "Default: all"
        ),
    )
    p.add_argument("--models", nargs="+", default=["hrnet", "vitpose", "dekr"],
                   choices=["hrnet", "vitpose", "dekr"])
    p.add_argument(
        "--species", nargs="+", default=None,
        metavar="SPECIES",
        help=(
            "Filter to specific species (case-insensitive substring match).\n"
            "Examples:\n"
            "  --species horse\n"
            "  --species horse lion frog crab\n"
            "  --species cat   (matches 'cat', 'wildcat', 'catfish', etc.)\n"
            "Omit to train on all species."
        ),
    )
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--num_workers", type=int,   default=4)

    # ── infer subcommand ────────────────────────────────────────────────────
    i = sub.add_parser("infer", help="Run inference on a single image")
    i.add_argument("--model",      required=True, choices=["hrnet", "vitpose", "dekr"])
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--image",      required=True)
    i.add_argument("--output",     default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.command == "infer":
        infer(args.model, args.checkpoint, args.image, args.output)
        return

    if args.data_root is None:
        print("ERROR: --data_root is required.")
        print("Example:")
        print("  python limb_tracking.py --data_root /path/to/ak --ann_dirs all")
        return

    ann_dir_names = _resolve_ann_dirs(args.data_root, args.ann_dirs)
    species = args.species or None

    print(f"\nDevice          : {DEVICE}")
    print(f"Annotation dirs : {ann_dir_names}")
    print(f"Species filter  : {species if species else 'all'}")
    print(f"Models          : {args.models}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}\n")

    train_ds, test_ds = build_datasets(args.data_root, ann_dir_names,
                                       species_filter=species)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=DEVICE.type == "cuda")
    test_dl  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          pin_memory=DEVICE.type == "cuda")

    results_dir   = Path(args.results_dir)
    all_histories = {}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'='*60}")
        all_histories[model_name] = train_and_evaluate(
            model_name   = model_name,
            train_loader = train_dl,
            test_loader  = test_dl,
            results_dir  = results_dir,
            device       = DEVICE,
            epochs       = args.epochs,
            lr           = args.lr,
        )

    # Combined comparison plot
    if len(args.models) > 1:
        ep = range(1, args.epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")
        colors = {"hrnet": "blue", "vitpose": "green", "dekr": "red"}
        for name, h in all_histories.items():
            c = colors.get(name, "black")
            ax1.plot(ep, h["test_loss"], "-o", color=c, markersize=4, label=name.upper())
            ax2.plot(ep, h["test_acc"],  "-o", color=c, markersize=4, label=name.upper())
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test MSE Loss")
        ax1.set_title("Test Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test PCK@0.2 (%)")
        ax2.set_title("Test Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        comp = results_dir / "model_comparison.png"
        fig.savefig(comp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nComparison plot -> {comp}")

    print("\nDone. Results in:", results_dir.resolve())


if __name__ == "__main__":
    main()