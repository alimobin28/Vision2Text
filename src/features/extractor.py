from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import torch
import torchvision.models as tv_models
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config


def _cache_key(cfg: Config) -> str:
    """12-char hex digest that uniquely identifies this extraction config.

    Changing backbone, image_size, or data_dir invalidates the cache and
    triggers a fresh extraction run automatically.
    """
    payload = json.dumps(
        {
            "backbone": cfg.backbone,
            "image_size": cfg.image_size,
            "data_dir": str(cfg.data_dir),
        },
        sort_keys=True,
    ).encode()
    return hashlib.md5(payload).hexdigest()[:12]


def _build_backbone(name: str) -> nn.Module:
    """Load a torchvision model and strip its classification head.

    Works with any model that exposes a top-level `classifier` or `fc`
    attribute (EfficientNet, ResNet, MobileNet, etc.).
    """
    weights_enum_name = f"{name.replace('_', '').upper()}_DEFAULT"  # best available
    try:
        weights = tv_models.get_model_weights(name).DEFAULT
    except Exception:
        weights = None  # fall back to random init if name is unusual

    model = tv_models.get_model(name, weights=weights)

    # remove the classification head so the model outputs feature vectors
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()

    model.eval()
    return model


def extract_features(
    cfg: Config,
    dataloader: DataLoader,
    image_ids: list[str],
) -> Path:
    """Extract CNN features for every image and cache them in an HDF5 file.

    Parameters
    ----------
    cfg:        project config (backbone, image_size, cache_dir)
    dataloader: yields (image_tensor, *_) batches in the same order as image_ids
    image_ids:  unique identifier for each image (e.g. filename stem)

    Returns
    -------
    Path to the HDF5 file (already exists if cache hit).
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(cfg)
    h5_path = cfg.cache_dir / f"features_{key}.h5"

    if h5_path.exists():
        return h5_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = _build_backbone(cfg.backbone).to(device)

    with h5py.File(h5_path, "w") as f, torch.inference_mode():
        idx = 0
        for batch in tqdm(dataloader, desc=f"Extracting {cfg.backbone} features"):
            images = batch[0].to(device)       # first element is always the image tensor
            feats = backbone(images)            # (B, feature_dim)
            feats_np = feats.cpu().numpy()

            for i in range(len(feats_np)):
                img_id = image_ids[idx]
                f.create_dataset(img_id, data=feats_np[i], compression="gzip")
                idx += 1

    return h5_path


def load_features(h5_path: Path, image_id: str) -> "np.ndarray":
    """Retrieve a single image's feature vector from the HDF5 cache."""
    with h5py.File(h5_path, "r") as f:
        return f[image_id][:]
