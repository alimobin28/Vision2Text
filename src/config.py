from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Fields that define the network. These travel with an exported model so the
# serving side rebuilds exactly the architecture that was trained, regardless
# of what the serving environment's V2T_* variables happen to say.
ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "backbone",
    "image_size",
    "vocab_size",
    "max_caption_len",
    "d_model",
    "nhead",
    "num_decoder_layers",
    "dim_feedforward",
    "dropout",
)

# Where each dataset keeps its caption file / images by default.
_DATASET_LAYOUTS: dict[str, tuple[str, str]] = {
    # name: (captions file, images sub-directory)
    "flickr8k": ("captions.txt", "Images"),
    "flickr30k": ("results.csv", "flickr30k_images"),
}


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # prefix keeps env vars unambiguous: V2T_DATA_DIR, V2T_SEED, etc.
        env_prefix="V2T_",
        # an unknown V2T_* key in .env should not stop the app from importing
        extra="ignore",
    )

    # --- dataset ---
    dataset_name: Literal["flickr8k", "flickr30k"] = "flickr8k"
    # The dataset lives OUTSIDE the python package (src/data is source code).
    data_dir: Path = Path("data")
    # Empty string = use the default layout for `dataset_name` (see above).
    # `captions_file` may point at a file, or at a directory that contains a
    # file of the same name (the Kaggle Flickr8k archive ships it that way).
    captions_file: str = ""
    images_subdir: str = ""
    # Held-out fractions, split BY IMAGE so the 5 captions of one photo never
    # straddle two splits. train = 1 - val_fraction - test_fraction.
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    # 0 = use every image. Handy for smoke runs: V2T_MAX_IMAGES=200.
    max_images: int = 0

    # --- artefact paths ---
    cache_dir: Path = Path("cache")
    vocab_path: Path = Path("cache/vocab.json")
    output_dir: Path = Path("outputs")

    # --- serving ---
    # Exported model produced by train.py (weights + architecture + tokenizer),
    # or a raw Lightning .ckpt.
    checkpoint: Path = Path("outputs/model.pt")
    device: str = "auto"          # "auto" | "cpu" | "cuda" | "mps"
    workers: int = 2              # inference threads in the API process

    # --- reproducibility ---
    seed: int = 42

    # --- images ---
    image_size: int = 224

    # --- model backbone ---
    # any torchvision model that exposes a convolutional `.features` trunk
    # (EfficientNet, MobileNet, ConvNeXt, RegNet, ...)
    backbone: str = "efficientnet_v2_s"
    # False = random init (tests / CI / serving: weights come from the checkpoint)
    pretrained_backbone: bool = True

    # --- text ---
    vocab_size: int = 8000
    max_caption_len: int = 40

    # --- model architecture ---
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # --- loss ---
    label_smoothing: float = 0.1

    # --- optimisation ---
    learning_rate: float = 1e-4
    # The pretrained CNN trunk learns at learning_rate * backbone_lr_mult;
    # the freshly initialised projection + decoder learn at learning_rate.
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 30
    early_stopping_patience: int = 5   # epochs without val_bleu4 gain; 0 = off

    # --- training ---
    batch_size: int = 32
    num_workers: int = 4
    augment: bool = True

    # ------------------------------------------------------------------
    # derived helpers
    # ------------------------------------------------------------------

    @property
    def captions_path(self) -> Path:
        """Resolved path of the caption file for the selected dataset."""
        name = self.captions_file or _DATASET_LAYOUTS[self.dataset_name][0]
        path = self.data_dir / name
        # Kaggle's Flickr8k archive nests it: <data_dir>/captions.txt/captions.txt
        if path.is_dir():
            path = path / Path(name).name
        return path

    @property
    def images_dir(self) -> Path:
        """Resolved directory holding the image files."""
        sub = self.images_subdir or _DATASET_LAYOUTS[self.dataset_name][1]
        return self.data_dir / sub

    def architecture(self) -> dict:
        """The subset of settings that determines the network's shape."""
        return {name: getattr(self, name) for name in ARCHITECTURE_FIELDS}


# module-level singleton — everything imports this
cfg = Config()
