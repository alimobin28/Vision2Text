from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # prefix keeps env vars unambiguous: V2T_DATA_DIR, V2T_SEED, etc.
        env_prefix="V2T_",
    )

    # --- dataset ---
    dataset_name: str = "flickr8k"        # "flickr8k" or "flickr30k"
    data_dir: Path = Path("data/flickr8k")
    captions_file: str = "captions.txt"   # filename inside data_dir

    # --- artefact paths ---
    cache_dir: Path = Path("cache")
    vocab_path: Path = Path("cache/vocab.json")
    output_dir: Path = Path("outputs")

    # --- reproducibility ---
    seed: int = 42

    # --- images ---
    image_size: int = 224

    # --- model backbone ---
    # any torchvision.models name whose final layer is a classifier head
    backbone: str = "efficientnet_v2_s"

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
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 30

    # --- training ---
    batch_size: int = 32
    num_workers: int = 4


# module-level singleton — everything imports this
cfg = Config()
