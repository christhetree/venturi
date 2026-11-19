import logging
import os

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))

CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "out")

assert os.path.isdir(DATA_DIR)
assert os.path.isdir(OUT_DIR)

WANDB_LOGS_DIR = os.path.join(OUT_DIR, "wandb_logs")
JAX_CACHE_DIR = os.path.join(OUT_DIR, "jax_cache")  # JAX's compilation cache
AUDIO_SAVE_DIR = os.path.join(OUT_DIR, "audio_samples")
