"""
Trains with the JAX trainer (`src/trainer_jax.py`), the JAX counterpart of
`scripts/train.py`. Run from the experiments directory, e.g.:

    python scripts/train_jax.py -c eval_808/train_mss_log_micro_jax.yml
    python scripts/train_jax.py -c eval_808/train_mss_log_micro_jax.yml --seeds 0 1 2
    python scripts/train_jax.py -c eval_808/train_mss_log_micro_jax.yml \
        --set trainer.max_epochs=2 data.init_args.n_train=32
    python scripts/train_jax.py -c eval_808/train_mss_log_micro_jax.yml --stages test

Without `--seeds`, a single run uses the config's `seed_everything`. With `--seeds`,
like `scripts/train.py` for the 808 experiments, each seed also sets the data
`shuffle_seed`, so every seed uses a different train / val / test split.

`--stages` runs `fit` (training) and / or `test` (of the best checkpoint of a
previous `fit` with the same config and seed); both by default.
"""

import argparse
import logging
import os
import re
import sys
from typing import Any

import yaml

EXPERIMENTS_DIR = os.path.abspath(os.path.join(__file__, "../../"))
CONFIGS_DIR = os.path.join(EXPERIMENTS_DIR, "configs")
# For `src` / `eval_808`, and for `losses_jax` (the JAX MSS losses in ../scripts)
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.append(os.path.join(EXPERIMENTS_DIR, "..", "scripts"))

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class YamlLoader(yaml.SafeLoader):
    """SafeLoader that also reads exponents without a dot (e.g. 1e-4) as floats, as
    jsonargparse does for the PyTorch configs."""


YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_config(path: str) -> dict[str, Any]:
    """Loads a YAML config, replacing `.yml` / `.yaml` strings by the contents of
    those files (paths relative to the file that references them)."""
    with open(path) as f:
        cfg = yaml.load(f, Loader=YamlLoader)

    def resolve(node: Any, base_dir: str) -> Any:
        if isinstance(node, dict):
            return {k: resolve(v, base_dir) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, base_dir) for v in node]
        if isinstance(node, str) and node.endswith((".yml", ".yaml")):
            return load_config(os.path.join(base_dir, node))
        return node

    return resolve(cfg, os.path.dirname(path))


def set_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Sets e.g. `trainer.max_epochs` in a nested config dict."""
    *parents, last = dotted_key.split(".")
    node = config
    for k in parents:
        node = node.setdefault(k, {})
    node[last] = value


def setup_env(config: dict[str, Any]) -> None:
    """Sets the environment variables that JAX reads when it is imported or
    initializes the GPU, so this must run before JAX is imported."""
    trainer_cfg = config.get("trainer") or {}
    cache_dir = trainer_cfg.get("compilation_cache_dir")
    if cache_dir and "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        # Compiled steps are saved on disk and reused by later seeds and runs with the
        # same programs (only those that took at least 1 s to compile, JAX's default)
        cache_dir = os.path.join(EXPERIMENTS_DIR, os.path.expanduser(cache_dir))
        os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.abspath(cache_dir)
        log.info(f"Using the JAX compilation cache in {cache_dir}")
    devices = trainer_cfg.get("devices")
    if isinstance(devices, list) and "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Otherwise JAX preallocates memory on every visible GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        log.info(f"Setting CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    if (config.get("custom") or {}).get("is_deterministic", True):
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_gpu_deterministic_ops" not in flags:
            os.environ["XLA_FLAGS"] = (
                f"{flags} --xla_gpu_deterministic_ops=true".strip()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-c", "--config", required=True, help="Path, absolute or in configs/"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--stages", nargs="+", choices=["fit", "test"], default=["fit", "test"]
    )
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Overrides, e.g. trainer.max_epochs=2",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        config_path = os.path.join(CONFIGS_DIR, config_path)
    config = load_config(config_path)
    for override in args.set:
        k, v = override.split("=", 1)
        set_config_value(config, k, yaml.load(v, Loader=YamlLoader))
    setup_env(config)

    from src.trainer_jax import fit, test

    log.info(f"Running {' and '.join(args.stages)} with config: {config_path}")
    seeds = args.seeds
    if seeds is None:
        seeds = [config.get("seed_everything", 0)]
    else:
        log.info(f"Running with seeds: {seeds}")
    for seed in seeds:
        if args.seeds is not None:
            set_config_value(config, "data.init_args.shuffle_seed", seed)
        if "fit" in args.stages:
            fit(config, seed=seed)
        if "test" in args.stages:
            test(config, seed=seed)
