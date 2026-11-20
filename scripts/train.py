"""
Trains with the JAX trainer (`src/experiments/trainer_jax.py`). Run from the
repository root, e.g.:

    python scripts/train.py -c train.yml
    python scripts/train.py -c train.yml --seeds 0 1 2
    python scripts/train.py -c train.yml \
        --set trainer.max_epochs=2 data.init_args.n_train=32
    python scripts/train.py -c train.yml --stages test

Without `--seeds`, a single run uses the config's `seed_everything`. With `--seeds`,
like the PyTorch `scripts/train.py` for the 808 experiments, each seed also sets the
data `shuffle_seed`, so every seed uses a different train / val / test split.

`--stages` runs `fit` (training) and / or `test` (of the best checkpoint of a
previous `fit` with the same config and seed); both by default.

`fit` saves to `out/jax_logs/<YYYY_MM_DD_HH_MM_SS>__<run name>/seed_<seed>`, with
the date and time this script started (`custom.run_timestamp`), so runs with the same
name do not overwrite each other. `test` uses the checkpoint of the `fit` of the same
call, or else of the latest run with the same name and seed. To test an earlier run:

    python scripts/train.py -c train.yml --stages test \
        --set custom.run_timestamp=2026_01_31_12_00_00

Without arguments (e.g. when run from an IDE), the defaults are the constants below
(`CONFIG`, `SEEDS`, `STAGES`, and `OVERRIDES`), which can be edited.
"""

import argparse
import logging
import os
import sys

import yaml

# So that `src` (and the `class_path`s of the configs) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

from src.experiments.util import (  # noqa: E402
    YamlLoader,
    load_config,
    set_config_value,
    setup_env,
)
from src.paths import CONFIGS_DIR  # noqa: E402

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Defaults of the command line arguments, e.g. for running from an IDE without any
CONFIG = "train.yml"  # Path, absolute or in configs/
SEEDS = None  # e.g. [0, 1, 2]; None for the config's `seed_everything`
STAGES = ["fit", "test"]
OVERRIDES = []  # e.g. ["trainer.max_epochs=2", "data.init_args.n_train=32"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-c", "--config", default=CONFIG, help="Path, absolute or in configs/"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--stages", nargs="+", choices=["fit", "test"], default=STAGES)
    parser.add_argument(
        "--set",
        nargs="+",
        default=OVERRIDES,
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

    # Imports JAX, so only after `setup_env`
    from src.experiments.trainer_jax import fit, new_run_timestamp, test

    config["custom"] = config.get("custom") or {}
    if "fit" in args.stages and not config["custom"].get("run_timestamp"):
        # The same run folder date and time for every seed and stage of this call
        config["custom"]["run_timestamp"] = new_run_timestamp()

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
