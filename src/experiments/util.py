"""
Config utilities for the JAX experiments: loading the YAML configs, overriding their
values, setting up JAX's environment from them, and instantiating their
`{class_path, init_args}` nodes.

This module does not import JAX at the top level, since `setup_env` sets environment
variables that JAX reads when it is imported.
"""

import importlib
import inspect
import logging
import os
import re
import zlib
from typing import TYPE_CHECKING, Any

import yaml

from src.paths import JAX_CACHE_DIR

if TYPE_CHECKING:
    import jax

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
    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        # Compiled steps are saved on disk and reused by later seeds and runs with the
        # same programs (only those that took at least 1 s to compile, JAX's default)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = JAX_CACHE_DIR
    log.info(
        f"Using the JAX compilation cache in {os.environ['JAX_COMPILATION_CACHE_DIR']}"
    )
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


def import_class(class_path: str) -> Any:
    """Imports e.g. `optax.adamw` or `src.experiments.models_jax.Spectral2DCNN`."""
    module_name, name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def instantiate(node: Any, key: "jax.Array", path: str = "") -> Any:
    """Recursively instantiates `{class_path, init_args}` config nodes.

    Classes whose `__init__` has a `key` argument receive a PRNG key derived from
    `key` and the node's config path, so each component's initialization only
    depends on the seed and on where it is in the config.
    """
    import jax  # Not at the top level, see the module docstring

    if isinstance(node, dict) and "class_path" in node:
        cls = import_class(node["class_path"])
        init_args = {
            k: instantiate(v, key, f"{path}.{k}")
            for k, v in (node.get("init_args") or {}).items()
        }
        if "key" in inspect.signature(cls).parameters and "key" not in init_args:
            init_args["key"] = jax.random.fold_in(key, zlib.crc32(path.encode()))
        return cls(**init_args)
    if isinstance(node, dict):
        return {k: instantiate(v, key, f"{path}.{k}") for k, v in node.items()}
    if isinstance(node, list):
        return [instantiate(v, key, f"{path}.{i}") for i, v in enumerate(node)]
    return node
