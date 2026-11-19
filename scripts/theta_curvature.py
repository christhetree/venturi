"""
Computes the curvature of the training loss for each synth parameter (the 14
`DDSP808Synth` parameters predicted by the model) of the training setup of
`scripts/train.py`, using the theta-IS warmup (`warmup_lc_hvp`) of
`src/scrapl_jax/warmup.py`. Run from the repository root, e.g.:

    python scripts/theta_curvature.py -c train.yml
    python scripts/theta_curvature.py -c train.yml --seeds 0 1 2
    python scripts/theta_curvature.py -c train.yml --ckpt best
    python scripts/theta_curvature.py -c train.yml --n_batches 8 --n_iter 50 --tol 0

The data, model, synth, and loss are built from the config like `scripts/train.py`
(with `trainer_jax.setup`, so with the same seed handling, CPU batch size, and
overrides). For each synth parameter, the curvature is the magnitude of the dominant
eigenvalue (estimated by power iteration) of the curvature-vector product of the
training loss through that parameter w.r.t. all the model's weights, summed over the
first `n_batches` batches of the first training epoch. The model is evaluated in
inference mode (no dropout), since the warmup needs a deterministic encoder.

By default, the curvature is computed at the model's initialization (like the
theta-IS warmup, which runs before training); `--ckpt` uses a checkpoint saved by
`fit` instead. The results are printed and saved as JSON to `out/theta_curvature/`.

Without arguments (e.g. when run from an IDE), the defaults are the constants below,
which can be edited.
"""

import argparse
import json
import logging
import os
import sys
import time
import zlib
from typing import Any

import yaml

# So that `src` (and the `class_path`s of the configs) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

from src.experiments.util import (  # noqa: E402
    YamlLoader,
    load_config,
    set_config_value,
    setup_env,
)
from src.paths import CONFIGS_DIR, OUT_DIR  # noqa: E402

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

THETA_CURVATURE_DIR = os.path.join(OUT_DIR, "theta_curvature")

# Defaults of the command line arguments, e.g. for running from an IDE without any
CONFIG = "train.yml"  # Path, absolute or in configs/
SEEDS = None  # e.g. [0, 1, 2]; None for the config's `seed_everything`
CKPT = None  # None for the initialization, "best" for the best checkpoint of `fit`,
# or the path of a checkpoint
N_BATCHES = 4  # Training batches the curvature-vector products are summed over
N_ITER = 20  # Max power iteration steps per synth parameter
TOL = 0.01  # Stops power iteration once its relative residual is below; 0.0: never
VMAP_THETA = False  # All synth parameters in parallel: faster, but 14x the memory
OVERRIDES = []  # e.g. ["custom.cpu_batch_size=2"]


def calc_theta_curvature(
    config: dict[str, Any],
    seed: int,
    ckpt: str | None = CKPT,
    n_batches: int = N_BATCHES,
    n_iter: int = N_ITER,
    tol: float = TOL,
    vmap_theta: bool = VMAP_THETA,
) -> dict[str, Any]:
    """Estimates the curvature of the training loss for each synth parameter.

    `setup_env(config)` must have been called before, since this imports JAX.

    Args:
        config: A training config, as loaded by `scripts/train.py`.
        seed: The run seed (model initialization, batch order, and power iteration).
        ckpt: None for the model's initialization, "best" for the best checkpoint of
            `fit` for this config and seed, or the path of a checkpoint.
        n_batches: Training batches the curvature-vector products are summed over.
        n_iter: Max power iteration steps per synth parameter.
        tol: Stops power iteration once its relative residual is below `tol`.
        vmap_theta: Evaluates the synth parameters in parallel (more memory).

    Returns:
        The settings and, for each synth parameter (in the synth's order), its
        curvature and the relative residual of its last power iteration step.
    """
    # Not at the top level, since `setup_env` sets environment variables for JAX
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from src.experiments.trainer_jax import load_ckpt, setup, trainable
    from src.scrapl_jax.warmup import warmup_lc_hvp

    run = setup(config, seed)
    model, loss, data = run.model, run.loss, run.data
    ckpt_path, ckpt_epoch = None, None
    if ckpt is not None:
        ckpt_path = run.best_ckpt_path if ckpt == "best" else ckpt
        ckpt_path = ckpt_path.removesuffix(".eqx")
        opt_state_like = run.optim.init(trainable(model))
        model, _, meta = load_ckpt(ckpt_path, model, opt_state_like)
        ckpt_epoch = meta["epoch"]
        log.info(f"Loaded checkpoint {ckpt_path}.eqx from epoch {ckpt_epoch}")
    # Created by `setup`, unused here
    for d in [run.dir, os.path.dirname(run.dir)]:
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)

    synth = loss.synth
    param_names = synth.param_names
    assert model.n_params == len(param_names), (
        f"The model predicts {model.n_params} parameters, but the synth has "
        f"{len(param_names)}"
    )
    assert synth.buffer_noise, "The warmup needs a deterministic synth (buffer_noise)"
    n_train_batches = data.n_batches("train")
    assert (
        0 < n_batches <= n_train_batches
    ), f"n_batches must be in [1, {n_train_batches}]"

    @eqx.filter_jit
    def prepare(loss_: eqx.Module, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Like `DDSP808Loss.__call__`: the loss compares the synthesized audio with
        # the delayed target, whose CQT is the model's input (with no gradient)
        x = loss_.prepare_target(x)
        return x, loss_.calc_U(x)

    # The first batches of the first training epoch, as in `fit` with this seed
    xs, Us, deltas = [], [], []
    for _, (x, _, delta) in zip(range(n_batches), data.batches("train")):
        x, U = prepare(loss, jnp.asarray(x))
        xs.append(x)
        Us.append(U)
        deltas.append(jnp.asarray(delta))
    xs, Us, deltas = jnp.stack(xs), jnp.stack(Us), jnp.stack(deltas)

    # The warmup's encoder: the model's trainable weights (as in `fit`) to the synth
    # parameters, without dropout so that it is deterministic
    enc_params, enc_static = eqx.partition(
        eqx.nn.inference_mode(model), eqx.is_inexact_array
    )

    def enc_fn(params: Any, x: jax.Array, U: jax.Array) -> jax.Array:
        # The model's input is the precomputed CQT `U` of the target `x`, since the
        # loss has no gradient through it, so `x` itself is unused
        return jax.vmap(eqx.combine(params, enc_static))(U)

    def dec_fn(theta: jax.Array, delta: jax.Array) -> jax.Array:
        return synth(theta, delta)

    key = jax.random.fold_in(jax.random.key(seed), zlib.crc32(b"theta_curvature"))
    start = time.perf_counter()
    # A single loss (the training loss), so row 0 has the curvature of every theta
    result = warmup_lc_hvp(
        [loss.loss_func],
        enc_fn,
        dec_fn,
        xs,
        enc_params,
        enc_kwargs={"U": Us},
        dec_kwargs={"delta": deltas},
        key=key,
        n_iter=n_iter,
        tol=tol,
        vmap_theta=vmap_theta,
    )
    log.info(f"warmup_lc_hvp took {time.perf_counter() - start:.1f} s")
    curvatures = np.asarray(result.curvatures[0])
    residuals = np.asarray(result.relative_residuals[0])
    return {
        "run_name": run.name,
        "seed": seed,
        "ckpt": None if ckpt_path is None else f"{ckpt_path}.eqx",
        "ckpt_epoch": ckpt_epoch,
        "loss_func": type(loss.loss_func).__name__,
        "batch_size": int(xs.shape[1]),
        "n_batches": n_batches,
        "n_iter": n_iter,
        "tol": tol,
        "params": {
            name: {"curvature": float(c), "relative_residual": float(r)}
            for name, c, r in zip(param_names, curvatures, residuals)
        },
    }


def format_results(results: dict[str, Any]) -> str:
    tol = results["tol"]
    lines = [
        f"{'idx':>3}  {'param':<12} {'curvature':>12}  {'rel. residual':>13}",
    ]
    for idx, (name, vals) in enumerate(results["params"].items()):
        resid = vals["relative_residual"]
        # With tol = 0.0, power iteration always runs all n_iter steps
        flag = "  (not converged)" if 0 < tol <= resid else ""
        lines.append(
            f"{idx:>3}  {name:<12} {vals['curvature']:>12.4e}  {resid:>13.4f}{flag}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-c", "--config", default=CONFIG, help="Path, absolute or in configs/"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--ckpt",
        default=CKPT,
        help='"best" for the best checkpoint of fit, or the path of a checkpoint '
        "(default: the model's initialization)",
    )
    parser.add_argument("--n_batches", type=int, default=N_BATCHES)
    parser.add_argument("--n_iter", type=int, default=N_ITER)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument(
        "--vmap_theta", action=argparse.BooleanOptionalAction, default=VMAP_THETA
    )
    parser.add_argument(
        "--set",
        nargs="+",
        default=OVERRIDES,
        metavar="KEY=VALUE",
        help="Overrides, e.g. custom.cpu_batch_size=2",
    )
    parser.add_argument(
        "--out_dir", default=THETA_CURVATURE_DIR, help="Where the JSON is saved"
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

    log.info(f"Computing the theta curvature with config: {config_path}")
    seeds = args.seeds
    if seeds is None:
        seeds = [config.get("seed_everything", 0)]
    os.makedirs(args.out_dir, exist_ok=True)
    for seed in seeds:
        if args.seeds is not None:
            # Like `scripts/train.py`: each seed also sets the data split
            set_config_value(config, "data.init_args.shuffle_seed", seed)
        results = calc_theta_curvature(
            config,
            seed,
            ckpt=args.ckpt,
            n_batches=args.n_batches,
            n_iter=args.n_iter,
            tol=args.tol,
            vmap_theta=args.vmap_theta,
        )
        results = {"config": config_path, "overrides": args.set, **results}
        if results["ckpt"] is None:
            tag = "init"
        else:
            ckpt_name = os.path.basename(results["ckpt"]).removesuffix(".eqx")
            tag = f"{ckpt_name}_epoch_{results['ckpt_epoch']}"
        out_path = os.path.join(
            args.out_dir, f"{results['run_name']}__seed_{seed}__{tag}.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(
            f"Theta curvature of seed {seed} ({tag}, {results['loss_func']}, "
            f"{results['n_batches']} x {results['batch_size']} items):\n"
            f"{format_results(results)}\nSaved to {out_path}"
        )
