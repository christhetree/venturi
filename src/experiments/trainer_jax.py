"""
Minimal Lightning-style trainer for the JAX / Equinox experiments.

A run is described by the same kind of config as the PyTorch runs (`class_path` /
`init_args` nodes), loaded by `scripts/train.py`, with these main nodes:

- `model`: the trainable Equinox model (e.g.
  `src.experiments.models_jax.Spectral2DCNN`).
- `loss`: the fixed parts of the objective, called as `loss(model, x, delta, *, key)`
  and returning `(value, aux)`, where `aux` has the target `x` and the prediction
  `x_hat` (e.g. `src.experiments.losses_jax.DDSP808Loss`).
- `metrics` (optional): `{name: node}` of functions of `(x_hat, x)`, e.g. MSS losses,
  computed on the validation and test sets whatever the training loss.
- `data`: a data module with `batches(stage)`, `n_batches(stage)`, and `sr`.

The training state is `(model, opt_state)`; the loss is passed to the jitted steps
as an argument.

For a config and a seed, `fit` trains with AdamW (or any optax optimizer), validates
after every epoch, and keeps the checkpoint with the lowest validation loss; `test`
evaluates a checkpoint on the test set, also in a separate process. `fit` saves the
resolved config and the checkpoint to
`JAX_LOGS_DIR/<YYYY_MM_DD_HH_MM_SS>__<name>/seed_<seed>`, where the date and time
(`custom.run_timestamp`, the current ones by default) keep runs with the same name
from overwriting each other; see `setup` and `find_best_ckpt_path`. Both log to
Weights & Biases (see `wandb_logging.py`): the training loss and learning rate, the
validation loss and metrics, and the audio and plots of the first
`trainer.n_val_audio` validation items after every epoch. The summary of the run has
the validation values of the best checkpoint (`best/...`), to which `test` adds the
test loss and metrics (`test/...`). The test audio is saved to `AUDIO_SAVE_DIR`.
"""

import copy
import inspect
import json
import logging
import math
import os
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
import wandb
import yaml
from scipy.io import wavfile
from tqdm import tqdm

from src.experiments.util import import_class, instantiate
from src.experiments.wandb_logging import val_example_media, wandb_run
from src.paths import AUDIO_SAVE_DIR, OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

JAX_LOGS_DIR = os.path.join(OUT_DIR, "jax_logs")
RUN_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"  # At the start of the run folders
N_TEST_AUDIO = 128  # Test items saved as audio, as in `DDSP808LightingModule`


# Optimization =======================================================================
def make_optimizer(
    opt_config: dict[str, Any],
    sched_config: dict[str, Any] | None,
    steps_per_epoch: int,
    accumulate_grad_batches: int = 1,
) -> optax.GradientTransformation:
    """Builds an optax optimizer from a `{class_path, init_args}` node whose
    `init_args` include `learning_rate` (e.g. `optax.adamw`). The learning rate is
    injected into the optimizer state, so `current_lr` can read it without computing
    it again. The other hyperparameters stay compile-time constants (injecting them
    too would make them float32 arrays, which changes the rounding of the updates)."""
    init_args = dict(opt_config.get("init_args") or {})
    base_lr = init_args.pop("learning_rate")
    if sched_config:
        # torch.optim.lr_scheduler.LinearLR (same defaults), stepped once per epoch,
        # which is Lightning's default scheduler interval
        sched = {"start_factor": 1 / 3, "end_factor": 1.0, "total_iters": 5}
        sched.update(sched_config)
        per_epoch = optax.linear_schedule(
            base_lr * sched["start_factor"],
            base_lr * sched["end_factor"],
            sched["total_iters"],
        )
        schedule = lambda step: per_epoch(step // steps_per_epoch)
    else:
        schedule = optax.constant_schedule(base_lr)
    optim_cls = import_class(opt_config["class_path"])
    static_args = [
        p for p in inspect.signature(optim_cls).parameters if p != "learning_rate"
    ]
    optim_fn = optax.inject_hyperparams(optim_cls, static_args=static_args)
    optim = optim_fn(learning_rate=schedule, **init_args)
    if accumulate_grad_batches > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=accumulate_grad_batches)
    return optim


def current_lr(opt_state: Any) -> jax.Array:
    """The learning rate of the last update, from the state of `make_optimizer`."""
    if isinstance(opt_state, optax.MultiStepsState):
        opt_state = opt_state.inner_opt_state
    return opt_state.hyperparams["learning_rate"]


def trainable(model: eqx.Module) -> eqx.Module:
    return eqx.filter(model, eqx.is_inexact_array)


def make_train_step(optim: optax.GradientTransformation) -> Callable[..., Any]:
    """Returns the jitted training step. The loss is passed as an argument (not
    closed over) so its arrays, e.g. the CQT kernels, are not baked into the
    compiled program as constants."""

    @eqx.filter_jit
    def train_step(model, opt_state, loss, x, delta, key):
        # Differentiates w.r.t. the first argument of `loss`, i.e. the model
        (value, _), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, x, delta, key=key
        )
        updates, opt_state = optim.update(grads, opt_state, trainable(model))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value

    return train_step


@eqx.filter_jit
def eval_step(
    model: eqx.Module,
    loss: Callable[..., Any],
    metrics: dict[str, Callable[..., Any]],
    x: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Validation / test step, with the model in inference mode (no dropout): the
    loss and `metrics` (of the prediction and target), and the outputs of the loss.
    Defined once, so `fit` and `test` share its compilation."""
    value, out = loss(eqx.nn.inference_mode(model), x)
    values = {"loss": value}
    values.update({name: m(out["x_hat"], out["x"]) for name, m in metrics.items()})
    return values, out


# Checkpoints ========================================================================
def save_ckpt(
    path: str, model: eqx.Module, opt_state: Any, meta: dict[str, Any]
) -> None:
    eqx.tree_serialise_leaves(path + ".eqx", (model, opt_state))
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_ckpt(
    path: str, model_like: eqx.Module, opt_state_like: Any
) -> tuple[eqx.Module, Any, dict[str, Any]]:
    model, opt_state = eqx.tree_deserialise_leaves(
        path + ".eqx", (model_like, opt_state_like)
    )
    with open(path + ".json") as f:
        meta = json.load(f)
    return model, opt_state, meta


# Training ===========================================================================
def log_train_steps(
    wb: wandb.Run,
    pending: list[tuple[int, jax.Array, jax.Array]],
    accumulate: int,
    pbar: tqdm,
) -> None:
    """Logs `(n_batches_seen, loss, lr)` of the pending training steps, against the
    number of optimizer steps like Lightning. Their arrays are copied to the host
    together, which waits for those steps to finish."""
    if not pending:
        return
    values = jax.device_get([(loss, lr) for _, loss, lr in pending])
    for (n_batches_seen, _, _), (loss, lr) in zip(pending, values):
        step = n_batches_seen // accumulate
        wb.log({"train/loss": float(loss), "train/lr": float(lr)}, step=step)
    loss, lr = values[-1]
    pbar.set_postfix(loss=f"{float(loss):.4f}", lr=f"{float(lr):.2e}")


def evaluate(
    model: eqx.Module,
    loss: Callable[..., Any],
    metrics: dict[str, Callable[..., Any]],
    data: Any,
    stage: str,
    n_outputs: int = 0,
) -> tuple[dict[str, float], list[dict[str, np.ndarray]]]:
    """Means of the loss and `metrics` over the batches of `stage` (as a dict with
    `"loss"` and the names of the metrics), and the outputs of (at least) the first
    `n_outputs` items. Everything is copied to the host once, at the end, so the
    batches run without waiting for each other."""
    values, outputs, n_kept = [], [], 0
    for x, _, _ in data.batches(stage):
        batch_values, out = eval_step(model, loss, metrics, x)
        values.append(batch_values)
        if n_kept < n_outputs:
            outputs.append(out)
            n_kept += x.shape[0]
    values, outputs = jax.device_get((values, outputs))
    means = {
        k: float(np.mean(np.asarray([v[k] for v in values], dtype=np.float64)))
        for k in values[0]
    }
    return means, outputs


def format_metrics(stage: str, values: dict[str, float]) -> str:
    return ", ".join(f"{stage}/{k} = {v:.6f}" for k, v in values.items())


def save_test_audio(
    outputs: list[dict[str, np.ndarray]],
    run_name: str,
    seed: int,
    sr: int,
    max_n: int = 128,
) -> None:
    """Saves the test targets and predictions with the same file names as
    `DDSP808LightingModule`."""
    os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
    x = np.concatenate([o["x"] for o in outputs])[:max_n]
    x_hat = np.concatenate([o["x_hat"] for o in outputs])[:max_n]
    for idx in range(x.shape[0]):
        name = f"{run_name}__seed_{seed}__test__{idx}"
        wavfile.write(os.path.join(AUDIO_SAVE_DIR, f"{name}.wav"), sr, x[idx, 0])
        wavfile.write(
            os.path.join(AUDIO_SAVE_DIR, f"{name}__hat.wav"), sr, x_hat[idx, 0]
        )


@dataclass
class Run:
    """What `fit` and `test` build from a config and a seed."""

    name: str
    dir: str  # The resolved config and checkpoints, created by `fit`
    seed: int
    config: dict[str, Any]  # With the CPU settings, data seed, and timestamp applied
    data: Any
    model: eqx.Module  # Freshly initialized
    loss: Callable[..., Any]
    metrics: dict[str, Callable[..., Any]]
    optim: optax.GradientTransformation
    accumulate: int
    # Whether the date and time in `dir` are the current ones, rather than those of
    # `custom.run_timestamp`
    is_new_timestamp: bool

    @property
    def best_ckpt_path(self) -> str:
        return os.path.join(self.dir, "best")


def new_run_timestamp() -> str:
    """The current date and time, as at the start of the run folders."""
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)


def parse_run_timestamp(value: str | int) -> str:
    """Checks a `custom.run_timestamp`, which YAML reads as an integer when it is
    not quoted (e.g. 2026_01_31_12_00_00 as 20260131120000, since underscores are
    digit separators in YAML 1.1), and returns it as YYYY_MM_DD_HH_MM_SS."""
    timestamp = str(value)
    if re.fullmatch(r"\d{14}", timestamp):
        parts = [timestamp[:4]] + [timestamp[i : i + 2] for i in range(4, 14, 2)]
        timestamp = "_".join(parts)
    try:
        datetime.strptime(timestamp, RUN_TIMESTAMP_FORMAT)
    except ValueError:
        raise ValueError(
            f"custom.run_timestamp must be YYYY_MM_DD_HH_MM_SS, not {value!r}"
        ) from None
    return timestamp


def setup(config: dict[str, Any], seed: int) -> Run:
    """Builds the data, model, loss, and optimizer of a run. Everything is derived
    from the config and the seed, so `fit` and `test` build identical runs.

    The run's folder starts with `custom.run_timestamp`, or the current date and time
    if it is not set. `scripts/train.py` sets it once, so that the seeds and stages
    of one call share it, and `test` finds the checkpoint saved by `fit`."""
    config = copy.deepcopy(config)
    custom = config.get("custom") or {}
    config["custom"] = custom
    trainer_cfg = config.get("trainer") or {}
    accumulate = trainer_cfg.get("accumulate_grad_batches", 1)
    name = custom.get("run_name") or (
        f"{custom.get('model_name', 'model')}__{custom.get('dataset_name', 'data')}"
    )

    # Like the PyTorch CLI: smaller batches when there is no GPU
    on_cpu = jax.default_backend() == "cpu"
    data_args = config["data"].setdefault("init_args", {})
    if on_cpu and "cpu_batch_size" in custom:
        data_args["batch_size"] = custom["cpu_batch_size"]
    data_args["seed"] = seed

    is_new_timestamp = not custom.get("run_timestamp")
    if is_new_timestamp:
        custom["run_timestamp"] = new_run_timestamp()
    else:
        custom["run_timestamp"] = parse_run_timestamp(custom["run_timestamp"])
    run_dir = os.path.join(
        JAX_LOGS_DIR, f"{custom['run_timestamp']}__{name}", f"seed_{seed}"
    )
    log.info(f"Run {name}, seed {seed}, backend {jax.default_backend()}: {run_dir}")

    key = jax.random.key(seed)
    data = instantiate(config["data"], key, "data")
    model = instantiate(config["model"], key, "model")
    loss = instantiate(config["loss"], key, "loss")
    metrics = instantiate(config.get("metrics") or {}, key, "metrics")
    steps_per_epoch = data.n_batches("train") // accumulate
    optim = make_optimizer(
        config["optimizer"], config.get("lr_scheduler"), steps_per_epoch, accumulate
    )
    return Run(
        name,
        run_dir,
        seed,
        config,
        data,
        model,
        loss,
        metrics,
        optim,
        accumulate,
        is_new_timestamp,
    )


def find_best_ckpt_path(run: Run) -> str:
    """The path (without `.eqx`) of the best checkpoint saved by `fit` for the config
    and seed of `run`: in the folder of `run` if `custom.run_timestamp` is set (e.g.
    by `scripts/train.py` before `fit`), else in the latest folder with the same
    run name and seed that has one."""
    if not run.is_new_timestamp:
        return run.best_ckpt_path
    # The run folders with the same name, oldest first: the one from before the
    # folders started with the date and time, then the others by date and time
    pattern = re.compile(r"\d{4}(_\d{2}){5}__" + re.escape(run.name))
    run_dirs = [run.name]
    if os.path.isdir(JAX_LOGS_DIR):
        run_dirs += sorted(d for d in os.listdir(JAX_LOGS_DIR) if pattern.fullmatch(d))
    for run_dir in reversed(run_dirs):
        path = os.path.join(JAX_LOGS_DIR, run_dir, f"seed_{run.seed}", "best")
        if os.path.isfile(f"{path}.eqx"):
            log.info(
                f"Using the latest checkpoint of {run.name}, seed {run.seed}: "
                f"{path}.eqx"
            )
            return path
    raise FileNotFoundError(
        f"No checkpoint of {run.name}, seed {run.seed} in {JAX_LOGS_DIR}"
    )


def fit(config: dict[str, Any], seed: int) -> float:
    """Trains, validating after every epoch, and saves the checkpoint with the lowest
    validation loss to `<run dir>/best.eqx`. Returns that validation loss."""
    run = setup(config, seed)
    trainer_cfg = run.config.get("trainer") or {}
    max_epochs = trainer_cfg.get("max_epochs", 1)
    # Training losses are copied to the host (which waits for the GPU) only every
    # `log_every_n_steps` steps and at the end of each epoch
    log_every_n_steps = trainer_cfg.get("log_every_n_steps", 50)
    # Validation items logged as audio and plots after every epoch
    n_val_audio = trainer_cfg.get("n_val_audio", 5)
    os.makedirs(run.dir, exist_ok=True)
    with open(os.path.join(run.dir, "config.yml"), "w") as f:
        yaml.safe_dump(run.config, f, sort_keys=False)

    model, loss, data, accumulate = run.model, run.loss, run.data, run.accumulate
    n_params = sum(p.size for p in jax.tree.leaves(trainable(model)))
    log.info(f"Trainable parameters: {n_params:,}")
    if getattr(loss, "use_curvature", False):  # E.g. in `DDSP808Loss`
        log.info(loss.describe_curvature_scaling())
    opt_state = run.optim.init(trainable(model))
    train_step = make_train_step(run.optim)

    train_key = jax.random.fold_in(jax.random.key(seed), zlib.crc32(b"train"))
    n_batches_seen = 0
    best_val_loss = math.inf
    with wandb_run(run.config, run.name, seed) as wb:
        for epoch in range(max_epochs):
            pbar = tqdm(
                data.batches("train"),
                total=data.n_batches("train"),
                desc=f"seed {seed} epoch {epoch}",
            )
            pending = []  # Device arrays of the training steps not logged yet
            for x, _, delta in pbar:
                step_key = jax.random.fold_in(train_key, n_batches_seen)
                model, opt_state, train_loss = train_step(
                    model, opt_state, loss, x, delta, step_key
                )
                n_batches_seen += 1
                pending.append((n_batches_seen, train_loss, current_lr(opt_state)))
                if len(pending) == log_every_n_steps:
                    log_train_steps(wb, pending, accumulate, pbar)
                    pending = []
            log_train_steps(wb, pending, accumulate, pbar)

            val_values, val_outputs = evaluate(
                model, loss, run.metrics, data, "val", n_outputs=n_val_audio
            )
            step = n_batches_seen // accumulate
            to_log = {f"val/{k}": v for k, v in val_values.items()}
            to_log["epoch"] = epoch
            if not wb.disabled:
                to_log.update(
                    val_example_media(
                        val_outputs, n_val_audio, data.sr, step, epoch == 0
                    )
                )
            wb.log(to_log, step=step)

            val_loss = val_values["loss"]
            is_best = val_loss < best_val_loss
            log.info(
                f"seed {seed} epoch {epoch}: {format_metrics('val', val_values)}"
                f"{' (best)' * is_best}"
            )
            if is_best:
                best_val_loss = val_loss
                meta = {
                    "seed": seed,
                    "epoch": epoch,
                    "step": step,
                    "val": val_values,
                    "wandb_id": None if wb.disabled else wb.id,
                }
                save_ckpt(run.best_ckpt_path, model, opt_state, meta)
                # The validation values of the best checkpoint (the one `test` uses
                # by default), all from the same epoch
                best = {f"best/val/{k}": v for k, v in val_values.items()}
                wb.summary.update({"best/epoch": epoch, "best/step": step, **best})

    return best_val_loss


def test(config: dict[str, Any], seed: int, ckpt_path: str | None = None) -> float:
    """Tests a checkpoint, by default the best one that `fit` saved for this config
    and seed (see `find_best_ckpt_path`): adds the test loss and metrics to the
    summary of its wandb run (resumed from the checkpoint's metadata) and saves the
    test audio. Can run in a different process from `fit`. Returns the test loss."""
    run = setup(config, seed)
    ckpt_path = (ckpt_path or find_best_ckpt_path(run)).removesuffix(".eqx")
    opt_state_like = run.optim.init(trainable(run.model))
    model, _, meta = load_ckpt(ckpt_path, run.model, opt_state_like)
    test_values, outputs = evaluate(
        model, run.loss, run.metrics, run.data, "test", n_outputs=N_TEST_AUDIO
    )
    log.info(
        f"seed {seed}: {format_metrics('test', test_values)} "
        f"(checkpoint {ckpt_path}.eqx from epoch {meta['epoch']})"
    )
    with wandb_run(run.config, run.name, seed, resume_id=meta.get("wandb_id")) as wb:
        wb.summary.update({f"test/{k}": v for k, v in test_values.items()})
        wb.summary.update({"test/ckpt_epoch": meta["epoch"]})
    save_test_audio(outputs, run.name, seed, sr=run.data.sr, max_n=N_TEST_AUDIO)
    return test_values["loss"]
