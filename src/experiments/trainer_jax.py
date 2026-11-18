"""
Minimal Lightning-style trainer for the JAX / Equinox experiments.

A run is described by the same kind of config as the PyTorch runs (`class_path` /
`init_args` nodes), loaded by `scripts/train.py`, with three main nodes:

- `model`: the trainable Equinox model (e.g.
  `src.experiments.models_jax.Spectral2DCNN`).
- `loss`: the fixed parts of the objective, called as `loss(model, x, delta, *, key)`
  and returning `(value, aux)` (e.g. `src.experiments.losses_jax.DDSP808Loss`).
- `data`: a data module with `batches(stage)`, `n_batches(stage)`, and `sr`.

The training state is `(model, opt_state)`; the loss is passed to the jitted steps
as an argument.

For a config and a seed, `fit` trains with AdamW (or any optax optimizer), validates
after every epoch, and keeps the checkpoint with the lowest validation loss; `test`
evaluates a checkpoint on the test set, also in a separate process.
Losses, and the audio of the first `trainer.n_val_audio` validation items after every
epoch, are logged to TensorBoard (event files in the run directory), viewable with
`tensorboard --logdir out/jax_logs --samples_per_plugin audio=1000` (by default,
TensorBoard only keeps 10 random steps of each audio tag). The test audio is saved to
`AUDIO_SAVE_DIR`.
"""

import copy
import inspect
import json
import logging
import math
import os
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
import yaml
from scipy.io import wavfile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.experiments.util import import_class, instantiate
from src.paths import AUDIO_SAVE_DIR, OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

JAX_LOGS_DIR = os.path.join(OUT_DIR, "jax_logs")
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
    model: eqx.Module, loss: Callable[..., Any], x: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Validation / test step: the loss and outputs of the model in inference mode
    (no dropout). Defined once, so `fit` and `test` share its compilation."""
    return loss(eqx.nn.inference_mode(model), x)


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
    writer: SummaryWriter,
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
        writer.add_scalar("train/loss", float(loss), step)
        writer.add_scalar("train/lr", float(lr), step)
    loss, lr = values[-1]
    pbar.set_postfix(loss=f"{float(loss):.4f}", lr=f"{float(lr):.2e}")


def evaluate(
    model: eqx.Module,
    loss: Callable[..., Any],
    data: Any,
    stage: str,
    n_outputs: int = 0,
) -> tuple[float, list[dict[str, np.ndarray]]]:
    """Mean loss over the batches of `stage`, and the outputs of (at least) the first
    `n_outputs` items. Everything is copied to the host once, at the end, so the
    batches run without waiting for each other."""
    values, outputs, n_kept = [], [], 0
    for x, _, _ in data.batches(stage):
        value, out = eval_step(model, loss, x)
        values.append(value)
        if n_kept < n_outputs:
            outputs.append(out)
            n_kept += x.shape[0]
    values, outputs = jax.device_get((values, outputs))
    return float(np.mean(np.asarray(values, dtype=np.float64))), outputs


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


def log_val_audio(
    writer: SummaryWriter,
    outputs: list[dict[str, np.ndarray]],
    n: int,
    step: int,
    sr: int,
    log_targets: bool,
) -> None:
    """Logs the first `n` validation items to TensorBoard's Audio tab: the
    predictions (`val_audio/<idx>/prediction`) and, if `log_targets`, the targets
    (`val_audio/<idx>/target`). The validation set is not shuffled, so `<idx>` is the
    same item at every epoch."""
    if n <= 0 or not outputs:
        return
    x = np.concatenate([o["x"] for o in outputs])[:n]
    x_hat = np.concatenate([o["x_hat"] for o in outputs])[:n]
    for idx in range(x.shape[0]):
        if log_targets:
            writer.add_audio(f"val_audio/{idx}/target", x[idx], step, sample_rate=sr)
        writer.add_audio(
            f"val_audio/{idx}/prediction", x_hat[idx], step, sample_rate=sr
        )


@dataclass
class Run:
    """What `fit` and `test` build from a config and a seed."""

    name: str
    dir: str  # TensorBoard logs, the resolved config, and checkpoints
    seed: int
    config: dict[str, Any]  # With the CPU batch size and the data seed applied
    data: Any
    model: eqx.Module  # Freshly initialized
    loss: Callable[..., Any]
    optim: optax.GradientTransformation
    accumulate: int

    @property
    def best_ckpt_path(self) -> str:
        return os.path.join(self.dir, "best")


def setup(config: dict[str, Any], seed: int) -> Run:
    """Builds the data, model, loss, and optimizer of a run. Everything is derived
    from the config and the seed, so `fit` and `test` build identical runs."""
    config = copy.deepcopy(config)
    custom = config.get("custom") or {}
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

    run_dir = os.path.join(JAX_LOGS_DIR, name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log.info(f"Run {name}, seed {seed}, backend {jax.default_backend()}: {run_dir}")

    key = jax.random.key(seed)
    data = instantiate(config["data"], key, "data")
    model = instantiate(config["model"], key, "model")
    loss = instantiate(config["loss"], key, "loss")
    steps_per_epoch = data.n_batches("train") // accumulate
    optim = make_optimizer(
        config["optimizer"], config.get("lr_scheduler"), steps_per_epoch, accumulate
    )
    return Run(name, run_dir, seed, config, data, model, loss, optim, accumulate)


def fit(config: dict[str, Any], seed: int) -> float:
    """Trains, validating after every epoch, and saves the checkpoint with the lowest
    validation loss to `<run dir>/best.eqx`. Returns that validation loss."""
    run = setup(config, seed)
    trainer_cfg = run.config.get("trainer") or {}
    max_epochs = trainer_cfg.get("max_epochs", 1)
    # Training losses are copied to the host (which waits for the GPU) only every
    # `log_every_n_steps` steps and at the end of each epoch
    log_every_n_steps = trainer_cfg.get("log_every_n_steps", 50)
    # Validation items logged as audio after every epoch (the targets only once)
    n_val_audio = trainer_cfg.get("n_val_audio", 4)
    with open(os.path.join(run.dir, "config.yml"), "w") as f:
        yaml.safe_dump(run.config, f, sort_keys=False)
    writer = SummaryWriter(log_dir=run.dir)

    model, loss, data, accumulate = run.model, run.loss, run.data, run.accumulate
    n_params = sum(p.size for p in jax.tree.leaves(trainable(model)))
    log.info(f"Trainable parameters: {n_params:,}")
    opt_state = run.optim.init(trainable(model))
    train_step = make_train_step(run.optim)

    train_key = jax.random.fold_in(jax.random.key(seed), zlib.crc32(b"train"))
    n_batches_seen = 0
    best_val_loss = math.inf
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
                log_train_steps(writer, pending, accumulate, pbar)
                pending = []
        log_train_steps(writer, pending, accumulate, pbar)

        val_loss, val_outputs = evaluate(
            model, loss, data, "val", n_outputs=n_val_audio
        )
        step = n_batches_seen // accumulate
        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("epoch", epoch, step)
        log_val_audio(
            writer, val_outputs, n_val_audio, step, data.sr, log_targets=epoch == 0
        )
        writer.flush()
        is_best = val_loss < best_val_loss
        log.info(
            f"seed {seed} epoch {epoch}: val/loss = {val_loss:.6f}{' (best)' * is_best}"
        )
        if is_best:
            best_val_loss = val_loss
            meta = {"seed": seed, "epoch": epoch, "step": step, "val_loss": val_loss}
            save_ckpt(run.best_ckpt_path, model, opt_state, meta)

    writer.close()
    return best_val_loss


def test(config: dict[str, Any], seed: int, ckpt_path: str | None = None) -> float:
    """Tests a checkpoint, by default the best one that `fit` saved for this config
    and seed: logs `test/loss` to TensorBoard and saves the test audio. Can run in a
    different process from `fit`. Returns the test loss."""
    run = setup(config, seed)
    ckpt_path = (ckpt_path or run.best_ckpt_path).removesuffix(".eqx")
    opt_state_like = run.optim.init(trainable(run.model))
    model, _, meta = load_ckpt(ckpt_path, run.model, opt_state_like)
    test_loss, outputs = evaluate(
        model, run.loss, run.data, "test", n_outputs=N_TEST_AUDIO
    )
    with SummaryWriter(log_dir=run.dir) as writer:
        writer.add_scalar("test/loss", test_loss, meta["step"])
    log.info(
        f"seed {seed}: test/loss = {test_loss:.6f} "
        f"(checkpoint {ckpt_path}.eqx from epoch {meta['epoch']})"
    )
    save_test_audio(outputs, run.name, seed, sr=run.data.sr, max_n=N_TEST_AUDIO)
    return test_loss
