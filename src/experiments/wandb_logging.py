"""
Weights & Biases logging for the JAX trainer (`trainer_jax.py`).

Each training run and seed is a wandb run, named `<run name>__seed_<seed>` and grouped
by run name. Metrics are logged against the number of optimizer steps. After every
epoch, a few fixed validation items are logged as listenable audio (the targets once,
the predictions every epoch) and as a plot of the target and prediction (waveforms
and spectrograms), under `val_examples/<idx>/...`; the media panels of the wandb UI
have a step slider to go through the epochs.

The mode is `custom.wandb_mode` in the config: `online` (the default, which needs
`wandb login` once), `offline` (upload later with
`wandb sync out/wandb_logs/wandb/offline-run-*`), or `disabled`. The `WANDB_MODE`
environment variable takes precedence. The local files of an online run are deleted
once it has finished and uploaded everything, so they do not pile up; those of an
offline run, or of a run that failed, are kept.
"""

import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import scipy.signal
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.paths import WANDB_LOGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


@contextmanager
def wandb_run(
    config: dict[str, Any], name: str, seed: int, resume_id: str | None = None
) -> Iterator[wandb.Run]:
    """Starts the wandb run of `name` and `seed`, or resumes the run `resume_id`, and
    finishes it on exit (marking it as failed if there is an exception). Then, if the
    run is online and succeeded, deletes its local directory: finishing the run
    uploads everything first."""
    custom = config.get("custom") or {}
    os.makedirs(WANDB_LOGS_DIR, exist_ok=True)
    run = wandb.init(
        project=custom.get("project_name", "venturi"),
        name=f"{name}__seed_{seed}",
        group=name,
        config=config,
        dir=WANDB_LOGS_DIR,
        mode=os.environ.get("WANDB_MODE") or custom.get("wandb_mode", "online"),
        id=resume_id,
        resume="allow" if resume_id else None,
    )
    is_online = not (run.offline or run.disabled)
    local_dir = os.path.dirname(run.dir)  # e.g. wandb/run-<date>_<time>-<id>
    with run:
        yield run
    if is_online:
        delete_local_dir(local_dir, run.id)


def delete_local_dir(local_dir: str, run_id: str) -> None:
    """Deletes the local directory of a finished wandb run, after checking that it is
    one (`<WANDB_LOGS_DIR>/wandb/run-<date>_<time>-<run_id>`)."""
    local_dir = os.path.abspath(local_dir)
    parent = os.path.join(os.path.abspath(WANDB_LOGS_DIR), "wandb")
    name = os.path.basename(local_dir)
    if os.path.dirname(local_dir) != parent or not (
        name.startswith("run-") and name.endswith(f"-{run_id}")
    ):
        log.warning(f"Not deleting {local_dir}: not the directory of run {run_id}")
        return
    shutil.rmtree(local_dir, ignore_errors=True)
    log.info(f"Deleted the local wandb files of run {run_id}: {local_dir}")


def spectrogram_db(
    x: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequencies (without 0 Hz, for a log axis), times, and STFT magnitudes in dB
    (shape `(n_freqs, n_frames)`) of a 1D signal."""
    freqs, times, X = scipy.signal.stft(
        x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    return freqs[1:], times, 20 * np.log10(np.abs(X[1:]) + 1e-8)


def plot_example(x: np.ndarray, x_hat: np.ndarray, sr: int, title: str) -> Figure:
    """Waveforms (top) and spectrograms (bottom, log frequency axis, same dB range)
    of a 1D target `x` (left) and prediction `x_hat` (right)."""
    fig = Figure(figsize=(10, 6), layout="constrained")
    axs = fig.subplots(2, 2, sharex=True, sharey="row")
    specs = [spectrogram_db(a, sr) for a in (x, x_hat)]
    db_max = max(db.max() for _, _, db in specs)
    times = np.arange(x.shape[-1]) / sr
    for col, (audio, label) in enumerate([(x, "target"), (x_hat, "prediction")]):
        axs[0, col].plot(times, audio, linewidth=0.5)
        axs[0, col].set(title=label, ylim=(-1.0, 1.0))
        freqs, spec_times, db = specs[col]
        mesh = axs[1, col].pcolormesh(
            spec_times, freqs, db, vmin=db_max - 80.0, vmax=db_max, shading="auto"
        )
        axs[1, col].set(yscale="log", ylim=(20.0, sr / 2), xlabel="time (s)")
    axs[0, 0].set_ylabel("amplitude")
    axs[1, 0].set_ylabel("frequency (Hz)")
    fig.colorbar(mesh, ax=axs[1, :], label="dB")
    fig.suptitle(title)
    return fig


def render(fig: Figure) -> np.ndarray:
    """RGB pixels of a figure, without pyplot (which `wandb.Image(fig)` needs)."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]


def val_example_media(
    outputs: list[dict[str, np.ndarray]],
    n: int,
    sr: int,
    step: int,
    log_targets: bool,
) -> dict[str, Any]:
    """wandb media of the first `n` validation items in `outputs` (from `evaluate`):
    `val_examples/<idx>/prediction` (audio), `val_examples/<idx>/plot`, and, if
    `log_targets`, `val_examples/<idx>/target` (audio). The validation set is not
    shuffled, so `<idx>` is the same item at every epoch."""
    if n <= 0 or not outputs:
        return {}
    x = np.concatenate([o["x"] for o in outputs])[:n, 0]
    x_hat = np.concatenate([o["x_hat"] for o in outputs])[:n, 0]
    media = {}
    for idx in range(x.shape[0]):
        key = f"val_examples/{idx}"
        if log_targets:
            media[f"{key}/target"] = wandb.Audio(x[idx], sample_rate=sr)
        media[f"{key}/prediction"] = wandb.Audio(
            x_hat[idx], sample_rate=sr, caption=f"step {step}"
        )
        title = f"Validation item {idx}, step {step}"
        fig = plot_example(x[idx], x_hat[idx], sr, title)
        media[f"{key}/plot"] = wandb.Image(render(fig))
    return media
