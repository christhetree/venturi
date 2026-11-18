"""
NumPy version of `WavDataModule` from `data.py` for the JAX training script, without
PyTorch, torchaudio, or Lightning.

The train / val / test split is identical to the PyTorch version for the same
`shuffle_seed`, and the audio is normalized like `torchaudio.load`. The batch order
and the time shifts (`delta`) are drawn from `seed` (the run seed) with NumPy, so they
differ from a PyTorch run with the same seed.
"""

import logging
import os
from collections.abc import Iterator

import numpy as np
import soundfile

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Relative paths are relative to the experiments directory
ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))

Batch = tuple[np.ndarray, list[str], np.ndarray]


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Loads an audio file as float32 of shape (n_ch, n_samples), scaled to [-1, 1]
    like `torchaudio.load`."""
    audio, sr = soundfile.read(path, dtype="float32", always_2d=True)
    return audio.T, sr


def drum_type_from_name(f_name: str) -> str:
    for prefix, drum_type in (
        ("BD ", "BD"),
        ("SD ", "SD"),
        ("OH ", "HH"),
        ("CH ", "HH"),
        ("Tom ", "Tom"),
    ):
        if f_name.startswith(prefix):
            return drum_type
    raise ValueError(f"Unknown drum type in file name: {f_name}")


class WavDataset:
    def __init__(
        self,
        samples: np.ndarray,
        drum_types: list[str],
        n_delta_per_item: int,
        delta_min: int,
        delta_max: int,
        rng: np.random.Generator,
    ):
        assert samples.shape[0] == len(drum_types)
        assert set(drum_types) == {"BD", "SD", "HH", "Tom"}
        self.samples = samples
        self.drum_types = drum_types
        self.n_delta_per_item = n_delta_per_item
        n = samples.shape[0] * n_delta_per_item
        self.deltas = rng.integers(delta_min, delta_max + 1, size=n, dtype=np.int32)

    def __len__(self) -> int:
        return self.samples.shape[0] * self.n_delta_per_item

    def get_batch(self, idxs: np.ndarray) -> Batch:
        sample_idxs = idxs // self.n_delta_per_item
        x = self.samples[sample_idxs]
        drum_types = [self.drum_types[i] for i in sample_idxs]
        return x, drum_types, self.deltas[idxs]


class WavDataModule:
    """Same arguments as the PyTorch `WavDataModule`, except that `num_workers` is
    replaced by `seed`. Batches are `(x, drum_types, delta)` with `x` of shape
    `(batch_size, 1, n_samples)`, and incomplete final batches are dropped.
    """

    def __init__(
        self,
        batch_size: int,
        root_dir: str,
        sr: int,
        n_samples: int,
        n_train: int,
        n_val: int,
        n_test: int,
        n_delta_per_item: int = 1,
        delta_min: int = 0,
        delta_max: int = 0,
        shuffle_seed: int = 42,
        seed: int = 0,
    ):
        root_dir = os.path.join(ROOT_DIR, root_dir)
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.sr = sr
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        log.info(
            f"shuffle_seed: {shuffle_seed}, seed: {seed}, delta_min: {delta_min}, "
            f"delta_max: {delta_max}, n_delta_per_item: {n_delta_per_item}"
        )

        sample_paths = [p for p in os.listdir(root_dir) if p.endswith(".wav")]
        log.info(f"Found {len(sample_paths)} .wav files in {root_dir}")
        assert len(sample_paths) >= (
            n_train + n_val + n_test
        ), f"Not enough .wav files in {root_dir}"
        # Same split as the PyTorch version
        sample_paths = sorted(sample_paths)
        np.random.seed(shuffle_seed)
        np.random.shuffle(sample_paths)

        samples = []
        drum_types = []
        for p in sample_paths:
            drum_types.append(drum_type_from_name(os.path.basename(p)))
            sample, sample_sr = load_wav(os.path.join(root_dir, p))
            assert sample_sr == sr
            if sample.shape[0] > 1:
                sample = np.mean(sample, axis=0, keepdims=True)
                log.warning(f"File {p} has more than 1 channel")
            if sample.shape[1] >= n_samples:
                sample = sample[:, :n_samples]
            else:
                sample = np.pad(sample, ((0, 0), (0, n_samples - sample.shape[1])))
            samples.append(sample)
        samples = np.stack(samples, axis=0)

        splits = {
            "train": slice(0, n_train),
            "val": slice(n_train, n_train + n_val),
            "test": slice(n_train + n_val, n_train + n_val + n_test),
        }
        self.datasets = {}
        for stage, s in splits.items():
            unique, counts = np.unique(drum_types[s], return_counts=True)
            log.info(f"{stage} drum type counts: {dict(zip(unique, counts))}")
            self.datasets[stage] = WavDataset(
                samples[s],
                drum_types[s],
                n_delta_per_item,
                delta_min,
                delta_max,
                self.rng,
            )

    def n_batches(self, stage: str) -> int:
        return len(self.datasets[stage]) // self.batch_size

    def batches(self, stage: str) -> Iterator[Batch]:
        """Iterates over the batches of a stage ("train" is reshuffled every call)."""
        dataset = self.datasets[stage]
        idxs = np.arange(len(dataset))
        if stage == "train":
            idxs = self.rng.permutation(idxs)
        for b in range(self.n_batches(stage)):
            yield dataset.get_batch(
                idxs[b * self.batch_size : (b + 1) * self.batch_size]
            )
