"""
JAX / Equinox port of nnAudio's `CQT` (`CQT1992v2`, magnitude output, librosa
normalization) and of `SCRAPLLightingModule.calc_cqt`, which together compute the
CNN input in `eval_808/lightning.py`.

Usage, with the settings `eval_808/lightning.py` derives from the synth:

    cqt = CQT(
        sr=synth.sr,
        hop_length=synth.hop_len,
        fmin=(0.4 * synth.sr) / (2**synth.J_cqt),
        n_bins=synth.J_cqt * synth.Q_cqt,
        bins_per_octave=synth.Q_cqt,
    )
    U = jax.vmap(lambda x_: calc_cqt(x_, cqt, cqt_eps=1e-3))(x[:, 0, :])
"""

import logging
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from jaxtyping import Array, Float

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def create_cqt_kernels(
    Q: float,
    sr: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    norm: int = 1,
    window: str = "hann",
) -> tuple[np.ndarray, int, np.ndarray]:
    """Time-domain CQT kernels, built exactly like `nnAudio.utils.create_cqt_kernels`.

    Returns:
        The complex64 kernels of shape `(n_bins, kernel_width)`, the kernel width
        (a power of 2), and the float32 kernel lengths of shape `(n_bins,)`.
    """
    freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    if np.max(freqs) > sr / 2:
        raise ValueError(
            f"The top bin {np.max(freqs)} Hz exceeds the Nyquist frequency"
        )
    lengths = np.ceil(Q * sr / freqs)
    kernel_width = int(2 ** (np.ceil(np.log2(int(max(lengths))))))

    kernels = np.zeros((n_bins, kernel_width), dtype=np.complex64)
    for k in range(n_bins):
        freq = freqs[k]
        l = lengths[k]
        # Center the kernel (one more zero on the right for odd lengths)
        start = int(np.ceil(kernel_width / 2.0 - l / 2.0))
        if l % 2 == 1:
            start -= 1
        win = scipy.signal.get_window(window, int(l), fftbins=True)
        sig = win * np.exp(np.r_[-l // 2 : l // 2] * 1j * 2 * np.pi * freq / sr) / l
        if norm:
            sig = sig / np.linalg.norm(sig, norm)
        kernels[k, start : start + int(l)] = sig

    return kernels, kernel_width, lengths.astype(np.float32)


class CQT(eqx.Module):
    """Constant-Q transform magnitude of a single 1D signal, computed as a strided
    convolution with time-domain kernels (port of nnAudio's `CQT1992v2`).

    The kernels are stored as arrays and excluded from gradients. The output is
    differentiable with respect to the input. Where a CQT coefficient is exactly zero
    (e.g. frames over zero padding), the magnitude has a zero gradient instead of the
    NaN that `sqrt` gives (nnAudio and `korvax.cqt` give NaN there); the forward
    output is unchanged.
    """

    sr: int = eqx.field(static=True)
    hop_length: int = eqx.field(static=True)
    fmin: float = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    bins_per_octave: int = eqx.field(static=True)
    center: bool = eqx.field(static=True)
    pad_mode: str = eqx.field(static=True)
    kernel_width: int = eqx.field(static=True)
    kernels: Float[Array, "n_kernels 1 kernel_width"]  # n_kernels = 2 * n_bins
    sqrt_lengths: Float[Array, "n_bins 1"]

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 32.70,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        filter_scale: float = 1,
        norm: int = 1,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        """Arguments and defaults follow `nnAudio.features.CQT`."""
        if pad_mode not in ("reflect", "constant"):
            raise ValueError(f"Unknown pad_mode: {pad_mode}")
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.center = center
        self.pad_mode = pad_mode

        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        kernels, self.kernel_width, lengths = create_cqt_kernels(
            Q, sr, fmin, n_bins, bins_per_octave, norm, window
        )
        # Real kernels stacked on top of imaginary ones, as conv weights (O, I, W)
        kernels = np.concatenate([kernels.real, kernels.imag], axis=0)
        self.kernels = jnp.asarray(kernels[:, None, :], dtype=jnp.float32)
        self.sqrt_lengths = jnp.sqrt(jnp.asarray(lengths))[:, None]

    def n_frames(self, n_samples: int) -> int:
        """Number of output frames for an input of `n_samples` samples."""
        if self.center:
            n_samples += 2 * (self.kernel_width // 2)
        return (n_samples - self.kernel_width) // self.hop_length + 1

    def __call__(self, x: Float[Array, "n_samples"]) -> Float[Array, "n_bins n_frames"]:
        kernels = jax.lax.stop_gradient(self.kernels)
        sqrt_lengths = jax.lax.stop_gradient(self.sqrt_lengths)
        if self.center:
            x = jnp.pad(x, self.kernel_width // 2, mode=self.pad_mode)
        out = jax.lax.conv_general_dilated(
            x[None, None, :].astype(kernels.dtype),
            kernels,
            window_strides=(self.hop_length,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )[0]
        # nnAudio negates the imaginary part, which does not change the magnitude
        cqt_real = out[: self.n_bins] * sqrt_lengths
        cqt_imag = out[self.n_bins :] * sqrt_lengths
        power = cqt_real**2 + cqt_imag**2
        # sqrt with a zero (not NaN) gradient where the power is exactly zero
        nonzero = power > 0
        return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, power, 1.0)), 0.0)


def calc_cqt(
    x: Float[Array, "n_samples"], cqt: CQT, cqt_eps: float = 1e-3
) -> Float[Array, "n_bins n_frames"]:
    """Log-compressed CQT used as the CNN input (`SCRAPLLightingModule.calc_cqt`)."""
    return jnp.log1p(cqt(x) / cqt_eps)
