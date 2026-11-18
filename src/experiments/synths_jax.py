"""
JAX / Equinox port of `DDSP808Synth` from `synths.py`.

The signal path, parameter order, parameter ranges, and output shape mirror the
PyTorch version (the `Snare808` voice followed by a per-item time shift), so a batch
of normalized parameters in [0, 1] produces the same audio in both frameworks up to
float32 rounding (given the same noise buffer).
"""

import logging
import os
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def exp_decay_envelope(
    decay_ms: Float[Array, "bs 1"], sr: int, n_samples: int
) -> Float[Array, "bs n_samples"]:
    """1 ms linear attack followed by an exponential decay (`ExpDecayEnvelope`)."""
    n_attack = int(0.001 * sr)
    n_decay = n_samples - n_attack
    assert n_decay > 0, "n_samples must be greater than the attack length"
    bs = decay_ms.shape[0]
    decay_samples = decay_ms * sr / 1000.0
    decay_rate = 1.0 - (6.91 / decay_samples)
    t = jnp.arange(n_decay, dtype=decay_rate.dtype)
    env = decay_rate**t
    attack = jnp.linspace(0.0, 1.0, n_attack, dtype=env.dtype)
    attack = jnp.broadcast_to(attack, (bs, n_attack))
    return jnp.concatenate([attack, env], axis=-1)


def sinusoidal_oscillator(
    freq_hz: Float[Array, "bs 1"],
    mod_env: Float[Array, "bs n_samples"],
    mod_amount: Float[Array, "bs 1"],
    sr: int,
) -> Float[Array, "bs n_samples"]:
    """Sine oscillator with an envelope-modulated frequency (`SinusoidalOscillator`)."""
    f0 = 2 * jnp.pi * freq_hz / sr
    freq_env = jnp.ones_like(mod_env) * f0
    freq_env = freq_env + mod_env * mod_amount * f0
    # Prepend a zero so the phase starts at 0, then integrate
    freq_env = jnp.concatenate([jnp.zeros_like(freq_env[:, :1]), freq_env], axis=-1)
    phase = jnp.cumsum(freq_env, axis=-1)[:, :-1]
    return jnp.sin(phase)


def db_gain(
    x: Float[Array, "bs n"], gain_db: Float[Array, "bs 1"]
) -> Float[Array, "bs n"]:
    """Applies a gain in decibels (`Gain`, and the input gain of `Tanh`)."""
    return jnp.power(10.0, gain_db / 20.0) * x


def highpass_coefficients(
    freq_hz: Float[Array, "bs 1"], q: Float[Array, "bs 1"], sr: int
) -> tuple[Float[Array, "bs 3"], Float[Array, "bs 2"]]:
    """Butterworth-style highpass biquad coefficients (`Biquad.highpass_coefficients`).

    Returns `b = (b0, b1, b2)` (numerator) and `a = (a1, a2)` (denominator, a0 = 1).
    `synths.py` calls these `a_coefs` and `b_coefs` the other way around.
    """
    k = jnp.tan(jnp.pi * freq_hz / sr)
    norm = 1 / (1 + k / q + k * k)
    b0 = 1 * norm
    b = jnp.concatenate([b0, -2 * b0, b0], axis=-1)
    a1 = 2 * (k * k - 1) * norm
    a2 = (1 - k / q + k * k) * norm
    a = jnp.concatenate([a1, a2], axis=-1)
    return b, a


def biquad_scan(
    x: Float[Array, "bs n"], b: Float[Array, "bs 3"], a: Float[Array, "bs 2"]
) -> Float[Array, "bs n"]:
    """Direct form I biquad with zero initial state, run sample by sample with
    `lax.scan` (same recurrence as `torchaudio.functional.lfilter`, without clamping).

    A parallel associative scan (e.g. `korvax.filter.lti.lfilter`) is not used
    because it diverges in float32 for low cutoffs with high Q (poles near |z| = 1),
    which this synth's parameter range includes.
    """
    # FIR part, all samples at once
    xp = jnp.pad(x, ((0, 0), (2, 0)))
    u = b[:, 2:3] * xp[:, :-2] + b[:, 1:2] * xp[:, 1:-1] + b[:, 0:1] * xp[:, 2:]
    a1, a2 = a[:, 0], a[:, 1]

    # IIR part, sequential over time
    def step(carry, u_t):
        y1, y2 = carry
        y_t = u_t - (a2 * y2 + a1 * y1)
        return (y_t, y1), y_t

    zeros = jnp.zeros_like(u[:, 0])
    _, y = jax.lax.scan(step, (zeros, zeros), u.T)
    return y.T


def time_shift(
    y: Float[Array, "bs n"], delta: Int[Array, "bs"], delta_min: int
) -> Float[Array, "bs n"]:
    """Delays item i by `delta[i] - delta_min` samples (zero padded on the left and
    truncated to the original length), like the loop in `DDSP808Synth.forward`.
    """
    n = y.shape[-1]
    n_pad = (delta - delta_min)[:, None]
    idx = jnp.arange(n)[None, :] - n_pad
    valid = (idx >= 0) & (idx < n)
    shifted = jnp.take_along_axis(y, jnp.clip(idx, 0, n - 1), axis=-1)
    return jnp.where(valid, shifted, 0.0)


class DDSP808Synth(eqx.Module):
    """808-style snare synthesizer (two swept sine oscillators plus enveloped,
    highpassed noise into a tanh waveshaper), ported from the PyTorch
    `eval_808.synths.DDSP808Synth`.

    Takes a batch of 14 normalized parameters in [0, 1] (order given by
    `param_names`) and returns audio of shape `(bs, 1, num_samples)`. Works with
    `jax.jit`, `jax.grad`, and `jax.vmap`. Unlike the PyTorch version, the [0, 1]
    range of `params` and the [delta_min, delta_max] range of `delta` are not
    checked at runtime.

    The noise buffer (`buffer_noise=True`) is stored in `noise` and is excluded from
    gradients. To reuse the exact buffer of a PyTorch synth:
    `synth = eqx.tree_at(lambda s: s.noise, synth, jnp.asarray(torch_noise))`.
    """

    param_names: ClassVar[tuple[str, ...]] = (
        "osc1_freq",
        "osc1_mod",
        "osc2_freq",
        "osc2_mod",
        "freq_decay",
        "osc1_decay",
        "osc2_decay",
        "noise_decay",
        "noise_freq",
        "noise_q",
        "osc1_gain",
        "osc2_gain",
        "noise_gain",
        "tanh_gain",
    )

    sr: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    buffer_noise: bool = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    J_cqt: int = eqx.field(static=True)
    Q_cqt: int = eqx.field(static=True)
    hop_len: int = eqx.field(static=True)
    delta_min: int = eqx.field(static=True)
    delta_max: int = eqx.field(static=True)
    param_ranges: tuple[tuple[float, float], ...] = eqx.field(static=True)
    noise: Float[Array, "buffer_size"] | None

    def __init__(
        self,
        sr: int,
        num_samples: int,
        buffer_noise: bool = False,
        buffer_size: int = 0,
        J_cqt: int = 9,
        Q_cqt: int = 12,
        hop_len: int = 256,
        delta_min: int = -2048,
        delta_max: int = 2048,
        *,
        key: Key[Array, ""] | None = None,
    ) -> None:
        """
        Args:
            sr: Sample rate in Hz.
            num_samples: Output length in samples.
            buffer_noise: If True, draws one noise buffer at init (from `key`) and
                reuses it for every call. If False, draws fresh noise on every call
                (from the `key` passed to `__call__`).
            buffer_size: Length of the noise buffer, at least `num_samples`.
            J_cqt, Q_cqt, hop_len: CQT settings, stored for the training code.
            delta_min, delta_max: Allowed range of the time shift `delta` (samples).
            key: PRNG key for the noise buffer. Required if `buffer_noise` is True.
        """
        self.sr = sr
        self.num_samples = num_samples
        self.buffer_noise = buffer_noise
        self.buffer_size = buffer_size
        self.J_cqt = J_cqt
        self.Q_cqt = Q_cqt
        self.hop_len = hop_len
        self.delta_min = delta_min
        self.delta_max = delta_max

        # (min, max) of each parameter, in the order of `param_names`
        osc_freq, osc_mod = (20.0, 2000.0), (-1.0, 2.0)
        decay_ms, gain_db = (10.0, 2000.0), (-60.0, 6.0)
        self.param_ranges = (
            osc_freq,
            osc_mod,
            osc_freq,
            osc_mod,
            decay_ms,
            decay_ms,
            decay_ms,
            decay_ms,
            (20.0, float(sr // 2)),  # noise highpass cutoff (Hz)
            (0.5, 10.0),  # noise highpass Q
            gain_db,
            gain_db,
            gain_db,
            (-24.0, 24.0),  # tanh input gain (dB)
        )

        if buffer_noise:
            if buffer_size < num_samples:
                raise ValueError("buffer_size must be at least num_samples")
            if key is None:
                raise ValueError("A PRNG key is required when buffer_noise is True")
            self.noise = jax.random.uniform(
                key, (buffer_size,), minval=-1.0, maxval=1.0
            )
        else:
            self.noise = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def __call__(
        self,
        params: Float[Array, "bs n_params"],
        delta: Int[Array, "bs"] | None = None,
        *,
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "bs 1 n_samples"]:
        """
        Args:
            params: Normalized synth parameters in [0, 1], shape `(bs, n_params)`.
            delta: Time shift per item in [delta_min, delta_max]. If None, uses 0
                (which still delays the sound by `-delta_min` samples, as in
                PyTorch).
            key: PRNG key for the noise. Required if `buffer_noise` is False.
        """
        if params.ndim != 2 or params.shape[-1] != self.n_params:
            raise ValueError(f"Expected params of shape (bs, {self.n_params})")
        bs = params.shape[0]
        sr, n = self.sr, self.num_samples

        # Map from [0, 1] to physical units
        lo, hi = zip(*self.param_ranges)
        lo = jnp.asarray(lo, dtype=params.dtype)
        hi = jnp.asarray(hi, dtype=params.dtype)
        p = lo + (hi - lo) * params
        p = dict(zip(self.param_names, jnp.split(p, self.n_params, axis=-1)))

        # Oscillators
        freq_env = exp_decay_envelope(p["freq_decay"], sr, n)
        y1 = sinusoidal_oscillator(p["osc1_freq"], freq_env, p["osc1_mod"], sr)
        y2 = sinusoidal_oscillator(p["osc2_freq"], freq_env, p["osc2_mod"], sr)
        env1 = exp_decay_envelope(p["osc1_decay"], sr, n)
        env2 = exp_decay_envelope(p["osc2_decay"], sr, n)
        y = db_gain(y1, p["osc1_gain"]) * env1
        y = y + db_gain(y2, p["osc2_gain"]) * env2

        # Noise
        if self.noise is not None:
            noise = jax.lax.stop_gradient(self.noise[:n]).astype(params.dtype)
            noise = jnp.broadcast_to(noise, (bs, n))
        else:
            if key is None:
                raise ValueError("A PRNG key is required when buffer_noise is False")
            noise = jax.random.uniform(
                key, (bs, n), dtype=params.dtype, minval=-1.0, maxval=1.0
            )
        noise = noise * exp_decay_envelope(p["noise_decay"], sr, n)
        b, a = highpass_coefficients(p["noise_freq"], p["noise_q"], sr)
        noise = biquad_scan(noise, b, a)
        noise = jnp.clip(noise, -1.0, 1.0)  # torchaudio's lfilter clamps by default
        noise = db_gain(noise, p["noise_gain"])

        # Mix and waveshape
        y = y + noise
        y = jnp.tanh(db_gain(y, p["tanh_gain"]))

        if delta is None:
            delta = jnp.zeros((bs,), dtype=jnp.int32)
        y = time_shift(y, delta, self.delta_min)
        return y[:, None, :]
