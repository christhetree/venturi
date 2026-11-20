"""
JAX equivalents of the LogMSSLoss class in losses.py and of
auraloss.freq.MultiResolutionSTFTLoss, using korvax for the STFT. Kept separate from
losses.py so that they can be imported without PyTorch.

Also `DDSP808Loss`, the training objective of the 808 experiments, which uses one of
these losses on the synthesized audio.
"""

import json
import logging
import math
import os
from collections.abc import Callable
from typing import Any, List, NamedTuple, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import korvax
import numpy as np
import optax
from jaxtyping import Array, Complex, Float, Int, Key
from korvax.loss import elementwise_loss

from src.experiments.cqt_jax import CQT, calc_cqt
from src.experiments.synths_jax import DDSP808Synth
from src.paths import ROOT_DIR
from src.scrapl_jax import safe_lp_norm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _flat_lp_norm(x: Float[Array, "..."], p: float, n_dims: int) -> Float[Array, "..."]:
    """
    Lp norm over the last `n_dims` axes treated as one flattened vector (like
    torch.linalg.vector_norm with a tuple of dims). jnp.linalg.norm with a tuple of
    axes computes a matrix norm instead, so it cannot be used here.
    """
    x = x.reshape(x.shape[:-n_dims] + (-1,))
    return safe_lp_norm(x, p=p, axis=-1)


def _stft(
    x: Float[Array, "... n_samples"],
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
) -> Complex[Array, "... n_bins n_frames"]:
    """
    STFT computed with korvax and centered like torch.stft(center=True), i.e. reflect
    padded by n_fft // 2 on both sides. korvax's own centering pads with zeros instead,
    and with one more sample on the right when n_fft is odd. Windows are periodic.
    """
    pad = n_fft // 2
    x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    return korvax.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
    )


class MultiResolutionSTFTLoss(eqx.Module):
    """
    auraloss.freq.MultiResolutionSTFTLoss in JAX, using korvax. Only the log and
    linear magnitude terms are implemented (no spectral convergence, phase,
    perceptual weighting, or frequency scale), which is all the mss_meso_log config
    uses. Like auraloss, the magnitudes are sqrt(max(|STFT|^2, eps)), and the loss is
    averaged over the resolutions. korvax.loss.mrstft_loss is not used since it
    computes magnitudes and pads the signal differently. An Equinox module with only
    static (configuration) fields, so it can be a field of other modules.
    """

    fft_sizes: tuple[int, ...] = eqx.field(static=True)
    hop_sizes: tuple[int, ...] = eqx.field(static=True)
    win_lengths: tuple[int, ...] = eqx.field(static=True)
    w_log_mag: float = eqx.field(static=True)
    w_lin_mag: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    mag_distance: str = eqx.field(static=True)

    def __init__(
        self,
        fft_sizes: Sequence[int],
        hop_sizes: Sequence[int],
        win_lengths: Sequence[int],
        window: str = "hann_window",
        w_sc: float = 0.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        eps: float = 1e-8,
        mag_distance: str = "L1",
    ):
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        assert w_sc == 0.0, "The spectral convergence term is not implemented"
        assert w_phs == 0.0, "The phase term is not implemented"
        assert window == "hann_window", "Only the hann_window window is implemented"
        assert mag_distance in ["L1", "L2"]
        # Tuples, since static fields must be hashable
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes)
        self.win_lengths = tuple(win_lengths)
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.eps = eps
        self.mag_distance = mag_distance

    def __call__(
        self,
        x: Float[Array, "bs n_ch n_samples"],
        x_target: Float[Array, "bs n_ch n_samples"],
    ) -> Float[Array, ""]:
        assert x.ndim == x_target.ndim == 3
        loss = 0.0
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            mags = []
            for audio in [x, x_target]:
                X = _stft(audio, fft_size, hop_size, win_length, "hann")
                power = jnp.real(X) ** 2 + jnp.imag(X) ** 2
                mags.append(jnp.sqrt(jnp.maximum(power, self.eps)))
            x_mag, x_target_mag = mags
            if self.w_log_mag:
                log_mag_loss = elementwise_loss(
                    jnp.log(x_mag), jnp.log(x_target_mag), metric=self.mag_distance
                )
                loss += self.w_log_mag * log_mag_loss
            if self.w_lin_mag:
                lin_mag_loss = elementwise_loss(
                    x_mag, x_target_mag, metric=self.mag_distance
                )
                loss += self.w_lin_mag * lin_mag_loss
        loss /= len(self.fft_sizes)
        return loss


class LogMSSLoss:
    """
    LogMSSLoss of losses.py in JAX, using korvax for the STFTs. korvax's
    smooth_mrstft_loss is not used since it computes the mean squared error
    averaged over resolutions (instead of the Lp norm summed over resolutions), and
    its first hop size is 32 instead of 33.
    """

    # The scipy.signal.get_window names of the window types of losses.py
    WINDOWS = {"rect": "boxcar", "hann": "hann", "flat_top": "flattop"}

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = "flat_top",
        log_mag_eps: float = 1.0,
        gamma: float = 1.0,
        p: int = 2,
    ):
        assert window in self.WINDOWS, f"Unknown window type: {window}"
        if win_lengths is None:
            win_lengths = [67, 127, 257, 509, 1021, 2053]
            log.info(f"win_lengths = {win_lengths}")
        if fft_sizes is None:
            fft_sizes = win_lengths
            log.info(f"fft_sizes = {fft_sizes}")
        if hop_sizes is None:
            hop_sizes = [w // 2 for w in win_lengths]
            log.info(f"hop_sizes = {hop_sizes}")
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = self.WINDOWS[window]
        self.log_mag_eps = log_mag_eps
        self.gamma = gamma
        self.p = p

    def __call__(
        self,
        x: Float[Array, "bs n_ch n_samples"],
        x_target: Float[Array, "bs n_ch n_samples"],
    ) -> Float[Array, ""]:
        assert x.ndim == x_target.ndim == 3
        assert x.shape[1] == x_target.shape[1] == 1
        x = x[:, 0, :]
        x_target = x_target[:, 0, :]
        dists = []
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            log_mags = []
            for audio in [x, x_target]:
                S = jnp.abs(_stft(audio, fft_size, hop_size, win_length, self.window))
                if self.log_mag_eps == 1.0:
                    log_mags.append(jnp.log1p(self.gamma * S))
                else:
                    log_mags.append(jnp.log(self.gamma * S + self.log_mag_eps))
            log_Sx, log_Sx_target = log_mags
            dist = _flat_lp_norm(log_Sx_target - log_Sx, p=self.p, n_dims=2)
            dists.append(dist)
        dist = jnp.stack(dists, axis=1).sum(axis=1)
        dist = dist.mean()  # Aggregate the batch dimension
        return dist


@jax.custom_jvp
def scale_grad(x: Float[Array, "... n"], w: Float[Array, "n"]) -> Float[Array, "... n"]:
    """
    Returns `x` unchanged, but multiplies its derivatives by `w` (over its last axis),
    like the PyTorch gradient hook `x.register_hook(lambda g: g * w)`. `w` gets no
    derivatives.

    Defined by its JVP (whose transpose scales the gradients) rather than its VJP,
    so that it also works with forward-mode autodiff, e.g. the forward-over-reverse
    curvature-vector products of `src.scrapl_jax.warmup`. Second derivatives are
    those of the scaled gradient (with the forward pass unchanged), so `w` appears in
    them once.
    """
    return x


@scale_grad.defjvp
def _scale_grad_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    x, w = primals
    x_dot, _ = tangents
    return x, x_dot * w.astype(x_dot.dtype)


class GradRMSState(NamedTuple):
    """The state of `normalize_grad_rms`, as in `optax.ScaleByAdamState`: the number
    of updates, and `nu`, the running mean of the squared gradient of each synth
    parameter."""

    count: Int[Array, ""]
    nu: Float[Array, "n_params"]


def normalize_grad_rms(
    g: Float[Array, "bs n_params"], state: GradRMSState, b2: float, eps: float
) -> tuple[Float[Array, "bs n_params"], GradRMSState]:
    """
    Normalizes the gradients `g` of the synth parameters of a batch like
    `optax.scale_by_adam(b1=0.0, b2=b2, eps=eps)`, i.e. Adam without momentum:
    divides them by the square root of the bias-corrected running mean of their
    squares, plus `eps`. The squared gradient of each parameter is its mean over the
    batch, so this is identical to `optax.scale_by_adam` for a batch of one.

    Adam's momentum (`b1`) does not apply: the gradients of each batch are those of
    its own items, and can only go back through the model for those items.
    """
    count = optax.safe_increment(state.count)
    # The same operations as `optax.scale_by_adam`
    nu = (1 - b2) * jnp.mean(g**2, axis=0) + b2 * state.nu
    nu_hat = nu / (1 - b2**count).astype(nu.dtype)
    g = g / (jnp.sqrt(nu_hat) + eps)
    return g, GradRMSState(count=count, nu=nu)


def load_theta_curvatures(path: str, param_names: Sequence[str]) -> dict[str, Any]:
    """
    Loads the curvatures saved by `scripts/theta_curvature.py`, checking that they
    are finite and positive for exactly the synth parameters `param_names`.

    Args:
        path: The JSON file, absolute or relative to the repository root.
        param_names: The names of the synth parameters.

    Returns:
        The saved results, whose `params` maps each parameter name to its
        `curvature` and `relative_residual`.
    """
    with open(os.path.join(ROOT_DIR, path)) as f:
        results = json.load(f)
    params = results["params"]
    if set(params) != set(param_names):
        raise ValueError(
            f"{path} has the curvatures of {sorted(params)}, but the synth "
            f"parameters are {sorted(param_names)}"
        )
    curvatures = np.array([params[name]["curvature"] for name in param_names])
    if not (np.isfinite(curvatures).all() and (curvatures > 0).all()):
        raise ValueError(f"The curvatures in {path} must be finite and positive")
    return results


class DDSP808Loss(eqx.Module):
    """
    The training objective of `DDSP808LightingModule` (`lightning.py`) in JAX: the CQT
    of the target audio goes into the model, which predicts the synth parameters, and
    the loss compares the synthesized audio with the target.

    Holds only the fixed parts (synth, CQT, audio loss); the trainable model is passed
    in on every call, so training state is just `(model, opt_state)` (plus the state
    of `init_train_state` with `use_grad_rms`):

        loss = DDSP808Loss(synth, MultiResolutionSTFTLoss(...))
        (value, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, x, delta, key=key
        )

    With `use_curvature`, the gradient of the loss w.r.t. each predicted synth
    parameter θ_i is multiplied by `(c_i / geometric mean of c) ** -curvature_alpha`
    before it reaches the model, where `c_i` is its curvature from
    `scripts/theta_curvature.py`. The loss values are unchanged. Since each parameter
    has its own head in `Spectral2DCNN` and Adam normalizes the scale of each
    weight's updates, this mostly rebalances the gradients of the shared layers,
    which the parameters with the largest curvatures dominate otherwise.

    With `use_grad_rms`, the gradient of the loss w.r.t. each predicted synth
    parameter is instead normalized like Adam (without momentum) before it goes back
    through the model: divided by the square root of the bias-corrected running mean
    of its square (see `normalize_grad_rms`). The running mean is a state kept between
    training steps, so training uses `value_and_grad` with the state of
    `init_train_state` instead of differentiating `__call__`.
    """

    synth: DDSP808Synth
    cqt: CQT
    loss_func: Callable[..., Float[Array, ""]]
    cqt_eps: float = eqx.field(static=True)
    use_curvature: bool = eqx.field(static=True)
    curvature_path: str | None = eqx.field(static=True)
    curvature_alpha: float = eqx.field(static=True)
    # The multipliers of dL/dθ, None without `use_curvature`
    theta_grad_weights: Float[Array, "n_params"] | None
    use_grad_rms: bool = eqx.field(static=True)
    grad_rms_b2: float = eqx.field(static=True)
    grad_rms_eps: float = eqx.field(static=True)

    def __init__(
        self,
        synth: DDSP808Synth,
        loss_func: Callable[..., Float[Array, ""]],
        feature_type: str = "cqt",
        cqt_eps: float = 1e-3,
        use_curvature: bool = False,
        curvature_path: str | None = None,
        curvature_alpha: float = 0.25,
        use_grad_rms: bool = False,
        grad_rms_b2: float = 0.999,
        grad_rms_eps: float = 1e-8,
    ) -> None:
        """
        Args:
            synth: The `DDSP808Synth`.
            loss_func: Maps `(x_hat, x)`, both `(bs, 1, n_samples)`, to a scalar
                (e.g. `MultiResolutionSTFTLoss` or `LogMSSLoss`).
            feature_type: Only "cqt" is supported, as in `lightning.py`.
            cqt_eps: Scale of the log compression of the CQT.
            use_curvature: Scales the gradient of each synth parameter according to
                its curvature (see the class docstring).
            curvature_path: The JSON file of the curvatures, saved by
                `scripts/theta_curvature.py`, absolute or relative to the repository
                root. Only read with `use_curvature`.
            curvature_alpha: The exponent of the scaling. 0 leaves the gradients
                unchanged, and 1 divides them by the (relative) curvatures.
            use_grad_rms: Normalizes the gradient of each synth parameter like Adam
                (see the class docstring). Not with `use_curvature`, whose constant
                scaling it would cancel.
            grad_rms_b2: The decay rate of the running mean of the squared gradients,
                as Adam's `b2`.
            grad_rms_eps: Added to the root mean square, as Adam's `eps`.
        """
        if feature_type != "cqt":
            raise NotImplementedError(f"Unsupported feature_type: {feature_type}")
        if use_curvature and use_grad_rms:
            raise ValueError("use_curvature and use_grad_rms cannot both be True")
        if not 0 <= grad_rms_b2 < 1:
            raise ValueError("grad_rms_b2 must be in [0, 1)")
        if not (math.isfinite(grad_rms_eps) and grad_rms_eps >= 0):
            raise ValueError("grad_rms_eps must be finite and nonnegative")
        self.synth = synth
        self.loss_func = loss_func
        self.cqt_eps = cqt_eps
        self.use_curvature = use_curvature
        self.curvature_path = curvature_path
        self.curvature_alpha = curvature_alpha
        self.use_grad_rms = use_grad_rms
        self.grad_rms_b2 = grad_rms_b2
        self.grad_rms_eps = grad_rms_eps
        self.theta_grad_weights = None
        if use_curvature:
            if curvature_path is None:
                raise ValueError("use_curvature needs a curvature_path")
            if not (math.isfinite(curvature_alpha) and curvature_alpha >= 0):
                raise ValueError("curvature_alpha must be finite and nonnegative")
            params = load_theta_curvatures(curvature_path, synth.param_names)["params"]
            log_c = np.log([params[name]["curvature"] for name in synth.param_names])
            weights = np.exp(-curvature_alpha * (log_c - log_c.mean()))
            self.theta_grad_weights = jnp.asarray(weights, dtype=jnp.float32)
        # Same CQT settings as `DDSP808LightingModule`
        self.cqt = CQT(
            sr=synth.sr,
            hop_length=synth.hop_len,
            fmin=(0.4 * synth.sr) / (2**synth.J_cqt),
            n_bins=synth.J_cqt * synth.Q_cqt,
            bins_per_octave=synth.Q_cqt,
        )

    def prepare_target(
        self, x: Float[Array, "bs 1 n_samples"]
    ) -> Float[Array, "bs 1 n_samples"]:
        """Delays the target by the padding the synth adds when `delta_min < 0`."""
        n_padding = -self.synth.delta_min
        if n_padding > 0:
            x = jnp.pad(x, ((0, 0), (0, 0), (n_padding, 0)))
            x = x[:, :, : self.synth.num_samples]
        return x

    def calc_U(
        self, x: Float[Array, "bs 1 n_samples"]
    ) -> Float[Array, "bs n_bins n_frames"]:
        return jax.vmap(lambda x_: calc_cqt(x_, self.cqt, self.cqt_eps))(x[:, 0, :])

    def describe_curvature_scaling(self) -> str:
        """Describes the gradient scaling of `use_curvature`, for the training log."""
        assert self.use_curvature, "use_curvature is False"
        results = load_theta_curvatures(self.curvature_path, self.synth.param_names)
        if results.get("ckpt") is None:
            model_state = "the model's initialization"
        else:
            model_state = f"{results['ckpt']} (epoch {results.get('ckpt_epoch')})"
        lines = [
            f"Scaling the gradient of each synth parameter by (curvature / geometric "
            f"mean) ** -{self.curvature_alpha}, with the curvatures of "
            f"{self.curvature_path} ({results.get('loss_func')}, "
            f"{results.get('n_batches')} batch(es) of {results.get('batch_size')}, "
            f"seed {results.get('seed')}, {model_state}):"
        ]
        tol = results.get("tol") or 0.0
        weights = np.asarray(self.theta_grad_weights)
        for name, weight in zip(self.synth.param_names, weights):
            p = results["params"][name]
            not_converged = 0 < tol <= p["relative_residual"]
            lines.append(
                f"    {name:<12} curvature {p['curvature']:.3e}, weight {weight:.3f}"
                f"{' (not converged)' * not_converged}"
            )
        return "\n".join(lines)

    def describe_grad_rms(self) -> str:
        """Describes the normalization of `use_grad_rms`, for the training log."""
        assert self.use_grad_rms, "use_grad_rms is False"
        return (
            f"Normalizing the gradient of each synth parameter like Adam without "
            f"momentum (b2 = {self.grad_rms_b2}, eps = {self.grad_rms_eps}), with "
            f"the mean of its square over each batch"
        )

    def init_train_state(self) -> GradRMSState | None:
        """The state kept between training steps: the running mean of the squared
        gradients with `use_grad_rms`, else None."""
        if not self.use_grad_rms:
            return None
        return GradRMSState(
            count=jnp.zeros([], jnp.int32),
            nu=jnp.zeros([self.synth.n_params], jnp.float32),
        )

    def predict(
        self,
        model: eqx.Module,
        x: Float[Array, "bs 1 n_samples"],
        *,
        key: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, "bs n_params"], Float[Array, "bs 1 n_samples"]]:
        """Predicts the synth parameters of a batch of target audio, as in
        `__call__`, and returns them with the delayed target."""
        x = self.prepare_target(x)
        U = jax.lax.stop_gradient(self.calc_U(x))
        keys = None if key is None else jax.random.split(key, x.shape[0])
        theta_hat = jax.vmap(model)(U, key=keys)
        return theta_hat, x

    def value_and_grad(
        self,
        model: eqx.Module,
        x: Float[Array, "bs 1 n_samples"],
        delta: Int[Array, "bs"] | None,
        state: GradRMSState,
        *,
        key: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, ""], eqx.Module, GradRMSState]:
        """With `use_grad_rms`: the loss of `model` on a batch, as `__call__`, and
        its gradient w.r.t. the floating arrays of `model`, as `eqx.filter_grad`, but
        with the gradient of each synth parameter normalized by `normalize_grad_rms`
        on its way back through the model.

        Returns:
            The loss, the gradient, and the updated state.
        """
        assert self.use_grad_rms, "use_grad_rms is False"
        # Backpropagates in two parts, normalizing the gradients in between: from
        # the loss to the synth parameters, then from them through the model
        theta_hat, model_vjp, x = eqx.filter_vjp(
            lambda m: self.predict(m, x, key=key), model, has_aux=True
        )
        value, g_theta = jax.value_and_grad(
            lambda theta: self.loss_func(self.synth(theta, delta), x)
        )(theta_hat)
        g_theta, state = normalize_grad_rms(
            g_theta, state, self.grad_rms_b2, self.grad_rms_eps
        )
        (grads,) = model_vjp(g_theta)
        return value, grads, state

    def __call__(
        self,
        model: eqx.Module,
        x: Float[Array, "bs 1 n_samples"],
        delta: Int[Array, "bs"] | None = None,
        *,
        key: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        """Computes the loss of `model` on a batch of target audio.

        Args:
            model: Maps a log-CQT of shape `(n_bins, n_frames)` to `(n_params,)` synth
                parameters in [0, 1] (e.g. `Spectral2DCNN`); vmapped over the batch.
            x: Target audio, as loaded (before the delay of `prepare_target`).
            delta: Time shifts for the synth. Use None for validation and testing.
            key: PRNG key for dropout. Not needed with `eqx.nn.inference_mode(model)`.

        Returns:
            The loss and a dict with the delayed target `x`, the synthesized `x_hat`,
            and the predicted parameters `theta_hat`.
        """
        theta_hat, x = self.predict(model, x, key=key)
        if self.theta_grad_weights is not None:
            theta_hat = scale_grad(theta_hat, self.theta_grad_weights)
        x_hat = self.synth(theta_hat, delta)
        loss = self.loss_func(x_hat, x)
        return loss, {"x": x, "x_hat": x_hat, "theta_hat": theta_hat}
