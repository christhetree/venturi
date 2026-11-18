"""
The training objective of `DDSP808LightingModule` (`lightning.py`) in JAX: the CQT of
the target audio goes into the model, which predicts the synth parameters, and the
loss compares the synthesized audio with the target.

`DDSP808Loss` holds only the fixed parts (synth, CQT, audio loss); the trainable
model is passed in on every call, so training state is just `(model, opt_state)`:

    loss = DDSP808Loss(synth, MultiResolutionSTFTLoss(...))
    (value, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        model, x, delta, key=key
    )
"""

import logging
import os
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key

from eval_808.synths_jax import DDSP808Synth
from src.cqt_jax import CQT, calc_cqt

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class DDSP808Loss(eqx.Module):
    synth: DDSP808Synth
    cqt: CQT
    loss_func: Callable[..., Float[Array, ""]]
    cqt_eps: float = eqx.field(static=True)

    def __init__(
        self,
        synth: DDSP808Synth,
        loss_func: Callable[..., Float[Array, ""]],
        feature_type: str = "cqt",
        cqt_eps: float = 1e-3,
    ) -> None:
        """
        Args:
            synth: The `DDSP808Synth`.
            loss_func: Maps `(x_hat, x)`, both `(bs, 1, n_samples)`, to a scalar,
                preferably an Equinox module (e.g. `MultiResolutionSTFTLoss`).
            feature_type: Only "cqt" is supported, as in `lightning.py`.
            cqt_eps: Scale of the log compression of the CQT.
        """
        if feature_type != "cqt":
            raise NotImplementedError(f"Unsupported feature_type: {feature_type}")
        self.synth = synth
        self.loss_func = loss_func
        self.cqt_eps = cqt_eps
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
        x = self.prepare_target(x)
        U = jax.lax.stop_gradient(self.calc_U(x))
        keys = None if key is None else jax.random.split(key, x.shape[0])
        theta_hat = jax.vmap(model)(U, key=keys)
        x_hat = self.synth(theta_hat, delta)
        loss = self.loss_func(x_hat, x)
        return loss, {"x": x, "x_hat": x_hat, "theta_hat": theta_hat}
