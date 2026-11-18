"""
JAX / Equinox port of `Spectral2DCNN` from `models.py`.

Like other Equinox models, it takes a single unbatched example; use `jax.vmap` for
batches. Dropout needs a PRNG key in training mode; use `eqx.nn.inference_mode`
for evaluation (the equivalent of `model.eval()`):

    model = Spectral2DCNN(n_bins=108, n_frames=173, ..., key=key_init)
    keys = jax.random.split(key, U.shape[0])
    theta_hat = jax.vmap(model)(U, key=keys)  # train: (bs, n_params)
    theta_hat = jax.vmap(eqx.nn.inference_mode(model))(U)  # eval
"""

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _layer_norm_2d(
    x: Float[Array, "ch h w"], eps: float = 1e-5
) -> Float[Array, "ch h w"]:
    """`nn.LayerNorm([h, w], elementwise_affine=False)` for each channel."""
    mean = jnp.mean(x, axis=(-2, -1), keepdims=True)
    var = jnp.var(x, axis=(-2, -1), keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _prelu() -> eqx.nn.PReLU:
    """PyTorch's default `nn.PReLU()`. The slope is created with an explicit dtype:
    `eqx.nn.PReLU(0.25)` makes it weakly typed, so it changes type after the first
    optimizer update and the jitted training step is compiled again."""
    return eqx.nn.PReLU(jnp.asarray(0.25, dtype=jnp.result_type(float)))


class ParamHead(eqx.Module):
    """Linear -> PReLU -> Dropout -> Linear -> Sigmoid, predicting one parameter."""

    fc1: eqx.nn.Linear
    act: eqx.nn.PReLU
    dropout: eqx.nn.Dropout
    fc2: eqx.nn.Linear

    def __init__(self, latent_dim: int, dropout_prob: float, *, key: Key[Array, ""]):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(latent_dim, latent_dim // 2, key=k1)
        self.act = _prelu()
        self.dropout = eqx.nn.Dropout(dropout_prob)
        self.fc2 = eqx.nn.Linear(latent_dim // 2, 1, key=k2)

    def __call__(
        self, x: Float[Array, "latent_dim"], *, key: Key[Array, ""] | None = None
    ) -> Float[Array, ""]:
        x = self.act(self.fc1(x))
        x = self.dropout(x, key=key)
        return jax.nn.sigmoid(self.fc2(x))[0]


class Spectral2DCNN(eqx.Module):
    """2D CNN over a (log-)CQT that predicts `n_params` values in [0, 1], with a
    separate MLP head per parameter.

    Each block is LayerNorm (optional, no affine) -> Conv2d ("same" padding) ->
    MaxPool2d -> PReLU. The output is averaged over time and frequency, then passes
    through Linear -> PReLU -> Dropout and the parameter heads. Unlike the PyTorch
    version, which returns a list of `n_params` tensors, this returns one array of
    shape `(n_params,)`. Initialization uses the same distributions as PyTorch;
    `load_torch_state_dict` copies weights from a PyTorch model.
    """

    n_bins: int = eqx.field(static=True)
    n_frames: int = eqx.field(static=True)
    in_ch: int = eqx.field(static=True)
    kernel_size: tuple[int, int] = eqx.field(static=True)
    out_channels: tuple[int, ...] = eqx.field(static=True)
    bin_dilations: tuple[int, ...] = eqx.field(static=True)
    temp_dilations: tuple[int, ...] = eqx.field(static=True)
    pool_sizes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    use_ln: bool = eqx.field(static=True)
    dropout_prob: float = eqx.field(static=True)
    n_params: int = eqx.field(static=True)
    convs: list[eqx.nn.Conv2d]
    pools: list[eqx.nn.MaxPool2d]
    acts: list[eqx.nn.PReLU]
    fc: eqx.nn.Linear
    fc_act: eqx.nn.PReLU
    dropout: eqx.nn.Dropout
    param_mlps: list[ParamHead]

    def __init__(
        self,
        n_bins: int,
        n_frames: int,
        in_ch: int = 1,
        kernel_size: Sequence[int] = (3, 3),
        out_channels: Sequence[int] | None = None,
        bin_dilations: Sequence[int] | None = None,
        temp_dilations: Sequence[int] | None = None,
        pool_sizes: Sequence[Sequence[int]] | None = None,
        latent_dim: int = 32,
        use_ln: bool = True,
        dropout_prob: float = 0.25,
        n_params: int = 2,
        *,
        key: Key[Array, ""],
    ) -> None:
        """Arguments and defaults follow the PyTorch `Spectral2DCNN`."""
        if in_ch != 1:
            raise ValueError("The input is a single CQT, so in_ch must be 1")
        if out_channels is None:
            out_channels = [128] * 5
        n_blocks = len(out_channels)
        if bin_dilations is None:
            bin_dilations = [1] * n_blocks
        if temp_dilations is None:
            temp_dilations = [1] * n_blocks
        if pool_sizes is None:
            pool_sizes = [(2, 2)] * n_blocks
        if not n_blocks == len(bin_dilations) == len(temp_dilations) == len(pool_sizes):
            raise ValueError("Per-block arguments must all have the same length")

        self.n_bins = n_bins
        self.n_frames = n_frames
        self.in_ch = in_ch
        self.kernel_size = tuple(kernel_size)
        self.out_channels = tuple(out_channels)
        self.bin_dilations = tuple(bin_dilations)
        self.temp_dilations = tuple(temp_dilations)
        self.pool_sizes = tuple(tuple(p) for p in pool_sizes)
        self.latent_dim = latent_dim
        self.use_ln = use_ln
        self.dropout_prob = dropout_prob
        self.n_params = n_params

        keys = jax.random.split(key, n_blocks + 1 + n_params)
        self.convs, self.pools, self.acts = [], [], []
        ch = in_ch
        for idx in range(n_blocks):
            self.convs.append(
                eqx.nn.Conv2d(
                    ch,
                    self.out_channels[idx],
                    self.kernel_size,
                    stride=1,
                    padding="SAME",
                    dilation=(self.bin_dilations[idx], self.temp_dilations[idx]),
                    key=keys[idx],
                )
            )
            pool = self.pool_sizes[idx]
            self.pools.append(eqx.nn.MaxPool2d(kernel_size=pool, stride=pool))
            self.acts.append(_prelu())
            ch = self.out_channels[idx]

        self.fc = eqx.nn.Linear(self.out_channels[-1], latent_dim, key=keys[n_blocks])
        self.fc_act = _prelu()
        self.dropout = eqx.nn.Dropout(dropout_prob)
        self.param_mlps = [
            ParamHead(latent_dim, dropout_prob, key=k) for k in keys[n_blocks + 1 :]
        ]

    def __call__(
        self,
        x: Float[Array, "n_bins n_frames"],
        *,
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "n_params"]:
        """
        Args:
            x: A (log-)CQT of shape `(n_bins, n_frames)`.
            key: PRNG key for dropout. Required unless in inference mode.
        """
        if x.shape != (self.n_bins, self.n_frames):
            raise ValueError(
                f"Expected input of shape {(self.n_bins, self.n_frames)}, got {x.shape}"
            )
        x = x[None]  # Add the channel axis
        for conv, pool, act in zip(self.convs, self.pools, self.acts):
            if self.use_ln:
                x = _layer_norm_2d(x)
            x = act(pool(conv(x)))
        x = jnp.mean(x, axis=(-2, -1))
        x = self.fc_act(self.fc(x))

        if key is None:
            keys = [None] * (1 + self.n_params)
        else:
            keys = list(jax.random.split(key, 1 + self.n_params))
        x = self.dropout(x, key=keys[0])
        out = [head(x, key=k) for head, k in zip(self.param_mlps, keys[1:])]
        return jnp.stack(out)


def load_torch_state_dict(
    model: Spectral2DCNN, state_dict: Mapping[str, Any], prefix: str = ""
) -> Spectral2DCNN:
    """Returns a copy of `model` with the weights of a PyTorch `Spectral2DCNN`.

    Args:
        model: A JAX model with the same architecture.
        state_dict: The PyTorch `state_dict()` (tensors or NumPy arrays).
        prefix: Prefix to strip from the keys, e.g. "model._orig_mod." for a
            checkpoint of the compiled model in `DDSP808LightingModule`.
    """
    sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            if hasattr(v, "detach"):  # torch.Tensor
                v = v.detach().cpu().numpy()
            sd[k[len(prefix) :]] = np.asarray(v)

    def take(name: str, like: Array) -> Array:
        if name not in sd:
            raise KeyError(f"Missing key in state_dict: {prefix}{name}")
        val = sd.pop(name)
        if val.size != like.size:
            raise ValueError(f"Shape mismatch for {name}: {val.shape} vs {like.shape}")
        return jnp.asarray(val, dtype=like.dtype).reshape(like.shape)

    def weight_bias(layer: Any, name: str) -> Any:
        new = (take(f"{name}.weight", layer.weight), take(f"{name}.bias", layer.bias))
        return eqx.tree_at(lambda m: (m.weight, m.bias), layer, new)

    def prelu(act: eqx.nn.PReLU, name: str) -> eqx.nn.PReLU:
        new = take(f"{name}.weight", act.negative_slope)
        return eqx.tree_at(lambda m: m.negative_slope, act, new)

    # Indices inside the PyTorch nn.Sequential: [LayerNorm], Conv2d, MaxPool2d, PReLU
    per_block = 4 if model.use_ln else 3
    offset = 1 if model.use_ln else 0
    convs, acts = [], []
    for idx, (conv, act) in enumerate(zip(model.convs, model.acts)):
        conv_idx = idx * per_block + offset
        convs.append(weight_bias(conv, f"cnn.{conv_idx}"))
        acts.append(prelu(act, f"cnn.{conv_idx + 2}"))
    heads = []
    for idx, head in enumerate(model.param_mlps):
        head = eqx.tree_at(
            lambda m: (m.fc1, m.act, m.fc2),
            head,
            (
                weight_bias(head.fc1, f"param_mlps.{idx}.0"),
                prelu(head.act, f"param_mlps.{idx}.1"),
                weight_bias(head.fc2, f"param_mlps.{idx}.3"),
            ),
        )
        heads.append(head)
    fc = weight_bias(model.fc, "fc")
    fc_act = prelu(model.fc_act, "fc_act")
    if sd:
        raise ValueError(f"Unused keys in state_dict: {sorted(sd)}")

    return eqx.tree_at(
        lambda m: (m.convs, m.acts, m.fc, m.fc_act, m.param_mlps),
        model,
        (convs, acts, fc, fc_act, heads),
    )
