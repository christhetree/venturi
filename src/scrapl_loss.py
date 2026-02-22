import functools
import logging
import os
from typing import Optional, List, Dict, Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import torch as tr
from jaxtyping import Array, Float
from torch import Tensor as T, nn
from torch.nn import Parameter

import hessian_eigenthings
from hessian_eigenthings.operator import LambdaOperator

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# Warmup methods ===================================================================
def _calc_batch_theta_param_grad(
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        loss_fn: Callable[[T, T], T],
        theta_fn_kwargs: Dict[str, Any],
        params: List[Parameter],
        n_theta: int,
        theta_idx: Optional[int] = None,
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> T:
    if synth_fn_kwargs is None:
        synth_fn_kwargs = {}
    assert "x" in theta_fn_kwargs, "x must be provided in theta_fn_kwargs"
    assert params, "params must not be empty"
    assert all(p.grad is None for p in params), "params must have no grad"
    if theta_idx is not None:
        assert 0 <= theta_idx < n_theta
    theta_hat = theta_fn(**theta_fn_kwargs)
    assert theta_hat.ndim == 2
    assert theta_hat.size(1) == n_theta
    bs = theta_hat.size(0)
    x_hat = synth_fn(theta_hat, **synth_fn_kwargs)
    x = theta_fn_kwargs["x"]
    loss = loss_fn(x, x_hat)

    theta_grad = tr.autograd.grad(
        loss, theta_hat, create_graph=True, materialize_grads=True
    )[0]
    if theta_idx is None:
        # Expand theta_grad to vectorize individual theta grad calculations
        theta_grad_ex = theta_grad.unsqueeze(0).expand(n_theta, -1, -1)
        theta_eye = tr.eye(n_theta, device=theta_grad.device, dtype=theta_grad.dtype)
        theta_eye_ex = theta_eye.unsqueeze(1).expand(-1, bs, -1)
        theta_grad_ex = theta_grad_ex * theta_eye_ex
        is_grads_batched = True
    else:
        # Zero out all but the specified theta grad
        mask = tr.zeros_like(theta_grad)
        mask[:, theta_idx] = 1.0
        theta_grad_ex = theta_grad * mask
        is_grads_batched = False
    # Calculate maybe vectorized param grad
    theta_param_grads = tr.autograd.grad(
        theta_hat,
        params,
        grad_outputs=theta_grad_ex,
        create_graph=True,
        materialize_grads=True,
        is_grads_batched=is_grads_batched,
    )
    if theta_idx is None:
        theta_param_grad = tr.cat(
            [g.view(n_theta, -1) for g in theta_param_grads], dim=1
        )
    else:
        theta_param_grad = tr.cat([g.view(-1) for g in theta_param_grads])
    return theta_param_grad


def _calc_param_hvp(
        tangent: T,
        param_grad: T,
        params: List[Parameter],
        retain_graph: bool = False,
) -> T:
    assert tangent.ndim == 1
    assert param_grad.shape == tangent.shape
    assert all(p.grad is None for p in params), "params must have no grad"
    param_hvps = tr.autograd.grad(
        param_grad,
        params,
        grad_outputs=tangent,
        materialize_grads=True,
        retain_graph=retain_graph,
    )
    param_hvp = tr.cat([g.contiguous().view(-1) for g in param_hvps])
    return param_hvp


def _calc_largest_eig(
        param_grad: T,
        params: List[Parameter],
        n_iter: int = 20,
) -> T:
    apply_fn = functools.partial(
        _calc_param_hvp,
        param_grad=param_grad,
        params=params,
        retain_graph=True,
    )
    size = param_grad.size(0)
    hvp_op = LambdaOperator(apply_fn, size)
    use_gpu = param_grad.is_cuda
    eigs, _ = hessian_eigenthings.deflated_power_iteration(
        operator=hvp_op,
        num_eigenthings=1,
        power_iter_steps=n_iter,
        to_numpy=False,
        use_gpu=use_gpu,
    )
    eig1 = tr.from_numpy(eigs.copy()).float()
    return eig1


def calc_theta_eigs(
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        loss_fn: Callable[[T, T], T],
        theta_fn_kwargs: Dict[str, Any],
        params: List[Parameter],
        n_theta: int,
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
        n_iter: int = 20,
) -> T:
    theta_param_grad = _calc_batch_theta_param_grad(
        theta_fn,
        synth_fn,
        loss_fn,
        theta_fn_kwargs,
        params,
        n_theta,
        synth_fn_kwargs=synth_fn_kwargs,
    )
    theta_eigs = []
    for theta_idx in range(n_theta):
        param_grad = theta_param_grad[theta_idx, :]
        eig1 = _calc_largest_eig(param_grad, params, n_iter=n_iter)
        theta_eigs.append(eig1)
    theta_eigs = tr.cat(theta_eigs, dim=0)
    return theta_eigs


def _calc_param_hvp_multibatch(
        tangent: T,
        theta_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        loss_fn: Callable[[T, T], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        n_theta: int,
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
) -> T:
    if synth_fn_kwargs is None:
        assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
        synth_fn_kwargs = [None] * len(theta_fn_kwargs)
    else:
        assert len(synth_fn_kwargs) == len(theta_fn_kwargs), (
            f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
            f"len(synth_fn_kwargs) ({len(synth_fn_kwargs)})"
        )
    param_hvp = None
    for curr_theta_fn_kwargs, curr_synth_fn_kwargs in zip(
            theta_fn_kwargs, synth_fn_kwargs
    ):
        curr_param_grad = _calc_batch_theta_param_grad(
            theta_fn,
            synth_fn,
            loss_fn,
            curr_theta_fn_kwargs,
            params,
            n_theta,
            theta_idx=theta_idx,
            synth_fn_kwargs=curr_synth_fn_kwargs,
        )
        curr_param_hvp = _calc_param_hvp(
            tangent, curr_param_grad, params, retain_graph=False
        )
        # TODO: should we average here?
        if param_hvp is None:
            param_hvp = curr_param_hvp
        else:
            param_hvp += curr_param_hvp
    return param_hvp


def _calc_theta_largest_eig_multibatch(
        theta_idx: int,
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        loss_fn: Callable[[T, T], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        n_theta: int,
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
) -> T:
    apply_fn = functools.partial(
        _calc_param_hvp_multibatch,
        theta_idx=theta_idx,
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        loss_fn=loss_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
        n_theta=n_theta,
        synth_fn_kwargs=synth_fn_kwargs,
    )
    size = sum(p.numel() for p in params)
    hvp_op = LambdaOperator(apply_fn, size)
    use_gpu = params[0].is_cuda
    eigs, _ = hessian_eigenthings.deflated_power_iteration(
        operator=hvp_op,
        num_eigenthings=1,
        power_iter_steps=n_iter,
        to_numpy=False,
        use_gpu=use_gpu,
    )
    eig1 = tr.from_numpy(eigs.copy()).float()
    return eig1


def calc_theta_eigs_multibatch(
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        loss_fn: Callable[[T, T], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        n_theta: int,
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
) -> T:
    theta_eigs = []
    for theta_idx in range(n_theta):
        eig1 = _calc_theta_largest_eig_multibatch(
            theta_idx,
            theta_fn,
            synth_fn,
            loss_fn,
            theta_fn_kwargs,
            params,
            n_theta,
            synth_fn_kwargs=synth_fn_kwargs,
            n_iter=n_iter,
        )
        theta_eigs.append(eig1)
    theta_eigs = tr.cat(theta_eigs, dim=0)
    return theta_eigs


def _aggregate_vals(
        vals: T,
        n_theta: int,
        agg: Literal["none", "mean", "max", "med"] = "none",
) -> T:
    assert vals.ndim == 2
    assert vals.size(1) == n_theta
    if agg == "none":
        assert vals.size(0) == 1
        vals = vals[0]
    elif agg == "mean":
        vals = vals.mean(dim=0)
    elif agg == "max":
        vals = vals.max(dim=0).values
    elif agg == "med":
        vals = vals.median(dim=0).values
    else:
        raise ValueError(f"Invalid agg = {agg}")
    return vals


def check_is_deterministic(
        theta_fn: Callable[..., T],
        theta_fn_kwargs: Dict[str, Any],
        synth: Callable[[T, ...], T],
        synth_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    theta_hat_1 = theta_fn(**theta_fn_kwargs)
    theta_hat_2 = theta_fn(**theta_fn_kwargs)
    if not tr.allclose(theta_hat_1, theta_hat_2):
        log.warning(
            f"theta_fn is not deterministic: "
            f"theta_hat_1 = {theta_hat_1}, theta_hat_2 = {theta_hat_2}"
        )
        return False
    if synth_fn_kwargs is None:
        synth_fn_kwargs = {}
    x_hat_1 = synth(theta_hat_1, **synth_fn_kwargs)
    x_hat_2 = synth(theta_hat_2, **synth_fn_kwargs)
    if not tr.allclose(x_hat_1, x_hat_2):
        log.warning(f"synth_fn is not deterministic")
        return False
    return True


def warmup_lc_hvp(
        theta_fn: Callable[..., T],
        synth_fn: Callable[[T, ...], T],
        loss_fn: Callable[[T, T], T],
        theta_fn_kwargs: List[Dict[str, Any]],
        params: List[Parameter],
        n_theta: int,
        synth_fn_kwargs: Optional[List[Dict[str, Any]]] = None,
        n_iter: int = 20,
        agg: Literal["none", "mean", "max", "med"] = "none",
        force_multibatch: bool = False,
) -> T:
    assert params, "params must not be empty"
    assert all(
        not p.grad for p in params
    ), "params must have no previous gradients before warmup"
    assert theta_fn_kwargs, "theta_fn_kwargs must not be empty"
    if synth_fn_kwargs is not None:
        assert len(synth_fn_kwargs) == len(theta_fn_kwargs), (
            f"len(theta_fn_kwargs) ({len(theta_fn_kwargs)}) != "
            f"len(synth_fn_kwargs) ({len(synth_fn_kwargs)})"
        )

    # Check determinism
    theta_fn_kwargs_batch = theta_fn_kwargs[0]
    synth_fn_kwargs_batch = {}
    if synth_fn_kwargs is not None:
        synth_fn_kwargs_batch = synth_fn_kwargs[0]
    check_is_deterministic(
        theta_fn, theta_fn_kwargs_batch, synth_fn, synth_fn_kwargs_batch
    )

    # Determine whether to use multibatch or not
    is_multibatch = force_multibatch or len(theta_fn_kwargs) > 1
    log.info(
        f"Starting warmup_lc_hvp with agg = {agg} for {len(params)} "
        f"parameter(s) and {len(theta_fn_kwargs)} batch(es), {n_iter} iter "
        f"(multibatch = {is_multibatch})"
    )
    if is_multibatch:
        calc_theta_eigs_fn = calc_theta_eigs_multibatch
    else:
        calc_theta_eigs_fn = calc_theta_eigs
        theta_fn_kwargs = theta_fn_kwargs[0]
        if synth_fn_kwargs is not None:
            synth_fn_kwargs = synth_fn_kwargs[0]
    # Separate the params for separate computations if aggregating LCs across them
    if agg == "none":
        param_groups = [params]
    else:
        param_groups = [[p] for p in params]
    # Compute the theta LCs
    vals = []
    for param_group in param_groups:
        curr_vals = calc_theta_eigs_fn(
            theta_fn=theta_fn,
            synth_fn=synth_fn,
            loss_fn=loss_fn,
            theta_fn_kwargs=theta_fn_kwargs,
            params=param_group,
            n_theta=n_theta,
            synth_fn_kwargs=synth_fn_kwargs,
            n_iter=n_iter,
        )
        # TODO: double check this
        curr_vals = curr_vals.abs()
        vals.append(curr_vals)
        log.info(f"curr_vals = {curr_vals}")
    # Aggregate the theta LCs across all param groups
    vals = tr.stack(vals, dim=0)
    vals = _aggregate_vals(vals, n_theta, agg=agg)
    return vals


@tr.no_grad()
def load_jax_to_pytorch(jax_model: eqx.Module, torch_model: nn.Module) -> None:
    # Map JAX layers to PyTorch layers
    # JAX layers[0] -> torch_model[0]
    # JAX layers[2] -> torch_model[2]

    # Layer 1
    torch_model[0].weight.copy_(tr.from_dlpack(jax_model.layers[0].weight))
    torch_model[0].bias.copy_(tr.from_dlpack(jax_model.layers[0].bias))

    # Layer 2
    torch_model[2].weight.copy_(tr.from_dlpack(jax_model.layers[2].weight))
    torch_model[2].bias.copy_(tr.from_dlpack(jax_model.layers[2].bias))


class EncoderJAX(eqx.Module):
    layers: list

    def __init__(self, n_samples: int, n_theta: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(n_samples, n_theta, key=key1),
            eqx.nn.PReLU(),
            eqx.nn.Linear(n_theta, n_theta, key=key2),
            jax.nn.sigmoid,
        ]

    def __call__(self, x: Float[Array, "n_samples"]) -> Float[Array, "n_theta"]:
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderJAX(eqx.Module):
    layers: list

    def __init__(self, n_theta: int, n_samples: int, key: Array):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(n_theta, n_theta, key=key1),
            eqx.nn.PReLU(),
            eqx.nn.Linear(n_theta, n_samples, key=key2),
            jax.nn.tanh,
        ]

    def __call__(self, theta: Float[Array, "n_theta"]) -> Float[Array, "n_samples"]:
        x = theta
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    tr.set_printoptions(precision=4, sci_mode=False)
    seed = 42
    master_key = jax.random.PRNGKey(seed)
    tr.manual_seed(seed)
    bs = 4
    n_samples = 8096
    n_theta = 3
    n_batches = 1

    enc_key, dec_key, data_key = jax.random.split(master_key, 3)
    # Encoder ==========================================================================
    encoder_jax = EncoderJAX(n_samples, n_theta, key=enc_key)


    @eqx.filter_jit
    def theta_fn_jax(x):
        return encoder_jax(x)


    encoder_torch = nn.Sequential(
        nn.Linear(n_samples, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_theta),
        nn.Sigmoid(),
    )
    load_jax_to_pytorch(encoder_jax, encoder_torch)
    theta_fn_torch = lambda x: encoder_torch(x)

    x_jax = jax.random.uniform(data_key, (bs, n_samples))
    theta_hat_jax = jax.vmap(theta_fn_jax)(x_jax)

    x_torch = tr.from_dlpack(x_jax)
    theta_hat_torch = encoder_torch(x_torch)

    # Check that the outputs are close by converting pytorch to jax
    theta_hat_torch_to_jax = jax.dlpack.from_dlpack(theta_hat_torch.detach())
    assert jnp.allclose(theta_hat_jax, theta_hat_torch_to_jax, atol=1e-7)

    # Decoder ==========================================================================
    decoder_jax = DecoderJAX(n_theta, n_samples, key=dec_key)


    @eqx.filter_jit
    def synth_fn_jax(theta):
        return decoder_jax(theta)


    decoder_torch = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )
    load_jax_to_pytorch(decoder_jax, decoder_torch)
    synth_fn_torch = lambda theta: decoder_torch(theta)

    x_hat_jax = jax.vmap(synth_fn_jax)(theta_hat_jax)
    x_hat_torch = decoder_torch(theta_hat_torch)

    x_hat_torch_to_jax = jax.dlpack.from_dlpack(x_hat_torch.detach())
    assert jnp.allclose(x_hat_jax, x_hat_torch_to_jax, atol=1e-7)
    exit()

    decoder = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.PReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )
    synth_fn = lambda theta: decoder(theta)

    theta_is_batches = [tr.rand((bs, n_samples)) for _ in range(n_batches)]
    theta_fn_kwargs = [{"x": batch} for batch in theta_is_batches]

    params = [p for p in encoder_torch.parameters()]

    loss_fn = nn.MSELoss()

    vals = warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        loss_fn=loss_fn,
        theta_fn_kwargs=theta_fn_kwargs,
        params=params,
        n_theta=n_theta,
        n_iter=20,
        agg="none",
        force_multibatch=True,
    )
