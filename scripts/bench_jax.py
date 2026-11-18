"""Times the JAX training step and its parts, e.g. to compare CPU and GPU:

python scripts/bench_jax.py
XLA_FLAGS=--xla_gpu_deterministic_ops=false python scripts/bench_jax.py
"""

import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

from src.experiments.util import instantiate, load_config, setup_env  # noqa: E402
from src.paths import CONFIGS_DIR  # noqa: E402

config = load_config(os.path.join(CONFIGS_DIR, "train_mss_log_micro_jax.yml"))
setup_env(config)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.experiments.synths_jax import (  # noqa: E402
    biquad_scan,
    exp_decay_envelope,
    highpass_coefficients,
    time_shift,
)
from src.experiments.trainer_jax import (  # noqa: E402
    make_optimizer,
    make_train_step,
    trainable,
)


def bench(name, f, *args, n=5):
    jax.block_until_ready(f(*args))  # Compile
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    print(f"{name:40s} {(time.perf_counter() - t0) / n * 1000:9.1f} ms")


print("backend:", jax.default_backend(), jax.devices())
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", ""))
key = jax.random.key(0)
config["data"]["init_args"]["n_train"] = 16
data = instantiate(config["data"], key, "data")
model = instantiate(config["model"], key, "model")
loss = instantiate(config["loss"], key, "loss")
optim = make_optimizer(config["optimizer"], None, 2)
opt_state = optim.init(trainable(model))
train_step = make_train_step(optim)
x, _, delta = next(data.batches("train"))
x, delta = jnp.asarray(x), jnp.asarray(delta)
bs = x.shape[0]
bench("full train step", train_step, model, opt_state, loss, x, delta, key)

# The parts of the step: the CQT of the target (no gradient), then the model, synth,
# and audio loss (forward + backward)
x_target = loss.prepare_target(x)
U = loss.calc_U(x_target)
theta = jax.random.uniform(key, (bs, loss.synth.n_params))
x_hat = loss.synth(theta, delta)
bench("CQT of the target (fwd)", eqx.filter_jit(lambda l, x_: l.calc_U(x_)), loss, x)


@eqx.filter_jit
@eqx.filter_grad
def model_grad(m, U_, k):
    return jax.vmap(m)(U_, key=jax.random.split(k, U_.shape[0])).sum()


@eqx.filter_jit
def synth_grad(th, s, d):
    return jax.grad(lambda th_: s(th_, d).sum())(th)


@eqx.filter_jit
def audio_loss_grad(x_hat_, f, x_):
    return jax.grad(lambda xh: f(xh, x_))(x_hat_)


bench("model fwd+bwd", model_grad, model, U, key)
bench(f"synth fwd+bwd (scan_unroll={loss.synth.scan_unroll})", synth_grad, theta, loss.synth, delta)  # fmt: skip
bench(f"audio loss fwd+bwd ({type(loss.loss_func).__name__})", audio_loss_grad, x_hat, loss.loss_func, x_target)  # fmt: skip

# The parts of the synth (forward + backward)
n = x.shape[-1]
u = jax.random.uniform(key, (bs, n), minval=-1.0, maxval=1.0)
fe = jax.random.uniform(key, (bs, n + 1), maxval=0.3)  # Phase increments
decay_ms = jnp.full((bs, 1), 500.0)
b, a = highpass_coefficients(jnp.full((bs, 1), 200.0), jnp.full((bs, 1), 0.7), 44100)
scan_add = partial(jax.lax.associative_scan, jnp.add, axis=-1)
for name, cumsum in [("jnp.cumsum", partial(jnp.cumsum, axis=-1)), ("associative_scan", scan_add)]:  # fmt: skip
    f = jax.jit(jax.grad(lambda fe_: jnp.sin(cumsum(fe_)).sum()))
    bench(f"  oscillator phase, {name} (x2)", f, fe)
f = jax.jit(jax.grad(lambda d: exp_decay_envelope(d, 44100, n).sum()))
bench("  decay envelope (x4)", f, decay_ms)
f = jax.jit(jax.grad(lambda u_: time_shift(u_, delta, -2048).sum()))
bench("  time_shift", f, u)
for unroll in [1, 16, 128]:
    f = jax.jit(jax.grad(lambda u_, b_, a_: biquad_scan(u_, b_, a_, unroll).sum(), argnums=(0, 1, 2)))  # fmt: skip
    bench(f"  noise biquad scan, unroll={unroll}", f, u, b, a)
