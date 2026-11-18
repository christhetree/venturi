"""Times the full JAX training step vs. the synth's biquad scan alone, e.g. to compare
CPU and GPU: python scripts/bench_jax.py"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

from src.experiments.util import instantiate, load_config, setup_env  # noqa: E402
from src.paths import CONFIGS_DIR  # noqa: E402

config = load_config(os.path.join(CONFIGS_DIR, "train_mss_log_micro_jax.yml"))
setup_env(config)

import jax
import jax.numpy as jnp

from src.experiments import synths_jax
from src.experiments.trainer_jax import make_optimizer, make_train_step, trainable


def bench(name, f, *args, n=5):
    jax.block_until_ready(f(*args))  # Compile
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    print(f"{name:45s} {(time.perf_counter() - t0) / n * 1000:9.1f} ms")


print("backend:", jax.default_backend(), jax.devices())
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
bench("full train step", train_step, model, opt_state, loss, x, delta, key)

# The IIR scan alone (forward + backward), with the shapes of the synth
bs, n = x.shape[0], x.shape[-1]
u = jax.random.uniform(key, (bs, n), minval=-1.0, maxval=1.0)
b, a = synths_jax.highpass_coefficients(
    jnp.full((bs, 1), 200.0), jnp.full((bs, 1), 0.7), 44100
)


def biquad_scan(x, b, a, unroll):
    xp = jnp.pad(x, ((0, 0), (2, 0)))
    u = b[:, 2:3] * xp[:, :-2] + b[:, 1:2] * xp[:, 1:-1] + b[:, 0:1] * xp[:, 2:]
    a1, a2 = a[:, 0], a[:, 1]

    def step(carry, u_t):
        y1, y2 = carry
        y_t = u_t - (a2 * y2 + a1 * y1)
        return (y_t, y1), y_t

    zeros = jnp.zeros_like(u[:, 0])
    return jax.lax.scan(step, (zeros, zeros), u.T, unroll=unroll)[1].T


for unroll in [1, 16, 128]:
    f = jax.jit(jax.value_and_grad(lambda u_: biquad_scan(u_, b, a, unroll).sum()))
    bench(f"biquad scan fwd+bwd, unroll={unroll}", f, u)
