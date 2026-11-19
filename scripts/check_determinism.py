"""
Checks whether training is deterministic on this machine (e.g. a GPU): runs each part
of the training step, and a few full training steps from the same initialization and
batches, several times in this process and compares the results bit for bit. Then
prints a fingerprint of the training results: run this script twice (in two
processes, launched like the training runs) and compare the fingerprints to check
determinism across runs.

    python scripts/check_determinism.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

from src.experiments.util import load_config, setup_env  # noqa: E402
from src.paths import CONFIGS_DIR  # noqa: E402

CONFIG = "train.yml"
N_STEPS = 10  # Training steps compared
N_REPEATS = 20  # Runs of each part (nondeterminism can be intermittent)
N_TRAIN_REPEATS = 3  # Runs of the training steps

config = load_config(os.path.join(CONFIGS_DIR, CONFIG))
config["custom"]["run_name"] = "determinism_check"
setup_env(config)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jaxlib  # noqa: E402
import numpy as np  # noqa: E402

from src.experiments.trainer_jax import (  # noqa: E402
    eval_step,
    make_train_step,
    setup,
    trainable,
)

print(f"jax {jax.__version__}, jaxlib {jaxlib.__version__}, devices {jax.devices()}")
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '')}")
print(f"JAX compilation cache: {jax.config.jax_compilation_cache_dir}")

run = setup(config, seed=config.get("seed_everything", 0))
for d in [run.dir, os.path.dirname(run.dir)]:  # Created by `setup`, unused here
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
batches = [b for _, b in zip(range(N_STEPS), run.data.batches("train"))]
x, _, delta = batches[0]
key = jax.random.key(0)


def as_arrays(tree):
    return [np.asarray(leaf) for leaf in jax.tree.leaves(jax.device_get(tree))]


def max_diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b))) if a.size else 0.0


def compare(name, fn, n_repeats=N_REPEATS):
    """Runs `fn` `n_repeats` times and reports whether the outputs are identical."""
    outs = [as_arrays(fn()) for _ in range(n_repeats)]
    n_diff = sum(
        not all(np.array_equal(a, b) for a, b in zip(outs[0], out)) for out in outs[1:]
    )
    if n_diff == 0:
        result = f"deterministic ({n_repeats} runs)"
    else:
        diff = max(max_diff(a, b) for out in outs[1:] for a, b in zip(outs[0], out))
        result = (
            f"NOT deterministic: {n_diff} of {n_repeats - 1} runs differ from the "
            f"first (max diff {diff:.2e})"
        )
    print(f"{name:38s} {result}")
    return outs


# The parts of the training step, on the first batch
calc_U = eqx.filter_jit(lambda l, x_: l.calc_U(l.prepare_target(x_)))
U = calc_U(run.loss, x)
compare("CQT of the target (fwd)", lambda: calc_U(run.loss, x))


@eqx.filter_jit
@eqx.filter_grad
def model_grad(m, U_, k):
    return jax.vmap(m)(U_, key=jax.random.split(k, U_.shape[0])).sum()


compare("model (fwd+bwd)", lambda: model_grad(run.model, U, key))

theta = jax.random.uniform(key, (x.shape[0], run.loss.synth.n_params))
synth_grad = eqx.filter_jit(lambda th, s, d: jax.grad(lambda th_: s(th_, d).sum())(th))
compare("synth (fwd+bwd)", lambda: synth_grad(theta, run.loss.synth, delta))

x_target = run.loss.prepare_target(x)
x_hat = run.loss.synth(theta, delta)
loss_grad = eqx.filter_jit(lambda xh, f, x_: jax.grad(lambda a: f(a, x_))(xh))
compare("audio loss (fwd+bwd)", lambda: loss_grad(x_hat, run.loss.loss_func, x_target))

compare(
    "eval step (loss and metrics)",
    lambda: eval_step(run.model, run.loss, run.metrics, x)[0],
)

# The whole training objective and the optimizer update, at the initialization
objective_grad = eqx.filter_jit(
    eqx.filter_value_and_grad(lambda m, l, x_, d, k: l(m, x_, d, key=k)[0])
)
compare(
    "training objective (fwd+bwd)",
    lambda: objective_grad(run.model, run.loss, x, delta, key),
)
compare(
    "  without dropout",
    lambda: objective_grad(eqx.nn.inference_mode(run.model), run.loss, x, delta, key),
)
_, grads = objective_grad(run.model, run.loss, x, delta, key)
opt_state = run.optim.init(trainable(run.model))
update = eqx.filter_jit(lambda g, s, m: run.optim.update(g, s, trainable(m))[0])
compare("optimizer update", lambda: update(grads, opt_state, run.model))

# Full training steps from the same initialization
train_step = make_train_step(run.optim)


def train():
    model, opt_state = run.model, run.optim.init(trainable(run.model))
    losses = []
    for i, (x_, _, delta_) in enumerate(batches):
        step_key = jax.random.fold_in(key, i)
        model, opt_state, loss = train_step(
            model, opt_state, run.loss, x_, delta_, step_key
        )
        losses.append(loss)
    return losses, trainable(model)


outs = compare(f"{N_STEPS} training steps", train, N_TRAIN_REPEATS)
losses = np.array([out[:N_STEPS] for out in outs])  # (n_repeats, n_steps)
differs = np.any(losses != losses[0], axis=0)
if differs.any():
    step = int(np.argmax(differs))
    print(
        f"  The losses first differ at step {step}: "
        + ", ".join(f"{v:.6f}" for v in losses[:, step])
    )
fingerprint = hashlib.sha256(b"".join(a.tobytes() for a in outs[0])).hexdigest()[:16]
print(f"Fingerprint of {N_STEPS} training steps: {fingerprint}")
