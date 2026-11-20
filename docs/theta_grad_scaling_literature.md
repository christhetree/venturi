# Related work: rebalancing the gradients of the synth parameters

*Literature review, November 2026.*

## Context

In the 808 experiments, `Spectral2DCNN` maps the CQT of a target sound to the 14
parameters θ of `DDSP808Synth`, and the loss (`LogMSSLoss`) compares the synthesized
audio with the target. Each parameter has its own head (Linear → PReLU → Linear →
sigmoid) on top of a shared CNN trunk. On one batch at initialization, the four
pitch parameters (`osc*_freq`, `osc*_mod`) account for about 72% of the trunk's
gradient, while the decays, `noise_freq`, and `noise_q` each get 0.2–1.3%.

`DDSP808Loss` (`src/experiments/losses_jax.py`) has two options that rescale
∂L/∂θ_i, the gradient of the loss w.r.t. each predicted synth parameter, before
backpropagation continues into the model. The forward pass and the loss values are
unchanged:

- **`use_curvature`**: multiplies ∂L/∂θ_i by $w_i = (c_i / \bar{c})^{-\alpha}$, where
  $c_i$ is the curvature of parameter $i$ from `scripts/theta_curvature.py` (θ-IS
  warmup, `src/scrapl_jax/warmup.py`), $\bar{c}$ is the geometric mean of the
  curvatures, and $\alpha$ is `curvature_alpha`. The weights are fixed for the
  whole run.
- **`use_grad_rms`**: normalizes ∂L/∂θ_i like Adam without momentum
  (`optax.scale_by_adam(b1=0)`), using the mean of its square over the batch:
  $\nu_i \leftarrow \beta_2 \nu_i + (1-\beta_2)\,\mathrm{mean}_b\, g_{b,i}^2$, then
  $\hat{g}_{b,i} = g_{b,i} / (\sqrt{\nu_i / (1-\beta_2^t)} + \epsilon)$, where
  $\beta_2$ is `grad_rms_b2` and $\epsilon$ is `grad_rms_eps`. Adam's momentum
  ($\beta_1$) is not used: each batch's gradients belong to that batch's items and
  can only go back through the model for those items.

Since each head only receives the gradient of its own parameter, and AdamW cancels a
constant scale on each weight's gradient, both options mainly change how the 14
parameters share the trunk's gradient.

## Summary

No paper was found that does exactly what `use_grad_rms` does: Adam's bias-corrected
running mean of squared gradients, one per output coordinate, applied where the
model hands its predictions to a downstream module (here, the synth). The pieces are
established:

| Work | Where gradients are rescaled | Statistic | Granularity |
|---|---|---|---|
| **This repo, `use_grad_rms`** | ∂L/∂θ at the synth input, then through the model | Bias-corrected running mean of the square (β₂) | Per synth parameter |
| **This repo, `use_curvature`** | Same | Fixed curvature from the θ-IS warmup | Per synth parameter |
| EnCodec loss balancer [1] | ∂ℓ_k/∂x̂ at the model output, then through the model | Bias-corrected running mean of the norm (0.999) | Per loss |
| Backward Gradient Normalization [2] | At chosen points inside the network | Current batch | Per layer |
| MGDA-UB [3] | Gradients w.r.t. the shared representation | Min-norm combination of tasks | Per task |
| AdaTask [4] | Weight gradients of shared parameters | Separate Adam accumulators per task | Per task × weight |
| GradNorm [5] | Loss weights (learned) | Gradient norms at the last shared layer | Per task |
| PopArt [6, 7] | Targets, with outputs preserved | Running mean and spread of targets | Per task |
| NovoGrad, Adam-mini, LARS/LAMB [9–12] | Weight gradients | Second moment or norm | Per layer or block |
| K-FAC [16] | Weight gradients of each layer | Running second moments of the gradients at layer outputs (and of the inputs) | Per layer |

## 1. Normalizing the backward signal partway through the network

These are the closest to the mechanism of both options: rescale the gradient at one
point, then let backpropagation continue through everything before it.

- **EnCodec's loss balancer** (Défossez et al. [1]) is the closest precedent, and it
  is in audio. For each loss ℓ_k, it computes the gradient w.r.t. the model's output
  audio x̂, divides it by a bias-corrected running average (decay 0.999) of its norm,
  sums the normalized gradients with the desired weights, and backpropagates the sum
  through the model (`input.backward(out_grad)` in the audiocraft code). The
  running average is the `flashy` `averager`, which divides by the sum of decayed
  weights, as Adam's bias correction does. Compared with `use_grad_rms`: per loss
  instead of per synth parameter, at the decoder output instead of the decoder input,
  and a running mean of the norm instead of the square. Its motivation, decoupling
  each term's contribution from its typical gradient scale, is the same.
- **Backward Gradient Normalization** (Cabana & Lago-Fernández [2]) inserts layers
  that are the identity in the forward pass and rescale the gradient in the backward
  pass, the same mechanism as `scale_grad` in this repo. It uses statistics of the
  current batch rather than a running mean, and targets vanishing and exploding
  gradients in very deep networks rather than balancing outputs.
- **MGDA-UB** (Sener & Koltun [3]) balances tasks with the gradients w.r.t. the
  shared representation instead of the weights. It supports intervening at a
  representation such as θ, rather than in the weight space.

## 2. Each synth parameter as a task: adaptive multi-task balancing

The 14 synth parameters can be seen as 14 tasks sharing a trunk, and the trunk is
dominated by the tasks with the largest gradients.

- **AdaTask** (Yang et al. [4]) is "Adam per task": for each shared weight, it keeps
  a separate accumulator (and learning rate) per task, for AdaGrad, RMSProp, and
  Adam, because otherwise one or a few tasks' accumulated gradients dominate the
  shared weights. It works at a finer granularity than `use_grad_rms` (per task and
  per weight, versus one statistic per parameter at the heads' outputs), with the
  same motivation.
- **GradNorm** (Chen et al. [5]) learns task weights so that each task's gradient
  norm at the last shared layer reaches a target that depends on how fast the task is
  training. Its exponent α sets the strength of the balancing, as `curvature_alpha`
  does here.
- **PopArt** (van Hasselt et al. [6]; multi-task use in Hessel et al. [7]) tracks
  running statistics of each task's targets and normalizes them, adjusting the last
  layer so that the outputs are unchanged. This makes the size of the updates
  independent of each task's scale while the tasks share one network.
- **Gradient pathologies in PINNs** (Wang, Teng & Perdikaris [8]) identifies an
  imbalance in the magnitudes of the backpropagated gradients of the terms of a
  composite loss, and sets adaptive weights from gradient statistics during
  training.

## 3. Adam at the module level, on the weights

These normalize the weights' gradients per layer or block. `use_grad_rms` instead
normalizes the gradient at the interface with the synth, which then affects every
layer before it.

- **NovoGrad** (Ginsburg et al. [9]) keeps one second moment per layer, normalizes
  each layer's gradient with it, and decouples weight decay (`optax.novograd`).
- **Adam-mini** (Zhang et al. [10]) keeps one second moment per block of parameters,
  with the blocks chosen from the structure of the Hessian.
- **LARS** and **LAMB** (You et al. [11, 12]) normalize the size of each layer's
  update relative to the norm of its weights.

## 4. Curvature-based counterparts (`use_curvature`)

- **Becker & LeCun** [13] and **Efficient BackProp** (LeCun et al. [14])
  backpropagate a diagonal approximation of the curvature (Gauss–Newton or Hessian)
  through the network to set learning rates per unit or per weight ("stochastic
  diagonal Levenberg–Marquardt").
- **The backpropagated metric** (Ollivier [15]) defines a metric on each unit's
  activity, working backward from the output layer, and derives unitwise and
  quasi-diagonal natural-gradient methods from it.
- **K-FAC** (Martens & Grosse [16]) preconditions each layer's weights with running
  averages of the second moments of its inputs and of the gradients at its outputs.
  The ν of `use_grad_rms` is the diagonal of the second of these statistics at the θ
  layer. The difference: K-FAC uses it only for that layer's weights, while
  `use_grad_rms` passes the normalized gradient on to all earlier layers.
- **Perceptual-Neural-Physical sound matching** (Han, Lostanlen & Lagrange [17]) builds
  a Riemannian metric in synth-parameter space from the Jacobian of the synth and the
  joint time–frequency scattering transform, as a quadratic approximation of the
  spectral loss, and trains with it on drum sounds. It is the sound-matching analogue
  of curvature-based scaling in θ space. A follow-up extends it to an AM/FM
  arpeggiator and a rectangular membrane [18].

## Positioning

`use_grad_rms` can be framed as EnCodec's loss balancer [1] (a bias-corrected running
normalization of the gradient at a module interface, in audio) and AdaTask's per-task
Adam [4], applied per synth parameter at the input of the synth. `use_curvature` is a
fixed, diagonal, curvature-based preconditioner at the same interface, in the
lineage of [13–17], with curvatures from the θ-IS warmup.

## Caveats

The papers were found by web search on 2026-11-20, and not all of them were read in
full: most details come from their abstracts and summaries. The EnCodec balancer was
checked in the audiocraft and `flashy` code. A search for backward-only normalization
of hidden-layer gradients with a running second moment did not find an exact match,
which does not rule one out.

## References

1. A. Défossez, J. Copet, G. Synnaeve, Y. Adi. *High Fidelity Neural Audio
   Compression*. arXiv 2210.13438, 2022 (TMLR 2023).
   <https://arxiv.org/abs/2210.13438>. Balancer code:
   <https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/losses/balancer.py>
2. A. Cabana, L. F. Lago-Fernández. *Backward Gradient Normalization in Deep Neural
   Networks*. arXiv 2106.09475, 2021. <https://arxiv.org/abs/2106.09475>
3. O. Sener, V. Koltun. *Multi-Task Learning as Multi-Objective Optimization*.
   NeurIPS 2018. <https://arxiv.org/abs/1810.04650>
4. E. Yang et al. *AdaTask: A Task-Aware Adaptive Learning Rate Approach to
   Multi-Task Learning*. AAAI 2023. <https://arxiv.org/abs/2211.15055>
5. Z. Chen, V. Badrinarayanan, C.-Y. Lee, A. Rabinovich. *GradNorm: Gradient
   Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. ICML 2018.
   <https://arxiv.org/abs/1711.02257>
6. H. van Hasselt, A. Guez, M. Hessel, V. Mnih, D. Silver. *Learning Values Across
   Many Orders of Magnitude*. NeurIPS 2016. <https://arxiv.org/abs/1602.07714>
7. M. Hessel, H. Soyer, L. Espeholt, W. Czarnecki, S. Schmitt, H. van Hasselt.
   *Multi-Task Deep Reinforcement Learning with PopArt*. AAAI 2019.
   <https://arxiv.org/abs/1809.04474>
8. S. Wang, Y. Teng, P. Perdikaris. *Understanding and Mitigating Gradient
   Pathologies in Physics-Informed Neural Networks*. arXiv 2001.04536, 2020.
   <https://arxiv.org/abs/2001.04536>
9. B. Ginsburg et al. *Stochastic Gradient Methods with Layer-wise Adaptive Moments
   for Training of Deep Networks* (NovoGrad). arXiv 1905.11286, 2019.
   <https://arxiv.org/abs/1905.11286>
10. Y. Zhang et al. *Adam-mini: Use Fewer Learning Rates To Gain More*. ICLR 2025.
    <https://arxiv.org/abs/2406.16793>
11. Y. You, I. Gitman, B. Ginsburg. *Large Batch Training of Convolutional Networks*
    (LARS). arXiv 1708.03888, 2017. <https://arxiv.org/abs/1708.03888>
12. Y. You et al. *Large Batch Optimization for Deep Learning: Training BERT in 76
    Minutes* (LAMB). ICLR 2020. <https://arxiv.org/abs/1904.00962>
13. S. Becker, Y. LeCun. *Improving the Convergence of Back-Propagation Learning with
    Second Order Methods*. Proceedings of the 1988 Connectionist Models Summer
    School, pp. 29–37, 1989.
14. Y. LeCun, L. Bottou, G. B. Orr, K.-R. Müller. *Efficient BackProp*. In *Neural
    Networks: Tricks of the Trade*, 1998.
    <https://link.springer.com/doi/10.1007/978-3-642-35289-8_3>
15. Y. Ollivier. *Riemannian Metrics for Neural Networks I: Feedforward Networks*.
    arXiv 1303.0818, 2013. <https://arxiv.org/abs/1303.0818>
16. J. Martens, R. Grosse. *Optimizing Neural Networks with Kronecker-factored
    Approximate Curvature*. ICML 2015. <https://arxiv.org/abs/1503.05671>
17. H. Han, V. Lostanlen, M. Lagrange. *Perceptual-Neural-Physical Sound Matching*.
    ICASSP 2023. <https://arxiv.org/abs/2301.02886>
18. H. Han, V. Lostanlen, M. Lagrange. *Learning to Solve Inverse Problems for
    Perceptual Sound Matching*. arXiv 2311.14213, 2023.
    <https://arxiv.org/abs/2311.14213>
