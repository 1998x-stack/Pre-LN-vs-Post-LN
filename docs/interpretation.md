# Interpretation and experimental controls

## Causal estimand

Warmup interacts with norm placement. The four-arm factorial design supports estimation of the difference in warmup effects:

`(loss_post_warm - loss_post) - (loss_pre_warm - loss_pre)`.

Use comparable budgets and report confidence intervals over paired random seeds. Note that warmup modifies the total integral of the learning-rate schedule, so repeat with matched cumulative LR exposure if attributing an improvement to warmup shape rather than the amount of early optimization.

## What to verify before reading any plot

- Same parameter initialization: reset torch seed independently for each model construction and hold architecture except placement constant.
- Same training batches: per-seed CPU generator used independently of model construction.
- Same validation examples: separate seed/split. Keep both data selection and number of validation windows identical across arms.
- Same optimizer/weight decay and global batch size.
- Norm parameter gradients contribute to `grad_global_preclip` but are not included in the per-layer *sublayer* gradient heatmap.
- Record `status` and `steps_completed`; a partial run must never be mistaken for an improving full-length trajectory.

## Optional research upgrades

- Sweep **warmup fractions** 0%, 1%, 5%, 10% at fixed depth and LR.
- Add initialization-only gradient probes before any optimizer step.
- Add layer ablation (`val_loss_with_layer_removed - baseline_val_loss`) as a functional estimate of layer contribution.
- Add fixed-token rather than only fixed-step budgets when comparing alternate batch/context settings.
- Perform optimizer variants (AdamW vs SGD), LayerNorm vs RMSNorm and residual scaling in **separate** ablations, not confounded with Pre/Post placement.
- Train with multiple seeds and plot paired confidence intervals; testing-only charts do not establish generalization.
