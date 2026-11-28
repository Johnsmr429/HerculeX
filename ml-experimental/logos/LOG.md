# LOGOS log

## v0 · 26–27 November 2026

v0 was the first trained LOGOS model. It ran as one fused CUDA megakernel on a Colab L4 with bf16 weights, batch two
slabs of `64×128×128`, and 1.8 training steps per second. The run is stored under
`data/outputs/runs/model/2026-11-26-v0`.

The held-out splits were not trained on and remain held out. `MODEL_ALL_LABELS` was added after this run and is off
by default.

### What ran

| Stage | Data | Schedule | Steps |
| --- | --- | --- | --- |
| Pre-training | Unlabelled CT from 31 scans: 86,817 training slabs and 2,807 held out | Reconstruction only; AdamW, peak learning rate `1e-3`, cosine to `1e-5` | 50,000 |
| Mid-training | 1,152 instance-cube slabs, 2,584 winding-point slabs, 480 fibre slabs, 86,817 unlabelled slabs and 12,141 Kaggle slabs | From pre step 50,000; shares `0.35,0.25,0.1,0.15,0.15`; peak learning rate `3e-4` | Stopped at 36,600 of 50,000 |

Mid-training was stopped by hand. The last checkpoint is step 30,000 at learning rate `1.07e-4`; steps 30,001–36,600
were not saved. Validation at step 35,000 is the only surviving evidence from that interval.

### Architecture that produced v0

- 10,725,541 parameters; bf16 model weights with fp32 master weights and optimiser moments.
- A four-stage 3D convolution stem, width-256 tokens, 128 context latents and a four-block recurrent-depth trunk.
- Each block used window attention, two-way latent exchange, bidirectional z recurrence and a SwiGLU MLP.
- Training drew 4–12 loops and backpropagated through the last six; evaluation used eight loops.
- There was no adaptive halting and no tiled whole-scroll inference.
- One cooperative launch ran about 760 operations for the complete training step.

### Data as used

| Group | Training / held out | Labels |
| --- | --- | --- |
| PHercParis4 instance cubes | 1,152 training slabs plus held-out cubes | Sheet, contact, same-sheet and next-turn pairs. |
| PHercParis4 winding points | 2,584 training slabs; sets hashed one in ten for holdout | Same-wrap, relative and absolute winding. |
| Dataset001 fibres | 480 training slabs; one cube in five held out | Sparse positive sheet labels at confidence 0.7. |
| Unlabelled CT | 86,817 / 2,807 slabs | Reconstruction only. |
| Kaggle surface volumes | 12,141 / 1,331 slabs | Sheet labels. |
| PHercParis4 and PHerc0332 lines and fibres | Evaluation only | Sheet recall and winding drift. |

The unlabelled data excluded the HerculeX holdout-A scrolls, PHerc0332, PHerc0139 z 5000–7500 and PHercParis4
`20230205180739` z 7680–7936.

### Training changes made during v0

- The first pre-training run developed an all-negative decoder level. Channel LayerNorm with ε 0.1 was added after
  each stem and decoder convolution, plus LayerNorm on the trunk output.
- A mixed-source test made the confidence loss about 16 and clipped every step. Its target changed from token loss to
  `log(1 + token loss)`.
- The Kaggle shard was once overwritten by an empty shard and then rebuilt identically.

### Pre-training validation

Held-out reconstruction MSE uses per-slab standardised CT, where predicting the mean is about 1.0.

| Step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recon MSE | 0.328 | 0.305 | 0.294 | 0.285 | 0.275 | 0.268 | 0.261 | 0.256 | 0.254 | 0.253 |

### Mid-training validation

The fixed validation set contained 480 held-out slabs.

| Step | Sheet Dice, Scroll 1 cubes | Next-turn accuracy, Scroll 1 cubes | Several-wrap accuracy, PHercParis4 | Kaggle surface Dice | Sheet recall, PHerc0332 lines |
| --- | --- | --- | --- | --- | --- |
| 5k | 0.914 | 0.311 | 0.000 | 0.289 | 0.980 |
| 10k | 0.929 | 0.543 | 0.000 | 0.296 | 0.985 |
| 15k | 0.926 | 0.629 | 0.000 | 0.292 | 0.995 |
| 20k | 0.936 | 0.661 | 0.011 | 0.299 | 0.997 |
| 25k | 0.936 | 0.749 | 0.069 | 0.317 | 0.988 |
| 30k, saved | 0.943 | 0.772 | 0.184 | 0.317 | 0.985 |
| 35k, not saved | 0.947 | 0.790 | 0.430 | 0.346 | 0.957 |

Step 30,000 on all 7,448 held-out records:

| Data | Result |
| --- | --- |
| Scroll 1 instance cubes | Sheet Dice 0.942; next-turn accuracy 0.771 on 230,678 pairs; contact F1 0. |
| Fibre cubes | Sheet Dice 0.999 on labelled fibre voxels. |
| PHercParis4 winding sets | Several-wrap accuracy 0.208 on 272,510 pairs; sheet recall 0.985. |
| PHercParis4 evaluation fibres | Sheet recall 0.943. |
| PHerc0332 slice lines | Sheet recall 0.982. |
| PHerc0332 manual fibres | Sheet recall 0.887. |
| Kaggle volumes | Surface Dice 0.311. |

Median winding drift along one traced sheet (computed offline from dumped predictions; `model eval` still logs
`winding_jumps_per_1000_voxels` and `lines_without_jumps` in `eval.json`) was 0.08 turns on PHercParis4 winding sets,
0.24 on PHercParis4 evaluation fibres, 0.53 on PHerc0332 manual fibres and 1.86 on PHerc0332 slice lines. Jump counts
are not reported in the summary because a field close to a half-integer can jump under noise without comparable
geometric drift.

### Invalid prize-scroll renders

The v0 prize-scroll renders were produced while `dump` still masked 75% of unlabelled CT, as reconstruction training
does. They do not measure the model and must not be used for visual judgement. `dump` was changed after v0 to run
unmasked.

### Checks and performance

- `model test` passed on the Colab L4 before training and later on an RTX 3090.
- The deterministic two-step repeat in `model model` passed.
- After LayerNorm was added, the bf16 central-difference check reported 55–58 of 228 tensors outside its 15%
  tolerance on both GPUs. This is the v0 numerical baseline; a new failure or meaningfully moved value is a
  regression, but the count alone is not a correctness gate.
- `model bench` measured about 1.8 steps per second at batch two on the L4, around 10 algorithmic TFLOP/s.

### Run files

| Path below `2026-11-26-v0/` | Contents |
| --- | --- |
| `pre/step-*.ckpt`, `mid/step-*.ckpt` | Weights, optimiser moments and hard-mining state. |
| `pre/`, `mid/` `train.jsonl`, `val.jsonl`, `gpu.csv` | Training, validation and GPU logs. |
| `mid/eval.json` | Step 30,000 evaluated on every held-out record. |
| `mid/dump/` | Held-out CT, sheet probability, winding, labels and points. |
| `mid/prize-boxes-masked/` | Invalid masked prize-scroll boxes, retained only as run history. |
| `renders/training-curves.png` | Training and validation curves. |
| `renders/held-out/sourceN.png` | Held-out slab renders by source. |

Render the retained records from the HerculeX root with:

```bash
run=data/outputs/runs/model/2026-11-26-v0
python3 -I ml-experimental/logos/curves.py "$run" "$run/renders/training-curves.png"
python3 -I ml-experimental/logos/slabs.py "$run/mid/dump" "$run/renders/held-out"
```

### Post-v0 code changes

- `dump` now reads slabs unmasked.
- Training now checkpoints at every 5,000-step validation, at completion and on `SIGTERM` or `SIGINT`.
- `--all-labels` and `MODEL_ALL_LABELS` were added; they remain off.
- The renamed `model test` passes and `model model` reproduces the v0 numerical baseline on an RTX 3090.

### Remaining model work

- Implement tiled whole-scroll inference and write the field directory expected by `unroll-trace --field`.
- Balance the contact loss if the contact head is kept.
- Continue or post-train from the best saved checkpoint only after choosing the data and holdout policy.

