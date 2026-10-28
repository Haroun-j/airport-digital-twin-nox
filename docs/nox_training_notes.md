# NOx training notes

What happened between the first NOx run and the final one. I keep it here because
the final numbers only make sense with this history.

## Run 1 (5 April 2026): collapse to the background

- Validation accuracy 0.9997, macro-F1 0.1666, R2 about 0, relative error about 1.
  The network was predicting near-zero everywhere and the metrics went flat after a
  few epochs. Uniform random crops and picking `best.pt` on the plain validation loss
  both reward that solution.
- The run also crashed once in validation: `.cpu().numpy()` on bf16 tensors. Fixed by
  casting to float32 first; resume from `interrupted_latest.pt` was added at the same
  time.
- Validation was only done on random crops, so the confusion matrix depended on which
  crop was drawn. Replaced by a full-volume test evaluation with a tiled fallback when
  the volume does not fit in memory.

## Run 2 (6 April): plume-aware patch, still collapsed

- Crop 128 -> 192, zero-particle scenarios removed from the training split, extra
  loss terms on active voxels, best checkpoint selected on plume RMSE instead of the
  global loss.
- Plume RMSE around 222 000 and F1 0.1666 from epoch 1 to 5, no movement.

## Run 3 (6 April, the one in `results/nox`)

Changes compared with run 2:

- crop 224, batch 8;
- background voxel weight 0.05, active (> 10 ug/m3) weight 24, high (> 50 ug/m3)
  weight 96, `lambda_plume_focus` 1.0, `lambda_active_mae` 0.15;
- 85 % of the training crops centred on the target plume, 10 % on the emission
  sources, 5 % random;
- a fixed set of 16 full-volume validation scenarios evaluated every epoch; the
  checkpoint is selected on their plume RMSE;
- horizontal flips now also flip the sign of the corresponding wind component
  (before that the augmentation produced physically inconsistent samples);
- output head `Softplus` with the bias initialised at -3 instead of `ReLU`.

Selection plume RMSE: 816 059 (epoch 1), 150 694 (2), 134 712 (12), 134 094 (163,
best). Macro-F1 on the same subset: 0.166 -> 0.217. Slow, but real, and training
stayed stable for the 200 epochs (about 4.5 min each on an H100).

## Post-run issues

- The interactive Plotly cell needed `nbformat >= 4.2`; the flythrough video cell used
  `fig.canvas.tostring_rgb`, which recent Matplotlib removed (`buffer_rgba` instead).
- The showcase cells were picking `scenario_00004`, a scenario with no active
  particle, so the 3D view and the video looked empty. They now prefer a non-zero test
  scenario and use a percentile-based colour scale.

## What I take from it

- Accuracy and weighted F1 say nothing on this problem; look at macro-F1, plume RMSE
  and per-scenario medians.
- Removing the zero-particle scenarios from training made the model invent plumes in
  the 45 zero-particle test scenarios, and those dominate every mean metric. Next time:
  keep a controlled share of them in training, or gate the output on the emission
  field.
