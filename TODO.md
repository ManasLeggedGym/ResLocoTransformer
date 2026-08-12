# Project TODO
## Ablations
Ablations are meant to be experiments that justify each and every architectural choice so that one can conretely say why a certain choice was made. The current approach is a more "reward focused" approach where we have adapted a particular reward function and weights to ensure a natural gait. Config present in `configs/go2_natural_gait_tf.json`. Ablations are basically there to justify why this method and how it compares against other methods.

### Architecture Ablations

**Goal:** decide whether visual tokens and attention residuals (AttnRes) are worth their extra compute on rough terrain.

- **MLP:** state-only baseline; no visual terrain features.
- **Visual pool:** uses the visual encoder, then averages its tokens; no attention layers.
- **AttnRes:** uses the same visual encoder plus attention-residual layers. This tests whether attention itself adds value beyond visual features.

Status — configs are in place and each arm has been checked to build and run a forward pass:

- [x] `configs/ablation_mlp.json` — `policy_type: "mlp"`, `get_image: false` (222k policy params)
- [x] `configs/ablation_visual_pool.json` — `transformer_params: []`, `max_pool: false` → mean-pooled tokens, no attention (288k)
- [x] `configs/ablation_attnres.json` — 2x `[4, 128]` layers, `attn_res_heads: 4` (422k)

All three run on `robot/go2/scene_rough.xml` and are byte-identical to
`configs/go2_natural_gait_tf.json` in reward weights, `use_*` flags, env
settings, PPO hyperparameters and epoch budget (2000) — only the architecture
differs, so any gap between arms is attributable to the architecture.

Run them with:

```
python3 scripts/run_ablations.py --seeds 0 1 2   # 3 arms x 3 seeds, then a summary table
```

Things to keep in mind:
- [x] Logging should be nicely written and done to prevent loss of progress.
      Every eval epoch appends to `log.csv` (file reopened and closed per write,
      so an interrupted run keeps everything up to that epoch) and to
      TensorBoard; all `reward/*` and `diag/*` terms are logged per epoch.
- [x] Checkpoints need to be saved at regular intervals.
      `save_interval: 50` plus a `best` snapshot on every eval improvement and a
      `finish` snapshot; `scripts/run_ablations.py --resume` picks up from the
      latest checkpoint.
- [x] Do not be fooled by just "high rewards" or low loss values - the final gait
      must be practical without the Quadruped dragging across the floor or
      something. `run_ablations.py` and `compare_experiments.py` report
      converged `reward/foot_slip`, `reward/foot_clearance`,
      `reward/feet_air_time` and `diag/base_height` next to reward, and print a
      reminder to replay the `best` checkpoint of each arm with `scripts/play.py`
      before calling a winner.

- [ ] Remaining: actually run the sweep on a GPU box and write the results into
      `report.tex`.
