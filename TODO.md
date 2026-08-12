# Project TODO
## Ablations
Ablations are meant to be experiments that justify each and every architectural choice so that one can conretely say why a certain choice was made. The current approach is a more "reward focused" approach where we have adapted a particular reward function and weights to ensure a natural gait. Config present in `configs/go2_natural_gait_tf.json`. Ablations are basically there to justify why this method and how it compares against other methods.

### Architecture Ablations

The arms map 1:1 onto the three Experiments bullets in `main`'s `TODO.md`, and
`scripts/run_ablations.py` prints one summary section per bullet.

**Bullet 1 — "Run a baseline training session (Tier 2/3)"**

- [x] `configs/ablation_baseline.json` — AttnRes, 2x `[4, 128]` layers,
      `attn_res_heads: 4`, depth camera on. Reference arm for both other bullets.

**Bullet 2 — "Experiment with different `attn_res_heads` in `LocoAttnResTransformer`"**

- [x] `configs/ablation_heads1.json` — `attn_res_heads: 1`
- [x] `configs/ablation_heads2.json` — `attn_res_heads: 2`
- [x] `configs/ablation_heads8.json` — `attn_res_heads: 8`
- 4 heads is the baseline arm, reused rather than retrained. `net.attn_res_heads`
      is the *only* key that differs across these four configs.
- `AttnResLayer` asserts `dim % num_heads == 0`; `encoder.visual_dim` is 64, so
      1/2/4/8 are all valid and each has been checked with a forward pass.

**Bullet 3 — "Abalate the effect of depth camera by training a state-only policy"**

- [x] `configs/ablation_stateonly.json` — `policy_type: "mlp"`,
      `get_image: false`, compared against `ablation_baseline`.
- Caveat: this arm cannot isolate the camera alone. `scripts/train.py:148-171`
      builds the transformer path with a hard-coded `visual_input_shape=(4,64,64)`,
      so a state-only policy is only reachable via `policy_type: mlp`, which drops
      the encoder and transformer with it. The gap therefore measures
      "vision + attention stack" vs "state-only MLP", not the camera in isolation.
      Isolating the camera would need a state-only transformer path in `train.py`.

#### Can an existing checkpoint serve as the fixed baseline? — No.

Checked every `params.json` under `log*/` against `configs/ablation_baseline.json`:

- **Architecture is compatible.** `log_gait/natural_gait_tf-summer/.../model_pf_best.pth`
  loads into a freshly built `GaussianContPolicyLocoAttnResTransformer` under
  `strict=True` (79 tensors, `attn_res` layers present, `state_dim=84`,
  `act_dim=12`). The "Architecture Incompatible" note on `main` refers to older
  pre-AttnRes checkpoints (e.g. `ckpt_850.zip`), not these.
- **But every arch-matching run was trained on `scene_flat.xml`,** while the
  ablations run `scene_rough.xml`. Different terrain is a different task, so
  reusing one as the reference would confound every arm.
- **And none converged.** Eval epochs logged: `natural_gait_tf-summer` 38
  (reward still climbing steeply, 1797 and rising, stopped ~epoch 380/2000),
  `log/go2_natural_gait_tf` 6, `log/ablation_attnres` 1.

So the baseline has to be trained fresh on rough terrain as `ablation_baseline` —
which is exactly what bullet 1 is for. Run it first; the other arms are only
interpretable against it.

All arms run on `robot/go2/scene_rough.xml` (the natural-gait reference config
`configs/go2_natural_gait_tf.json` uses `scene_flat.xml`) and share its reward
weights, `use_*` flags, env settings and PPO hyperparameters. `num_epochs` is
500 across all arms — reduced locally from the 2000 that is committed at HEAD;
keep it uniform across arms whichever value is used.

`run_ablations.py` verifies the one-variable property at startup and exits
non-zero if an arm differs from the baseline on any key outside its whitelist.

#### Environment / hardware constraints

- Run with the **`resloco` conda env** (`conda run -n resloco python3 ...`). Base
  python has no `gymnasium`; `run_ablations.py` now preflights `sys.executable`
  and exits with instructions rather than failing per-arm.
- **Only GPU 0 (RTX 3070, 8 GB) is usable.** GPU 1 is a GTX 1080 (sm_61) and the
  installed torch 2.11+cu130 ships sm_75+ kernels only — a plain matmul on it
  raises `no kernel image is available for execution on the device`. Ablations
  cannot be parallelised across the two GPUs on this box; they run sequentially
  on GPU 0, or GPU 1 needs a torch build with sm_61 kernels.
- **One arm at a time on GPU 0.** A single arm at `--vec_env_nums 16` settles
  around 6.3 GB of the 7.64 GB usable; a second concurrent arm OOMs during
  startup (measured, not estimated). At `--vec_env_nums 8` an arm uses ~4.6 GB —
  still too much for two, and 2.9x slower per epoch (208 s vs 72 s), so 16 is
  strictly better given arms cannot overlap anyway.
- **Budget.** 72 s/epoch x 500 epochs = ~10 h per arm-seed. The full matrix
  (5 arms x 3 seeds) is ~150 h ≈ 6.3 days of wall clock. Decide before launching
  the rest: fewer seeds, fewer epochs, or a bigger box.

Run them with:

```
python3 scripts/run_ablations.py --seeds 0 1 2   # 5 arms x 3 seeds, grouped summary
python3 scripts/run_ablations.py --filter heads  # bullet 2 only
```

`configs/extra_visual_pool.json` (mean-pooled tokens, no attention) is kept
outside the `ablation_*` glob because no bullet on `main` asks for it. Rename it
back to `configs/ablation_visual_pool.json` to fold it into the sweep again.

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
- [ ] `log_ablation/ablation_attnres/` is an orphaned run dir from the old config
      names (no `log.csv`, never reached an eval epoch). Delete it before the
      real sweep so it is not mistaken for a result.

### Audit of the seed-0 baseline run (2026-08-13, during `log_ablation/baseline.out`)

Findings from auditing the live baseline run. None were acted on mid-sweep except
the logging fix, since changing the reward or hyperparameters now would make the
remaining arms uncomparable to the baseline already in flight.

- [x] `Running_Training_Average_Rewards` logged as NaN every eval epoch.
      `torchrl/collector/on_policy.py` only appended to `train_rews` on `dones`,
      never on the `max_episode_frames` path. With `max_episode_frames: 999`
      against `horizon: 1000`, episodes essentially always end by time limit, so
      the deque stayed empty and `np.mean([])` produced the NaN that tensorboardX
      warned about. Fixed in `on_policy.py` and the same latent bug in
      `torchrl/collector/base.py` (`VecCollector`, off-policy path, which also
      never reset `train_rew` on time-limit resets). Logging-only: gradients,
      checkpoints and eval are unaffected, so the seed-0 baseline stays valid.
- [ ] **`use_feet_gait: true` is a no-op.** `r_feet_gait` is computed
      (`envs/mujoco_env.py:757`) and logged (`:795`) but never added to `total` --
      every other gated term has an `if self.use_X: total += r_X` line at
      `:762-775` and this one does not. Verified against the log at epoch 220:
      logged total 2.02581, sum of all terms 2.78470, sum excluding feet_gait
      2.02580. Also `_compute_feet_gait_reward` scales by `gait_phase_weight`
      (`:615`) -- no `feet_gait_weight` exists -- and measures nearly the same
      trot-phase agreement as `r_gait_phase`, which is why the two logged series
      match to 3 decimals across all epochs. Decide after the sweep: either wire
      it in with its own weight and rerun all arms, or drop the flag. Do not
      change it mid-sweep.
- [ ] Eval collapsed at epochs 110-130 (`Running_Average_Rewards` 1788 -> 41.8 ->
      recovered) while training `reward/total` only dipped 1.89 -> 1.37 per step,
      which predicts a ~1370 return. A ~30x train/eval gap. Only structural
      difference is deterministic `pf.eval_act` vs stochastic `explore`; the obs
      normalizer is copied correctly. Investigate if it recurs.
- [ ] Model selection is noise-dominated. `best` is a running max over ~50 eval
      points on a metric that swings 41 <-> 2050, which systematically favours the
      noisiest arm. `run_ablations.py` now reports a `Tail` column (mean over the
      final 10% of evals) and aggregates on it instead of on the last point, and
      warns when only one seed is present. Run the sweep with >=3 seeds.
- [ ] Both optimizers are permanently grad-clip-saturated: `clip_grad_norm_(..., 0.5)`
      at `torchrl/algo/on_policy/ppo.py:73` and `:118`, against observed pre-clip
      norms of 1300-2200 (vf) and ~5 (pf). `vf_loss` climbed 5 -> 250 as returns
      grew, i.e. the critic never tracks. With `opt_epochs: 10` on a stale
      `target_pf` this explains `ratio/max` reaching 76810 at epoch 210. Shared
      across arms so the comparison is fair, but it caps absolute performance.
- [ ] `diag/base_height` sits at 0.259 against `target_height: 0.29` for 220+
      epochs and `reward/height` never improves off -0.042, despite
      `height_weight: 30` being the largest weight in the config. Either the
      target is unreachable in the learned gait or the height is read from a
      different frame than the target assumes.
- [ ] `_is_fallen` triggers only below 0.15 m CoM and `alive_reward` is 0.0, so
      termination never fires -- `eval_traj_length` is exactly 1000.0 at every
      eval. The ablation never exercises falling, and this is the upstream reason
      the train-reward deque was empty.
