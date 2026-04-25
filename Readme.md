# ResLocoTransformer

Standalone separation of the **LocoAttnResTransformer** integrated with **MuJoCo** for end-to-end RL training.

### 1. Environment Setup
Unless you want to be a snail, use `mamba` (a faster `conda` alternative) to manage your base environment, You might have to install mamba from miniforge-conda.

```bash
# Create a new environment
mamba create -n resloco python=3.10
mamba activate resloco
```

### 2. Package Management (using uv)
Use `uv` for lightning-fast dependency installation.

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install core dependencies
uv pip install mujoco>=3.0.0 torch>=2.0.0 gymnasium numpy
```

---

## Primary Task: Reward Tuning

The current focus is on systematic overhaul of the reward function to improve locomotion stability and gait naturalness.

### Relevant Files
- **Reward Logic**: [`envs/mujoco_env.py`](envs/mujoco_env.py) (See `_compute_reward`)
- **Tuning TODO**: [`.venv/reward_improvement_todo.md`](.venv/reward_improvement_todo.md)
- **Configuration**: [`configs/go2_attnres_mujoco.json`](configs/go2_attnres_mujoco.json) (Adjust weights here)

### Tuning Strategy
Following the [Reward Systematic Overhaul Plan](.venv/reward_improvement_todo.md), the workflow involves:
1.  **Baseline Restoration**: Return to a known working state (velocity tracking + fall penalty).
2.  **Incremental Build**: Add one parameter per run (Smoothness -> Energy -> Body Regulation).
3.  **Evaluation**: Monitor individual terms in Tensorboard/WandB to ensure the robot maintains tracking while improving gait quality.

---

## 🛠️ Modular Structure
- `models/`: LocoTransformer and AttnRes architectures.
- `envs/`: MuJoCo-specific Gym environments and sensor processing.
- `torchrl/`: RL framework (PPO, Buffer, Collector).
- `scripts/`: Training and evaluation entry points.
