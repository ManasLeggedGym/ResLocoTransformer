"""
scripts/play.py — Run a trained policy in the environment.

Usage:
    # Use best checkpoint (default):
    python3 scripts/play.py --config configs/go2_attnres_mujoco.json

    # Use a specific epoch checkpoint:
    python3 scripts/play.py --config configs/go2_attnres_mujoco.json --checkpoint 1000

    # Pass a direct path to a .pth file:
    python3 scripts/play.py --config configs/go2_attnres_mujoco.json \
        --pf_path log/go2_attnres_mujoco/UnitreeMujocoGymEnv/0/model/model_pf_best.pth

    # Control number of episodes:
    python3 scripts/play.py --config configs/go2_attnres_mujoco.json --episodes 10
"""

import os
import sys
import glob
import re
import copy
import pickle
import argparse
import time
import os.path as osp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from envs.builder import get_env, ENV_DICT
import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import get_params


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_play_args():
    parser = argparse.ArgumentParser(description="Play a trained ResLocoTransformer policy")

    parser.add_argument("--config", type=str, required=True,
                        help="Config JSON used during training")
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Checkpoint to load: 'best', 'finish', or a numeric epoch "
                             "(e.g. 1000). Ignored when --pf_path is set.")
    parser.add_argument("--pf_path", type=str, default=None,
                        help="Direct path to a model_pf_*.pth file. "
                             "The matching _obs_normalizer_*.pkl and model_vf_*.pth "
                             "are inferred from the same directory.")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="Log root used during training (default: ./log)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used during training (selects the run directory)")
    parser.add_argument("--id", type=str, default=None,
                        help="Experiment ID override (same as used in train.py)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="Run on CPU even if a GPU is available")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Enable MuJoCo viewer rendering (requires a display)")
    parser.add_argument("--use_mlp", action="store_true", default=False,
                        help="Use MLP policy")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_pf_path(args, params):
    """Return (pf_path, model_dir) based on args."""
    if args.pf_path is not None:
        return osp.abspath(args.pf_path), osp.dirname(osp.abspath(args.pf_path))

    experiment_name = (
        os.path.split(os.path.splitext(args.config)[0])[-1]
        if args.id is None
        else args.id
    )
    model_dir = osp.join(
        args.log_dir, "model"
    )
    pf_path = osp.join(model_dir, f"model_pf_{args.checkpoint}.pth")
    return pf_path, model_dir


def _infer_ckpt_id(pf_path):
    """Extract the checkpoint ID (e.g. 'best', '1000') from a pf file path."""
    m = re.search(r"model_pf_(.+)\.pth$", osp.basename(pf_path))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def play(args):
    params = get_params(args.config)
    device = torch.device(f"cuda:{args.device}" if args.cuda else "cpu")

    # ------------------------------------------------------------------
    # Resolve checkpoint paths
    # ------------------------------------------------------------------
    pf_path, model_dir = _resolve_pf_path(args, params)
    if not osp.exists(pf_path):
        print(f"ERROR: checkpoint not found: {pf_path}")
        sys.exit(1)

    ckpt_id = _infer_ckpt_id(pf_path)
    norm_path = osp.join(model_dir, f"_obs_normalizer_{ckpt_id}.pkl") if ckpt_id else None

    print(f"Policy checkpoint : {pf_path}")
    if norm_path and osp.exists(norm_path):
        print(f"Obs normalizer    : {norm_path}")
    else:
        print("Obs normalizer    : not found (running without normalizer)")

    # ------------------------------------------------------------------
    # Build a single evaluation environment (with obs normalizer wrapper)
    # ------------------------------------------------------------------
    env_params = dict(params["env"])

    # Override rendering for play mode if requested
    if args.render:
        env_params["env_build"] = dict(env_params.get("env_build", {}))
        env_params["env_build"]["enable_rendering"] = True

    # get_env applies NormObsWithImg when obs_norm=True, matching training
    env = get_env(params["env_name"], env_params)

    # ------------------------------------------------------------------
    # Restore observation normalizer into the wrapper, then freeze it
    # ------------------------------------------------------------------
    if norm_path and osp.exists(norm_path):
        with open(norm_path, "rb") as f:
            loaded_norm = pickle.load(f)
        # Normalizers saved from VecEnv training may have mean/var shaped
        # (num_envs, obs_dim) due to a batched update bug — collapse to
        # (obs_dim,) by averaging across the env dimension.
        if hasattr(loaded_norm, "mean") and np.asarray(loaded_norm.mean).ndim > 1:
            loaded_norm.mean = np.asarray(loaded_norm.mean).mean(axis=0)
            loaded_norm.var  = np.asarray(loaded_norm.var).mean(axis=0)
        env._obs_normalizer = loaded_norm

    # eval() sets training=False so the normalizer stops accumulating stats
    env.eval()

    # ------------------------------------------------------------------
    # Peek at checkpoint to detect expected input dimension
    # ------------------------------------------------------------------
    ckpt_sd = torch.load(pf_path, map_location=device)

    # ------------------------------------------------------------------
    # Build policy network
    # ------------------------------------------------------------------
    params["net"]["base_type"] = networks.MLPBase

    _use_mlp = args.use_mlp or params.get("policy_type", "") == "mlp"
    obs_slice = None  # set below if checkpoint expects fewer dims than env

    if _use_mlp:
        # Simple MLP policy — no encoder, no transformer, state-only
        env_obs_dim = env.unwrapped.observation_space.shape[0]

        # Detect checkpoint input dim from first linear layer
        ckpt_obs_dim = env_obs_dim
        if "base.seq_fcs.0.weight" in ckpt_sd:
            ckpt_obs_dim = ckpt_sd["base.seq_fcs.0.weight"].shape[1]

        if ckpt_obs_dim != env_obs_dim:
            print(f"  NOTE: checkpoint expects {ckpt_obs_dim}-dim input, "
                  f"env provides {env_obs_dim}-dim. "
                  f"Will slice observations to first {ckpt_obs_dim} dims.")
            obs_slice = slice(0, ckpt_obs_dim)

        obs_dim = ckpt_obs_dim
        pf = policies.GaussianContPolicyBasicBias(
            input_shape=obs_dim,
            output_shape=env.action_space.shape[0],
            **params["net"],
            **params["policy"],
        )
        vf = networks.Net(
            input_shape=obs_dim,
            output_shape=1,
            **params["net"],
        )
    else:
        encoder = networks.LocoTransformerEncoder(
            in_channels=4,
            state_input_dim=env.unwrapped.state_dim,
            **params["encoder"],
        )
        encoder_vf = copy.deepcopy(encoder)

        _use_attn_res = "attn_res_heads" in params.get("net", {})

        if _use_attn_res:
            pf = policies.GaussianContPolicyLocoAttnResTransformer(
                encoder=encoder,
                state_input_shape=env.unwrapped.state_dim,
                visual_input_shape=(4, 64, 64),
                output_shape=env.action_space.shape[0],
                **params["net"],
                **params["policy"],
            )
            vf = networks.LocoAttnResTransformer(
                encoder=encoder_vf,
                state_input_shape=env.unwrapped.state_dim,
                visual_input_shape=(4, 64, 64),
                output_shape=1,
                **params["net"],
            )
        else:
            pf = policies.GaussianContPolicyLocoTransformer(
                encoder=encoder,
                state_input_shape=env.unwrapped.state_dim,
                visual_input_shape=(4, 64, 64),
                output_shape=env.action_space.shape[0],
                **params["net"],
                **params["policy"],
            )
            vf = networks.LocoTransformer(
                encoder=encoder_vf,
                state_input_shape=env.unwrapped.state_dim,
                visual_input_shape=(4, 64, 64),
                output_shape=1,
                **params["net"],
            )

    pf.load_state_dict(ckpt_sd)
    pf.to(device)
    pf.eval()
    print(f"Loaded policy ({pf.__class__.__name__})")

    # Real-time wall-clock duration of one env.step() (sim_dt * num_action_repeat)
    env_build = params["env"].get("env_build", {})
    step_wall_dt = (
        env_build.get("sim_dt", 0.001) * env_build.get("num_action_repeat", 10)
    )

    # ------------------------------------------------------------------
    # Rollout loop — NormObsWithImg handles normalisation transparently
    # ------------------------------------------------------------------
    episode_rewards = []
    episode_lengths = []

    for ep in range(args.episodes):
        obs, _ = env.reset()

        ep_reward = 0.0
        ep_len = 0
        done = False

        while not done:
            step_start = time.monotonic()

            ob = obs[obs_slice] if obs_slice is not None else obs
            ob_tensor = torch.tensor(ob, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = pf.explore(ob_tensor)
                act = out["action"].squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(act)
            done = bool(terminated) or bool(truncated)
            ep_reward += reward
            ep_len += 1

            # Pace to real-time so the viewer is watchable
            if args.render:
                elapsed = time.monotonic() - step_start
                remainder = step_wall_dt - elapsed
                if remainder > 0:
                    time.sleep(remainder)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        print(f"  Episode {ep + 1:3d} | reward: {ep_reward:8.3f} | length: {ep_len}")

    print()
    print(f"  Mean reward : {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Mean length : {np.mean(episode_lengths):.1f}")

    env.close()


if __name__ == "__main__":
    args = get_play_args()
    play(args)
