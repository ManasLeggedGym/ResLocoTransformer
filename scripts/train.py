"""
scripts/train.py — PPO training entry point for ResLocoTransformer.

Usage:
    python3 scripts/train.py --config configs/go2_attnres_mujoco.json
    python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume
    python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume --checkpoint 1000
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("MUJOCO_GL", "egl")
import copy
import glob
import re
import time
import sys
import pickle
import os.path as osp
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.builder import get_subprocvec_env, get_vec_env
import torch
import gymnasium as gym

from torchrl.collector.on_policy import VecOnPolicyCollector
from torchrl.algo import PPO
import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import Logger
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import get_params, get_args


args = get_args()
params = get_params(args.config)


def _find_latest_checkpoint_epoch(model_dir):
    """Return the highest numeric epoch found in model_dir, or None."""
    files = glob.glob(osp.join(model_dir, "model_pf_*.pth"))
    epochs = []
    for f in files:
        m = re.search(r"model_pf_(\d+)\.pth$", osp.basename(f))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def experiment(args):

    device = torch.device(
        "cuda:{}".format(args.device) if args.cuda else "cpu"
    )

    env = get_vec_env(
        params["env_name"],
        params["env"],
        args.vec_env_nums,
    )
    eval_env = get_vec_env(
        params["env_name"],
        params["env"],
        max(1, args.vec_env_nums),
    )

    if hasattr(env, "_obs_normalizer"):
        eval_env._obs_normalizer = env._obs_normalizer
    #possible error here
    # env.seed(seed = args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    buffer_param = params["replay_buffer"]

    experiment_name = (
        os.path.split(os.path.splitext(args.config)[0])[-1]
        if args.id is None
        else args.id
    )

    # ------------------------------------------------------------------
    # Resolve resume checkpoint before creating the logger so we know
    # whether to keep or wipe the existing work directory.
    # ------------------------------------------------------------------
    resume_epoch = 0
    if args.resume:
        model_dir = osp.join(
            args.log_dir, experiment_name, params["env_name"],
            str(args.seed), "model",
        )
        if args.checkpoint is not None:
            ckpt_epoch = int(args.checkpoint)
        else:
            ckpt_epoch = _find_latest_checkpoint_epoch(model_dir)

        if ckpt_epoch is not None:
            resume_epoch = ckpt_epoch + 1
            print(f"Resuming from checkpoint epoch {ckpt_epoch} "
                  f"(training will start at epoch {resume_epoch})")
        else:
            print("No checkpoint found — starting from scratch")

    logger = Logger(
        experiment_name, params["env_name"],
        args.seed, params, args.log_dir,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    params["general_setting"]["env"] = env

    replay_buffer = OnPolicyReplayBuffer(
        env_nums=args.vec_env_nums,
        max_replay_buffer_size=int(buffer_param["size"]),
        time_limit_filter=buffer_param["time_limit_filter"],
    )
    params["general_setting"]["replay_buffer"] = replay_buffer
    params["general_setting"]["logger"] = logger
    params["general_setting"]["device"] = device

    params["net"]["base_type"] = networks.MLPBase

    _use_mlp = params.get("policy_type", "") == "mlp"

    if _use_mlp:
        # Simple MLP policy — no encoder, no transformer, state-only
        obs_dim = env.unwrapped.observation_space.shape[0]
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

    print(pf)
    print(vf)

    params["general_setting"]["collector"] = VecOnPolicyCollector(
        vf, env=env, eval_env=eval_env, pf=pf,
        replay_buffer=replay_buffer, device=device,
        train_render=False,
        **params["collector"],
    )
    params["general_setting"]["save_dir"] = osp.join(
        logger.work_dir, "model"
    )

    # ------------------------------------------------------------------
    # Load checkpoint weights when resuming
    # ------------------------------------------------------------------
    if args.resume and resume_epoch > 0:
        ckpt_epoch = resume_epoch - 1
        model_dir = params["general_setting"]["save_dir"]
        pf_path = osp.join(model_dir, f"model_pf_{ckpt_epoch}.pth")
        vf_path = osp.join(model_dir, f"model_vf_{ckpt_epoch}.pth")
        norm_path = osp.join(model_dir, f"_obs_normalizer_{ckpt_epoch}.pkl")

        pf.load_state_dict(torch.load(pf_path, map_location=device))
        vf.load_state_dict(torch.load(vf_path, map_location=device))
        print(f"Loaded pf weights from {pf_path}")
        print(f"Loaded vf weights from {vf_path}")

        if osp.exists(norm_path) and hasattr(env, "_obs_normalizer"):
            with open(norm_path, "rb") as f:
                loaded_norm = pickle.load(f)
            env._obs_normalizer = loaded_norm
            eval_env._obs_normalizer = loaded_norm
            print(f"Loaded obs normalizer from {norm_path}")

    elif args.load_from is not None:
        load_dir = args.load_from
        tag = args.load_epoch if args.load_epoch else "best"
        pf_path = osp.join(load_dir, f"model_pf_{tag}.pth")
        vf_path = osp.join(load_dir, f"model_vf_{tag}.pth")
        norm_path = osp.join(load_dir, f"_obs_normalizer_{tag}.pkl")

        pf.load_state_dict(torch.load(pf_path, map_location=device))
        vf.load_state_dict(torch.load(vf_path, map_location=device))
        print(f"Loaded pretrained pf from {pf_path}")
        print(f"Loaded pretrained vf from {vf_path}")

        if osp.exists(norm_path) and hasattr(env, "_obs_normalizer"):
            with open(norm_path, "rb") as f:
                loaded_norm = pickle.load(f)
            env._obs_normalizer = loaded_norm
            eval_env._obs_normalizer = loaded_norm
            print(f"Loaded obs normalizer from {norm_path}")

    if args.freeze_backbone:
        frozen = []
        for name, param in pf.named_parameters():
            if name.startswith(("encoder.base.", "encoder.state_projector.",
                                "token_ln.", "attn_res_layers.")):
                param.requires_grad = False
                frozen.append(name)
        print(f"Froze {len(frozen)} policy params (transformer + state MLP)")
        trainable = sum(p.numel() for p in pf.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pf.parameters())
        print(f"Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)")

    if args.residual_policy:
        state_dim = env.unwrapped.state_dim
        action_dim = env.action_space.shape[0]
        pf = policies.ResidualPolicy(
            base_policy=pf, state_dim=state_dim, action_dim=action_dim,
        ).to(device)
        base_params = sum(p.numel() for p in pf.base_policy.parameters())
        res_params = sum(p.numel() for p in pf.residual_mlp.parameters())
        print(f"Residual policy: base={base_params} (frozen), "
              f"residual={res_params} (trainable), logstd=1")

    agent = PPO(
        pf=pf,
        vf=vf,
        **params["ppo"],
        **params["general_setting"],
        resume_epoch=resume_epoch,
    )
    agent.train()


if __name__ == "__main__":
    experiment(args)
