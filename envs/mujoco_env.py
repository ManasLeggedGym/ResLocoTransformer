"""
UnitreeMujocoGymEnv — gym environment for the Unitree Go2 quadruped
using MuJoCo 3.x as the physics backend.

Observation format (flat, matches LocoTransformerEncoder convention):
  [state (84-dim) | depth_frames_flat (16384-dim)]

  State layout (84-dim):
    [0 : 12]  — IMU history   : [roll, pitch, gyro_x, gyro_y] × 3 steps
    [12 : 48] — Joint angles  : 12 joints × 3 steps
    [48 : 84] — Last actions  : 12 joints × 3 steps

  Depth (16384-dim):
    4 stacked 64×64 single-channel depth frames, clipped to [0.3, 10] m,
    log-transformed via sqrt(log(d + 1)), then optionally z-scored.

``observation_space`` reports only the state part (84-dim) so that
``LocoTransformerEncoder`` receives ``state_input_dim = env.observation_space.shape[0]``.
The ``image_channels`` property returns 4.

Action space: 12-dim joint position targets (one per actuator), bounded
around the Go2 home posture.  Wrapped by ``NormAct`` in the training
pipeline, so the policy sees a [-1, 1] action space.
"""

import os
import math
import collections

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


from mujoco import viewer
import wandb

wandb.init(project="go2-rl-base", name="base-rewards")
    

NUM_MOTORS   = 12
STATE_DIM    = 84           # 12 (IMU×3hist) + 36 (joints×3hist) + 36 (action×3hist)
NUM_HIST     = 3            # proprioceptive history length

IMU_DIM      = 4            # [roll, pitch, gyro_x, gyro_y]

NUM_DEPTH_FRAMES = 4        # stacked depth frames  (== in_channels for encoder)
IMG_H = IMG_W = 64
IMAGE_DIM = NUM_DEPTH_FRAMES * IMG_H * IMG_W   # 16 384

# ---------------------------------------------------------------------------
# Sensor layout in mj_data.sensordata for Go2
#   Sensors defined in go2.xml (in declaration order):
#     jointpos   × 12  → dim 1 each → [0 : 12]
#     jointvel   × 12  → dim 1 each → [12 : 24]
#     jointfrc   × 12  → dim 1 each → [24 : 36]
#     imu_quat   (framequat)        → [36 : 40]  (w, x, y, z)
#     imu_gyro   (gyro)             → [40 : 43]  (gx, gy, gz)
#     imu_acc    (accelerometer)    → [43 : 46]
#     frame_pos  (framepos)         → [46 : 49]
#     frame_vel  (framelinvel)      → [49 : 52]  world-frame linear velocity
# ---------------------------------------------------------------------------

SD_JOINT_POS = slice(0,  12)
SD_JOINT_VEL = slice(12, 24)
SD_IMU_QUAT  = slice(36, 40)   # (w, x, y, z)
SD_IMU_GYRO  = slice(40, 43)   # (gx, gy, gz)
SD_FRAME_VEL = slice(49, 52)   # (vx, vy, vz) in world frame


# ---------------------------------------------------------------------------
# Go2 home posture — matches the <key name="home"> keyframe in go2.xml
#   Actuator order: FR_hip, FR_thigh, FR_calf,
#                   FL_hip, FL_thigh, FL_calf,
#                   RR_hip, RR_thigh, RR_calf,
#                   RL_hip, RL_thigh, RL_calf
# ---------------------------------------------------------------------------

_PER_LEG_HOME = [0.0, 0.9, -1.8]
INIT_JOINT_ANGLES = np.array(_PER_LEG_HOME * 4, dtype=np.float32)

# Symmetric position bounds around home (hip±0.3, thigh/calf±0.4)
_PER_LEG_CLIP  = [0.3, 0.4, 0.4]
_ACTION_CLIP   = np.array(_PER_LEG_CLIP * 4, dtype=np.float32)
ACTION_LB      = INIT_JOINT_ANGLES - _ACTION_CLIP
ACTION_UB      = INIT_JOINT_ANGLES + _ACTION_CLIP

# ---------------------------------------------------------------------------
# Depth normalisation — matches original env (locomotion_gym_env_with_rich_info)
# ---------------------------------------------------------------------------

DEPTH_FAR  = 10.0    # metres — clip far range
DEPTH_NEAR = 0.3     # metres — clip near range (closer than robot body)
DEPTH_MEAN = 1.25    # z-score mean after log-transform
DEPTH_STD  = 0.425   # z-score std


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rpy(quat: np.ndarray):
    """Convert MuJoCo's (w, x, y, z) quaternion to (roll, pitch, yaw)."""
    w, x, y, z = quat
    roll  = math.atan2(2.0 * (w * x + y * z),
                       1.0 - 2.0 * (x * x + y * y))
    sin_p = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_p)))
    yaw   = math.atan2(2.0 * (w * z + x * y),
                       1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _patch_go2_xml(scene_xml_path: str):
    """
    Load go2/scene.xml and the included go2.xml, then:
      1. Fix ``meshdir`` to an absolute path so ``from_xml_string`` can
         resolve mesh assets on disk.
      2. Inject a forward-facing depth camera into the ``base_link`` body.

    Returns
    -------
    scene_xml : str
        The scene XML string (contains ``<include file="go2.xml"/>``).
    assets : dict
        Mapping ``{"go2.xml": patched_go2_xml_str}`` passed to
        ``mujoco.MjModel.from_xml_string``.
    """
    go2_dir      = os.path.dirname(os.path.abspath(scene_xml_path))
    go2_xml_path = os.path.join(go2_dir, "go2.xml")

    with open(scene_xml_path) as fh:
        scene_xml = fh.read()
    with open(go2_xml_path) as fh:
        go2_xml = fh.read()

    # ------------------------------------------------------------------
    # 1. Make meshdir absolute so MuJoCo can find mesh files from a
    #    string-loaded model (no implicit base directory).
    # ------------------------------------------------------------------
    assets_abs = os.path.join(go2_dir, "assets")
    go2_xml = go2_xml.replace(
        'meshdir="assets"',
        f'meshdir="{assets_abs}"',
    )

    # ------------------------------------------------------------------
    # 2. Inject a forward-facing depth camera into base_link.
    #
    #    Camera is placed 26 cm forward and 3 cm above the body origin,
    #    looking along body +x (forward).
    #
    #    xyaxes="0 -1 0  0 0 1":
    #      camera-x (image right) = body -y
    #      camera-y (image up)    = body +z
    #      ⟹ camera -z (viewing)  = body +x  (forward)
    # ------------------------------------------------------------------
    _CAMERA_XML = (
        '\n      <camera name="depth_cam"'
        ' pos="0.26 0 0.03"'
        ' xyaxes="0 -1 0 0 0 1"'
        ' fovy="60"/>'
    )
    go2_xml = go2_xml.replace(
        '<site name="imu" pos="-0.02557 0 0.04232" />',
        '<site name="imu" pos="-0.02557 0 0.04232" />' + _CAMERA_XML,
    )

    return scene_xml, {"go2.xml": go2_xml.encode("utf-8")}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class UnitreeMujocoGymEnv(gym.Env):
    """Gym environment for the Unitree Go2 using MuJoCo 3.x.

    Parameters
    ----------
    scene_path : str
        Absolute path to the MuJoCo scene XML
        (e.g. ``…/unitree_robots/go2/scene.xml``).
    Kp : float
        PD proportional gain applied to joint position error.
    Kd : float
        PD derivative gain applied to joint velocity error.
    num_action_repeat : int
        Number of ``mj_step`` calls per ``env.step()``.
    sim_dt : float
        MuJoCo simulation time-step in seconds.
    target_vel : float
        Desired forward velocity (m/s) — used for reward clipping reference.
    alive_reward : float
        Positive constant added to the reward each step.
    fall_reward : float
        Negative penalty added when the episode terminates by falling.
    max_episode_steps : int
        Episode horizon (steps, each comprising ``num_action_repeat`` sim steps).
    depth_norm : bool
        If ``True``, z-score the depth frames using precomputed statistics.
    enable_rendering : bool
        If ``True``, open a passive MuJoCo viewer window.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        scene_path: str,
        Kp: float = 40.0,
        Kd: float = 0.5,
        num_action_repeat: int = 10,
        sim_dt: float = 0.001,
        target_vel: float = 0.6,
        target_angular_vel: float = 0.5,
        alive_reward: float = 0.1,
        fall_reward: float = -10.0,
        max_episode_steps: int = 1000,
        depth_norm: bool = True,
        enable_rendering: bool = False,
        # --- Phase 2 reward shaping weights ---
        vel_tracking_weight: float = 2.0,
        vel_sigma: float = 0.25,        # σ² in exp(-||Δv||²/σ²)
        angular_vel_weight: float = 2.0,
        angular_vel_sigma: float = 0.25,
        torque_weight: float = 0.002,
        action_rate_weight: float = 0.01,
        orientation_weight: float = 0.1,
        height_weight: float = 1.0,
        target_height: float = 0.28,    # metres — nominal Go2 CoM height
        lat_yaw_weight: float = 0.1,    # penalise lateral vel + yaw rate
        # --- per-term enable flags (False = computed but not added to total) ---
        use_torque: bool = True,
        use_action_rate: bool = True,
        use_orientation: bool = True,
        use_height: bool = True,
        use_lat_yaw: bool = True,
    ):
        super().__init__()

        self.scene_path       = scene_path
        self.Kp               = Kp
        self.Kd               = Kd
        self.num_action_repeat = num_action_repeat
        self.sim_dt           = sim_dt
        self.target_vel       = target_vel
        self.target_angular_vel = target_angular_vel
        self.alive_reward     = alive_reward
        self.fall_reward      = fall_reward
        self.max_episode_steps = max_episode_steps
        self.depth_norm       = depth_norm
        self.enable_rendering = enable_rendering
        # reward shaping
        self.vel_tracking_weight = vel_tracking_weight
        self.vel_sigma           = vel_sigma
        self.angular_vel_weight = angular_vel_weight
        self.angular_vel_sigma = angular_vel_sigma
        self.torque_weight       = torque_weight
        self.action_rate_weight  = action_rate_weight
        self.orientation_weight  = orientation_weight
        self.height_weight       = height_weight
        self.target_height       = target_height
        self.lat_yaw_weight      = lat_yaw_weight
        self.use_torque          = use_torque
        self.use_action_rate     = use_action_rate
        self.use_orientation     = use_orientation
        self.use_height          = use_height
        self.use_lat_yaw         = use_lat_yaw

        # ------------------------------------------------------------------
        # Build MuJoCo model with the patched go2.xml (camera + abs meshdir)
        # ------------------------------------------------------------------
        scene_xml, assets = _patch_go2_xml(scene_path)
        self.mj_model = mujoco.MjModel.from_xml_string(scene_xml, assets)
        self.mj_model.opt.timestep = sim_dt
        self.mj_data  = mujoco.MjData(self.mj_model)

        assert self.mj_model.nu == NUM_MOTORS, (
            f"Expected {NUM_MOTORS} actuators, got {self.mj_model.nu}"
        )

        # Precompute frequently used body / camera IDs
        self._cam_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam"
        )
        self._base_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )

        # ------------------------------------------------------------------
        # Offscreen depth renderer (64 × 64, body-attached camera)
        # ------------------------------------------------------------------
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        try:
            self._depth_renderer = mujoco.Renderer(
                self.mj_model, height=IMG_H, width=IMG_W
            )
        except Exception as e:
            print(f"Warning: Renderer creation failed with default backend. Error: {e}")
            print("Attempting to force EGL backend...")
            os.environ["MUJOCO_GL"] = "egl"
            self._depth_renderer = mujoco.Renderer(
                self.mj_model, height=IMG_H, width=IMG_W
            )
        self._depth_renderer.enable_depth_rendering()

        # ------------------------------------------------------------------
        # Gym spaces
        #
        # observation_space reports only the 84-dim STATE part so that
        # LocoTransformerEncoder receives state_input_dim correctly.
        # The actual observation returned by step() / reset() is
        # [state(84) | depth_flat(16384)].
        # ------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=ACTION_LB.astype(np.float32),
            high=ACTION_UB.astype(np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Proprioceptive history deques
        # ------------------------------------------------------------------
        self._imu_hist    = collections.deque(maxlen=NUM_HIST)
        self._joint_hist  = collections.deque(maxlen=NUM_HIST)
        self._action_hist = collections.deque(maxlen=NUM_HIST)

        # Depth frame ring-buffer (4 frames × (1, 64, 64))
        self._depth_buf = collections.deque(maxlen=NUM_DEPTH_FRAMES)

        self._step_count  = 0
        self._last_action = INIT_JOINT_ANGLES.copy()

        # Optional passive viewer (only for local visualisation)
        self._viewer = None
        if enable_rendering:
            self._viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data
            )

    # ------------------------------------------------------------------
    # Property required by ppo_locotransformer.py
    # ------------------------------------------------------------------

    @property
    def image_channels(self) -> int:
        """Number of depth image channels (stacked frames = 4)."""
        return NUM_DEPTH_FRAMES

    # ------------------------------------------------------------------
    # Internal observation helpers
    # ------------------------------------------------------------------

    def _get_imu_obs(self) -> np.ndarray:
        """Return 4-dim IMU reading: [roll, pitch, gyro_x, gyro_y]."""
        quat = self.mj_data.sensordata[SD_IMU_QUAT]
        roll, pitch, _ = _quat_wxyz_to_rpy(quat)
        gyro = self.mj_data.sensordata[SD_IMU_GYRO]
        return np.array([roll, pitch, gyro[0], gyro[1]], dtype=np.float32)

    def _get_joint_obs(self) -> np.ndarray:
        """Return current 12-dim joint position vector."""
        return self.mj_data.sensordata[SD_JOINT_POS].astype(np.float32)

    def _get_depth_frame(self) -> np.ndarray:
        """
        Render one 64×64 depth frame from the body-attached camera.

        Processing pipeline (matches original pybullet env):
          1. Clip to [DEPTH_NEAR, DEPTH_FAR] metres.
          2. Apply sqrt(log(d + 1)) log-compression.

        Returns
        -------
        np.ndarray, shape (1, 64, 64), dtype float32
        """
        self._depth_renderer.update_scene(self.mj_data, camera=self._cam_id)
        raw = self._depth_renderer.render().copy()   # (64, 64), metres
        raw = np.clip(raw, DEPTH_NEAR, DEPTH_FAR)
        processed = np.sqrt(np.log(raw + 1.0)).astype(np.float32)
        return processed[np.newaxis]                 # (1, 64, 64)

    def _reset_buffers(self):
        """Pre-fill history deques with the current (post-reset) state."""
        imu0   = self._get_imu_obs()
        joint0 = self._get_joint_obs()
        act0   = np.zeros(NUM_MOTORS, dtype=np.float32)

        self._imu_hist.clear()
        self._joint_hist.clear()
        self._action_hist.clear()
        for _ in range(NUM_HIST):
            self._imu_hist.append(imu0.copy())
            self._joint_hist.append(joint0.copy())
            self._action_hist.append(act0.copy())

        depth0 = self._get_depth_frame()
        self._depth_buf.clear()
        for _ in range(NUM_DEPTH_FRAMES):
            self._depth_buf.append(depth0.copy())

    def _build_observation(self) -> np.ndarray:
        """
        Concatenate history buffers into the flat observation vector:
          [imu_3hist(12) | joint_3hist(36) | action_3hist(36) | depth_flat(16384)]
        """
        # State part (84-dim)
        imu_hist   = np.concatenate(list(self._imu_hist),    axis=0)   # (12,)
        joint_hist = np.concatenate(list(self._joint_hist),  axis=0)   # (36,)
        act_hist   = np.concatenate(list(self._action_hist), axis=0)   # (36,)
        state = np.concatenate([imu_hist, joint_hist, act_hist])        # (84,)

        # Depth part (16384-dim)
        depth_stack = np.concatenate(list(self._depth_buf), axis=0)    # (4, 64, 64)
        depth_flat  = depth_stack.reshape(-1).astype(np.float32)       # (16384,)
        if self.depth_norm:
            depth_flat = (depth_flat - DEPTH_MEAN) / DEPTH_STD

        return np.concatenate([state, depth_flat]).astype(np.float32)

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _apply_pd_action(self, q_target: np.ndarray):
        """
        Apply joint position targets via PD law for ``num_action_repeat``
        simulation steps.

        Control law (matches unitree_sdk2py_bridge LowCmdHandler):
            ctrl[i] = kp * (q_target[i] - q_actual[i])
                    + kd * (0           - dq_actual[i])
        """
        for _ in range(self.num_action_repeat):
            q_actual  = self.mj_data.sensordata[SD_JOINT_POS]
            dq_actual = self.mj_data.sensordata[SD_JOINT_VEL]
            for i in range(NUM_MOTORS):
                self.mj_data.ctrl[i] = (
                    self.Kp * (q_target[i] - q_actual[i])
                    + self.Kd * (-dq_actual[i])
                )
            mujoco.mj_step(self.mj_model, self.mj_data)

    def _is_fallen(self) -> bool:
        """True when the base body's CoM drops below 0.20 m."""
        return float(self.mj_data.xpos[self._base_body_id, 2]) < 0.20

    def _compute_reward(self):
        """
        Shaped reward (Phase 2).  Returns (total_reward, terms_dict).

        Terms
        -----
        r_vel         Smooth exponential forward-velocity tracking.
        r_alive       Constant survival bonus.
        r_torque      Penalty for large actuator torques (prevents overheating).
        r_action_rate Penalty for jerky actions (Transformer stability).
        r_orientation Penalty for non-zero roll / pitch.
        r_height      Penalty for CoM height deviation from target.
        r_lat_yaw     Penalty for lateral velocity and yaw rate when target is
                      strictly forward.
        """
        sd = self.mj_data.sensordata

        # --- velocity tracking ---
        vx  = float(sd[SD_FRAME_VEL][0])
        print(f"Linear Velocity: {vx}")
        dv  = (vx - self.target_vel) ** 2
        r_vel = self.vel_tracking_weight * math.exp(-dv / self.vel_sigma)

        # --- angular velocity tracking ---
        wx = float(sd[SD_IMU_GYRO][2])
        print(f"Angular Velocity: {wx}")
        d_angular = (wx - self.target_angular_vel) ** 2
        r_ang_vel = self.angular_vel_weight * math.exp(-d_angular / self.angular_vel_sigma)

        # --- survival ---
        r_alive = self.alive_reward

        # --- torque penalty ---
        torques   = self.mj_data.ctrl  # (12,) current control signals
        r_torque  = -self.torque_weight * float(np.sum(torques ** 2))


        #! NOTE -  WE WILL BE REMOVING THE OTHER REWARD TERMS FOR NOW - AND CHECKING WITH EACH 

        # --- action rate penalty (requires ≥2 steps of history) ---
        if len(self._action_hist) >= 2:
            a_prev     = self._action_hist[-2]
            a_curr     = self._action_hist[-1]
            r_action_rate = -self.action_rate_weight * float(
                np.sum((a_curr - a_prev) ** 2)
            )
        else:
            r_action_rate = 0.0

        # --- orientation penalty ---
        quat         = sd[SD_IMU_QUAT]
        roll, pitch, _ = _quat_wxyz_to_rpy(quat)
        r_orientation = -self.orientation_weight * (roll ** 2 + pitch ** 2)

        # --- body height penalty ---
        z         = float(self.mj_data.xpos[self._base_body_id, 2])
        r_height  = -self.height_weight * (z - self.target_height) ** 2

        # --- lateral velocity + yaw rate penalty ---
        vy       = float(sd[SD_FRAME_VEL][1])
        yaw_rate = float(sd[SD_IMU_GYRO][2])
        r_lat_yaw = -self.lat_yaw_weight * (vy ** 2 + yaw_rate ** 2)

        # Base terms always active; optional terms gated by enable flags.
        # All terms are still computed and logged regardless of the flag so
        # we can observe what a disabled term would have contributed.
        total = r_vel + r_alive + r_ang_vel
        # if self.use_torque:      total += r_torque
        # if self.use_action_rate: total += r_action_rate
        # if self.use_orientation: total += r_orientation
        # if self.use_height:      total += r_height
        # if self.use_lat_yaw:     total += r_lat_yaw

        active_count = (3) #+ self.use_torque + self.use_action_rate
                        #+ self.use_orientation + self.use_height + self.use_lat_yaw)

        terms = {
            "reward/forward_vel":   r_vel,
            "reward/alive":         r_alive,
            "reward/ang_vel":       r_ang_vel,
            # "reward/torque":        r_torque,
            # "reward/action_rate":   r_action_rate,
            # "reward/orientation":   r_orientation,
            # "reward/height":        r_height,
            # "reward/lat_yaw":       r_lat_yaw,
            "reward/total":         total,
            "reward/active_terms":  active_count,
            "diag/vx":              vx,
            "diag/base_height":     z,
            "diag/roll":            roll,
            "diag/pitch":           pitch,
        }
        return total, terms

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # qpos layout for a free-body robot:
        #   [tx, ty, tz, qw, qx, qy, qz, j0 … j11]
        self.mj_data.qpos[7:] = INIT_JOINT_ANGLES
        self.mj_data.qvel[:]  = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._last_action = INIT_JOINT_ANGLES.copy()
        self._step_count  = 0

        self._reset_buffers()

        if self._viewer is not None:
            self._viewer.sync()
        fallen  = self._is_fallen()
        timeout = self._step_count >= self.max_episode_steps
        info = {"fallen": fallen, "timeout": timeout}
        return self._build_observation(), info

    def step(self, action: np.ndarray):
        action = np.clip(action, ACTION_LB, ACTION_UB).astype(np.float32)

        self._apply_pd_action(action)

        # Update history buffers
        self._imu_hist.append(self._get_imu_obs())
        self._joint_hist.append(self._get_joint_obs())
        self._action_hist.append(action.copy())
        self._depth_buf.append(self._get_depth_frame())

        self._last_action  = action.copy()
        self._step_count  += 1

        reward, reward_terms = self._compute_reward()
        fallen  = self._is_fallen()
        timeout = self._step_count >= self.max_episode_steps

        if fallen:
            reward += self.fall_reward

        if self._viewer is not None:
            self._viewer.sync()

        info = {"fallen": fallen, "timeout": timeout, **reward_terms}
        wandb.log(info)
        terminated = fallen
        truncated  = timeout
        return self._build_observation(), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        """Render a 480×640 third-person RGB view (for recording / debugging)."""
        renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
        renderer.update_scene(self.mj_data)
        img = renderer.render().copy()
        renderer.close()
        return img

    def close(self):
        self._depth_renderer.close()
        if self._viewer is not None:
            self._viewer.close()
