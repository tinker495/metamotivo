#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary libraries
from packaging.version import Version
from metamotivo.fb_cpr.huggingface import FBcprModel
from huggingface_hub import hf_hub_download
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation, RecordEpisodeStatistics, RecordVideo
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards

import torch
import h5py
import numpy as np
import os

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
unbalanced = True
camera = "front"
render_width = 640 * 2
render_height = 480 * 2

# Download the model
model = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
model.to(device)

# Configure observation transformation based on gymnasium version
if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
        )
else:
    transform_obs_wrapper = lambda env: TransformObservation(
            env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device), env.observation_space
        )

# Create video directory
video_dir = os.path.join(os.getcwd(), "videos_reward_prompt")
os.makedirs(video_dir, exist_ok=True)

# Create environment
env, _ = make_humenv(
    num_envs=1,
    unbalanced=unbalanced,
    camera=camera,
    render_width=render_width,
    render_height=render_height,
    wrappers=[
        FlattenObservation,
        transform_obs_wrapper,
        RecordEpisodeStatistics,
        lambda env: RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    ],
    state_init="Default",
)
# Download the inference buffer
local_dir = "metamotivo-M-1-datasets"
dataset = "buffer_inference_500000.hdf5"
buffer_path = hf_hub_download(
        repo_id="facebook/metamotivo-M-1",
        filename=f"data/{dataset}",
        repo_type="model",
        local_dir=local_dir,
    )

# Load the buffer
hf = h5py.File(buffer_path, "r")
data = {}
for k, v in hf.items():
    data[k] = v[:]
buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)
del data

# Set up the forward locomotion reward (move ahead with speed 2.0)
reward_fn = humenv_rewards.LocomotionReward(move_speed=2.0)

# Sample from buffer and compute rewards
N = 100_000
batch = buffer.sample(N)

# Use multi-threaded relabeling for faster processing
from metamotivo.wrappers.humenvbench import relabel
rewards = relabel(
    env,
    qpos=batch["next_qpos"],
    qvel=batch["next_qvel"],
    action=batch["action"],
    reward_fn=reward_fn, 
    max_workers=8
)
print(rewards.ravel())

# We can now infer the context `z` for the selected task.
z = model.reward_wr_inference(
    next_obs=batch["next_observation"],
    reward=torch.tensor(rewards, device=model.cfg.device, dtype=torch.float32)
)
print(z.shape)

# Run the episode with forward locomotion reward
observation, _ = env.reset()
for i in range(3000):
    action = model.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
    if terminated or truncated:
        observation, _ = env.reset()

print(f"Forward reward video saved to {video_dir}")

# Cleanup
env.close() 