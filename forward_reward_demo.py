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

import click
import torch
import h5py
import numpy as np
import os

@click.command()
@click.option('--model-name', default="motivo-S-unbalanced", help='Name of the model to use')
@click.option('--camera', default="front", help='Camera view (front, top, side, etc.)')
@click.option('--unbalanced', is_flag=True, default=False, help='Use unbalanced physics')
@click.option('--render-width', default=640*2, help='Width of rendered video')
@click.option('--render-height', default=480*2, help='Height of rendered video')
@click.option('--video-dir', default="videos_reward_prompt", help='Directory to save videos')
@click.option('--num-steps', default=3000, type=int, help='Number of steps to run the simulation')
@click.option('--move-speed', default=2.0, type=float, help='Target movement speed for locomotion')
@click.option('--sample-size', default=100000, type=int, help='Number of samples from buffer')
def main(model_name, camera, unbalanced, render_width, render_height, video_dir, num_steps, move_speed, sample_size):
    """Run forward locomotion reward demonstration with a pretrained model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = FBcprModel.from_pretrained(model_name)
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
    video_dir_path = os.path.join(os.getcwd(), video_dir)
    os.makedirs(video_dir_path, exist_ok=True)
    
    # Create environment
    print(f"Creating environment with camera={camera}, unbalanced={unbalanced}")
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
            lambda env: RecordVideo(env, video_dir_path, episode_trigger=lambda x: True)
        ],
        state_init="Default",
    )
    
    # Download the inference buffer
    local_dir = f"metamotivo-S-1-datasets"
    dataset = "buffer_inference_500000.hdf5"
    print(f"Downloading buffer from facebook/metamotivo-S-1")
    buffer_path = hf_hub_download(
            repo_id=f"facebook/metamotivo-S-1",
            filename=f"data/{dataset}",
            repo_type="model",
            local_dir=local_dir,
        )
    
    # Load the buffer
    print("Loading buffer")
    hf = h5py.File(buffer_path, "r")
    data = {}
    for k, v in hf.items():
        data[k] = v[:]
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)
    del data
    
    # Set up the forward locomotion reward
    print(f"Setting up locomotion reward with move_speed={move_speed}")
    reward_fn = humenv_rewards.LocomotionReward(move_speed=move_speed)
    
    # Sample from buffer and compute rewards
    print(f"Sampling {sample_size} examples from buffer")
    batch = buffer.sample(sample_size)
    
    # Use multi-threaded relabeling for faster processing
    print("Computing rewards...")
    from metamotivo.wrappers.humenvbench import relabel
    rewards = relabel(
        env,
        qpos=batch["next_qpos"],
        qvel=batch["next_qvel"],
        action=batch["action"],
        reward_fn=reward_fn, 
        max_workers=16
    )
    print("Reward statistics:", np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards))
    
    # Infer the context `z` for the selected task
    print("Inferring reward context...")
    z = model.reward_wr_inference(
        next_obs=batch["next_observation"],
        reward=torch.tensor(rewards, device=model.cfg.device, dtype=torch.float32)
    )
    print(f"Context shape: {z.shape}")

    # Run the episode with forward locomotion reward
    print(f"Running simulation for {num_steps} steps...")
    observation, _ = env.reset()
    for i in range(num_steps):
        action = model.act(observation, z, mean=True)
        observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
        if terminated or truncated:
            observation, _ = env.reset()
    
    print(f"Forward reward video saved to {video_dir_path}")
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    main() 