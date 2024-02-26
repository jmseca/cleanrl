# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

print("Import 1")

import gymnasium as gym
import numpy as np
print("Import 2")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
print("Import 3")
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
print("Import 4")
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
print("Import 5")


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    capture_video: bool = False
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""

    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    models_dir: str = "../runs/newRuns/"
    """the directory which stores the models we want to evaluate"""
    choose_model_in_runtime: bool = True
    """whether to print the available models from the 'models_dir' dir
    so user can specify the name of the model"""
    default_name: str = ""
    """default model id, if we do not choose model during runtime"""
    evaluate_all: bool = False
    """if toggled, evaluate all models in the `models_dir`"""
    
    # Model specific
    frame_stack: int = 4
    """the number of frames to stack"""
    image_size: int = 84
    """the size of the preprocessed frame (height and width)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (args.image_size, args.image_size))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, args.frame_stack)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        #hidden_input = 3136 if args.image_size == 84 else 9216
        hidden_input = 1024
        last_hidden = 128
        
        self.network = nn.Sequential(
            nn.Conv2d(args.frame_stack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_input, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    
    if args.evaluate_all:
        run_names = os.listdir(args.models_dir)
    else:

        # Do Not Modify - jmseca
        if args.choose_model_in_runtime:
            print(os.listdir(f"{args.models_dir}"))
            run_names = [input("Please, Choose the model to evaluate:\n> ")]
        else:
            run_names = [args.default_name]
    
    for run_name in run_names:
        print(f"Evaluating {run_name}")
        
        args.frame_stack = 2
        args.image_size = 64
        #if "Random06" in run_name or "Stack" in run_name:
        #    print("LESS STACK")
        #    args.frame_stack = 2
        #if "Img" in run_name or "BiggerImage" in run_name or "Random05" in run_name:
        #    print("BIGGER IMAGE")
        #    args.image_size = 128
            
        
        writer = SummaryWriter(f"{args.models_dir}{run_name}/eval")

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        
        q_network = QNetwork(envs).to(device)
            
        model_name = list(filter(lambda x : ".cleanrl_model" in x, os.listdir(f"{args.models_dir}{run_name}")))[0] 
        print(f"\n{args.models_dir}{run_name}/{model_name}\n")
        q_network.load_state_dict(torch.load(f"{args.models_dir}{run_name}/{model_name}"))
        

        model_path = f'{args.models_dir}{run_name}/{model_name}'
        from cleanrl_utils.evals.dqn_eval import evaluate
        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        envs.close()
        writer.close()
