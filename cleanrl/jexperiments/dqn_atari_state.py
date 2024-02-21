# Here I will create "states"
# I will parse the images first, only Framestack = 2, and then I will create a state from the two images
# I will then use this state to train the DQN

# Failed idea. Did not work :(((. Very low reward
# Maybe try this later, but not with DQN. With a different model
    
    
import matplotlib.pyplot as plt
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


# Johnny State Constants

#lv = line+value
blue_brick_lv = (87,92,85)
green_brick_lv = (81,86,124)
yellow_brick_lv = (75,80,148)
orange1_brick_lv = (69,74,131)
orange2_brick_lv = (63,68,129)
red_brick_lv = (57,62,110)

bar_line : int = 190
"""In which horizontal line the bar is located, in each frame"""
before_wall_left : int = 8
"""What pixel column on the left is next to the vertical wall"""
before_wall_right : int = 151
"""What pixel column on the right is next to the vertical wall"""
bar_middle_pixel : int = 10
"""The pixel width that is the middle of the bar"""
bar_pixel_value : int = 110
"""The grey scale pixel value of the bar"""

ball_start_line : int = 32
"""The first line where the ball might appear"""
ball_end_line : int = 195
"""The last line where the ball might appear"""
ball_pixel_value : int = 110
"""The grey scale pixel value of the ball"""


# TODO
# APAGAR ISTO
speed_xv = []
speed_yv = []
ball_posxv = []
ball_posyv = []
bar_posv = []


bricks_lv = [blue_brick_lv, green_brick_lv, yellow_brick_lv, orange1_brick_lv, orange2_brick_lv, red_brick_lv]

# Johnny State Functions

def get_bricks_column_value(frame, col_number):
    out_arr = [0,0,0,0,0,0]
    row_pixels_to_check = (9 + 8*col_number, 14 + 8*col_number)
    for i, brick in enumerate(bricks_lv):
        line1 = brick[0]+1
        line2 = brick[1]-1
        value = brick[2]
        if frame[line1][row_pixels_to_check[0]] == value and frame[line1][row_pixels_to_check[1]] == value and frame[line2][row_pixels_to_check[0]] == value and frame[line2][row_pixels_to_check[1]] == value:
            out_arr[i] = 1
    return sum(map(lambda x: x[1]*2**x[0], enumerate(out_arr)))


def get_bar_horizontal_pos(frame):
    line = frame[bar_line][before_wall_left:(before_wall_right+1)]
    start_pixel = 0
    end_pixel = 0
    found = False
    first_empty = line[0] != bar_pixel_value
    for i, pixel in enumerate(line):
        if not(found):
            if pixel == bar_pixel_value:
                start_pixel = i
                found = True
        else:
            if pixel != bar_pixel_value:
                end_pixel = i
                break
    if first_empty:
        return start_pixel + before_wall_left + bar_middle_pixel
    else:
        return end_pixel + before_wall_left - bar_middle_pixel
    
    
def get_ball_pos(frame):
    """
    Returns (row, column)
    """
    ball_frame = frame[ball_start_line:ball_end_line]
    for n, line in enumerate(ball_frame):
        for i in range(before_wall_left, (before_wall_right+1)):
            if line[i] == ball_pixel_value:
                
                if frame[ball_start_line+n-1][i] != ball_pixel_value and\
                    line[i+3] != ball_pixel_value and line[i-3] != ball_pixel_value and\
                    frame[ball_start_line+n+1][i+1] == ball_pixel_value:
                    
                    return (ball_start_line+n, i)
                
    # print("Ball not found - Will save image")
    # Will assume that ball is gone

    return (195,80)


def get_ball_speed(frame1, frame2):
    ball1 = get_ball_pos(frame1)
    ball2 = get_ball_pos(frame2)
    return (ball2[0] - ball1[0], ball2[1] - ball1[1])
    
    
        


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    
    # Retraining
    retrain: bool = False
    """whether to retrain a trained model or not"""
    retrain_default: bool = True
    """whether to retrain the latest model with the env-id chosen or not"""
    retrain_run_name: str = "Laranjas e Bananas"
    """the name of the model to retrain, if it is not the default"""
    
    # Model Name
    default_name: bool = False
    """whether to use the default name for the model or not"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.2
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.20
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 20000
    """timestep to start learning"""
    train_frequency: int = 8
    """the frequency of training"""
    
    #Model Consts
    normalize_net_inputs: bool = True
    """whether to normalize the inputs of the NN or not"""


def make_pre_env(env_id, seed, idx, capture_video, run_name):
    def pre_thunk():
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
        #env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return pre_thunk

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
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 2)

        env.action_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(23, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, env.single_action_space.n),
        )
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.double()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.double()

    def forward(self, obs):        
        bar_pos = get_bar_horizontal_pos(obs[0][1])
        ball_pos = get_ball_pos(obs[0][1])
        ball_speed = get_ball_speed(obs[0][0], obs[0][1])
        bricks = [get_bricks_column_value(obs[0][1], i) for i in range(18)]
        
        
        if args.normalize_net_inputs: 
            norm_bar_pos = max(bar_pos - 6,0) / 148 # 6 -> minimum bar position (I guess), 154  -> maximum bar position (154-6=148)
            norm_ball_pos = [(ball_pos[0] - 32) / 163, (ball_pos[1]  - 32) / 163]
            
            if ball_speed[0]>10 or ball_speed[1]>10 or ball_speed[0]<-10 or ball_speed[1]<-10:
                # Ball badly recognized or game restart
                norm_ball_speed = [1,1]
            else:
                # Assuming 10px/frame is the maximum accepted speed
                norm_ball_speed = [(ball_speed[0] + 10) / 10, (ball_speed[1] + 10) / 10]
            
            norm_ball_speed = [ball_speed[0] / 4, ball_speed[1] / 4]
            norm_bricks = [brick / 63 for brick in bricks]
            
            x = np.array([norm_bar_pos, norm_ball_pos[0], norm_ball_pos[1],
                          norm_ball_speed[0], norm_ball_speed[1]] + norm_bricks)
            
        else:            
            x = np.array([bar_pos, ball_pos[0], ball_pos[1], ball_speed[0], ball_speed[1]] + bricks)
        
        return self.network(torch.tensor(x, dtype=torch.double))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):#
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    
    if args.default_name:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = input("Please, insert the name of the run/model:\n> ")
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv(
    #    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #)
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    envs = gym.vector.SyncVectorEnv(
        [make_pre_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    
    q_network = QNetwork(envs).to(device)
    if args.retrain:
        
        model_name = f'{args.exp_name}.cleanrl_model'
        if not(args.retrain_default):
            q_network.load_state_dict(torch.load(f"runs/{args.retrain_run_name}/{model_name}"))
        else:
            # I have to pick the second last model, because the last one is the current one
            retrain_run_name = sorted(filter(lambda x: f'{args.env_id}__{args.exp_name}' in x, os.listdir("runs/")), key=lambda x: int(x.split('_')[-1]))[-2]
            print(f"runs/{retrain_run_name}/{model_name}")
            q_network.load_state_dict(torch.load(f"runs/{retrain_run_name}/{model_name}"))
    
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.tensor(obs, dtype=torch.double).to(device))
            actions = np.array([torch.argmax(q_values).cpu().numpy()])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max()
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # SPS = Steps per Second (I believe)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
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

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
