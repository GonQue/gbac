# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=8):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=96, height=96, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = cv2.resize(
        #     frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        # )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = WarpFrame(env, grayscale=args.grayscale)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 1)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def wrap_pytorch(env):
    return ImageToPyTorch(env)


class RewardShapingEnv(gym.Wrapper):
    """
    Environment wrapper for CarRacing 
    """


    def reset(self):
        obs = super(RewardShapingEnv, self).reset()
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        return obs

    def step(self, action):
        obs, reward, die, info = super(RewardShapingEnv, self).step(action)
        # don't penalize "die state"
        if die:
            reward += 100
        # green penalty
        if np.mean(obs[:, :, 1]) > 185.0:
            reward -= 0.05
        # if no reward recently, end the episode
        done = True if self.av_r(reward) <= -0.1 or die else False
        return obs, reward, done, info

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid
import imageio.v2 as iio

import argparse
import shutil
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, RecordVideo
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import plotly.express as px
from utils.ram_utils import get_num_layers, get_kernels_and_strides

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos['real_reward'] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--filename', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--add-comment', type=str, default="",
                        help='Add comment to the name')
    parser.add_argument('--gym-id', type=str, default="CarRacing-v0",
                        help='the id of the gym environment')
    parser.add_argument('--grayscale', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='toogles the grayscale of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='number of episodes to evaluate the model')
    parser.add_argument('--avg-episodes', type=int, default=100,
                        help='number of episodes to calculate the average episodic return')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, 
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--record-rate', type=int, default=100,
                        help="number of episodes between each saved video of gameplay (with glimpses) and each heatmap")

    # Algorithm specific arguments
    parser.add_argument('--resize', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help="weather to resize the game frame to 84x84 or keep the original size")
    parser.add_argument('--n-minibatch', type=int, default=32,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=2048,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=10,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    # Glimpse specific arguments
    parser.add_argument('--patch-size', type=int, default=0, 
                        help='size of extracted patch at highest res')
    parser.add_argument('--patch-h', type=int, default=12, 
                        help='height of extracted patch at highest res')
    parser.add_argument('--patch-w', type=int, default=12, 
                        help='width of extracted patch at highest res')
    parser.add_argument('--num-glimpses', type=int, default=1, 
                        help='number of glimpses to take in each frame')
    parser.add_argument('--glimpse-scale', type=int, default=1, 
                        help='scale of successive patches')
    parser.add_argument('--num-patches', type=int, default=1, 
                        help='# of downscaled patches per glimpse')

    args = parser.parse_args()
    if args.patch_size != 0:
        args.patch_h = args.patch_size
        args.patch_w = args.patch_size
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)
full_image_str = "_F" if not args.resize else ""
args.exp_name = f"{args.gym_id}{full_image_str}__{args.filename}__{args.num_patches}" 
if len(args.add_comment) != 0:
    args.exp_name += f"_{args.add_comment}"

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# TRY NOT TO MODIFY: setup the environment
full_image_str = "F" if not args.resize else ""	
experiment_name = f"{args.gym_id}__{args.filename}__{args.num_glimpses}g_{args.patch_h}x{args.patch_w}{full_image_str}_{args.glimpse_scale}s_{args.num_patches}p"  	
if len(args.add_comment) != 0:	
    experiment_name += f"_{args.add_comment}" 	
experiment_name += f"__{args.seed}__{int(time.time())}" 
if args.track:
    import wandb
    wandb.init(
        project=args.wandb_project_name, 
        entity=args.wandb_entity, 
        sync_tensorboard=True, 
        config=vars(args), 
        name=experiment_name, 
        monitor_gym=True, 
        save_code=True,
    )
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

def save_checkpoint(state, is_best):
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")

    ckpt_path = "ckpt/" + experiment_name + "_ckpt.pth.tar"
    torch.save(state, ckpt_path)
    # if args.track:
    #     wandb.save(ckpt_path)
    if is_best:
        best_path = "ckpt/" + experiment_name + "_model_best.pth.tar"
        shutil.copyfile(ckpt_path, best_path)

def make_env(gym_id, seed, idx, test=False):
    def thunk():
        env = gym.make(gym_id)
        env = ClipActionsWrapper(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = wrap_pytorch(
            wrap_deepmind(
                env,
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        )
        if args.capture_video:
            if idx == 0:
                videos_folder = f"videos/Test_{experiment_name}" if test else f"videos/{experiment_name}"
                env = RecordVideo(env, videos_folder, episode_trigger=lambda t: t % 1000 == 0)
        env = NormalizedEnv(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]), device)
# if args.prod_mode:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
#         device
#     )
assert isinstance(envs.action_space, Box), "only continuous action space is supported"

# ALGO LOGIC: initialize agent here:
def denormalize_coords(H, W, coords, g_H, g_W):
    """Convert coordinates in the range [-1, 1] to coordinates in the 
    range [0, T] where `T` is the size of the image.
    """
    denorm_coords = torch.zeros_like(coords).to(device)
    denorm_coords[:, 0] = (0.5 * ((coords[:, 0] + 1.0) * (H - 1))) - (g_H // 2)
    denorm_coords[:, 1] = (0.5 * ((coords[:, 1] + 1.0) * (W - 1))) - (g_W // 2)
    return denorm_coords.long()


def update_loc_heatmap(locs_heatmap, loc, args):
    denorm_coords = denormalize_coords(locs_heatmap.shape[0], locs_heatmap.shape[1], loc, args.patch_h, args.patch_w)
    for coords in denorm_coords:
        locs_heatmap[coords[0], coords[1]] += 1 
    return locs_heatmap

def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Retina:
    def __init__(self, num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor):
        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.num_patches = num_patches
        self.scaling_factor = scaling_factor

    def foveate(self, x, l, save_glimpses=False):
        phi = []
        B = x.shape[0]
        bboxes = torch.zeros((B, self.num_glimpses * self.num_patches, 4)) 
        counter = 0

        for j in range(self.num_glimpses):
            g_H = self.glimpse_h
            g_W = self.glimpse_w
            # extract k patches of increasing size
            for i in range(self.num_patches):
                patch, bboxes = self.extract_patch(x, l[:, j, :], g_H, g_W, bboxes, counter)
                # print("patch", patch[0].permute(1, 2, 0).shape)
                # fig = px.imshow(patch[0].squeeze(0).cpu().numpy())
                # fig.write_image(f"patch{i}.png")
                counter += 1
                phi.append(patch)
                g_H = int(self.scaling_factor * g_H)
                g_W = int(self.scaling_factor * g_W)
                # size += self.glimpse_size

            # resize the patches to squares of size g
            for i in range(1, len(phi)):
                k_h = phi[i].shape[-2] // self.glimpse_h
                k_w = phi[i].shape[-1] // self.glimpse_w
                phi[i] = F.avg_pool2d(phi[i], (k_h, k_w))
        
        
        # print("phi:", len(phi))
        # phi = torch.cat(phi, 1)
        # print("phi:", phi.shape)
        # print(phi[0].unsqueeze(1).shape)
        
        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        # phi = phi.view(phi.shape[0], -1)
        # print("phi:", phi.shape)

        if save_glimpses:
            images = wandb.Image(make_grid(phi[0].unsqueeze(1), pad_value=1))
            wandb.log({"test/test_glimpses": images})
        
        return phi, bboxes

    def extract_patch(self, x, l, glimpse_h, glimpse_w, bboxes, idx):
        B, C, H, W = x.shape

        # start = denormalize_coords(H, W, l) - (size // 2)
        start = denormalize_coords(H, W, l, glimpse_h, glimpse_w)
        end = torch.zeros_like(start).to(device)
        end[:, 0] = start[:, 0] + glimpse_h
        end[:, 1] = start[:, 1] + glimpse_w

        # pad with zeros
        # x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))

        # loop through minibatch and extract patches
        patch = []
        for i in range(B):
            start_x, start_y, end_x, end_y = self.keep_glimpse_inside_image(start[i], end[i], H, W)
            patch.append(x[i, :, start_x : end_x, start_y : end_y])
            bboxes[i, idx, :] = torch.tensor([start_y, start_x, end_y, end_x])
        
        return torch.stack(patch), bboxes

    def keep_glimpse_inside_image(self, start, end, H, W):
        start_x, start_y = start
        end_x, end_y = end

        if start_x < 0:
            end_x += abs(start_x)
            start_x = 0
        if start_y < 0:
            end_y += abs(start_y)
            start_y = 0
        if end_x > H:
            start_x -= end_x - H
            end_x = H    
        if end_y > W:
            start_y -= end_y - W
            end_y = W
        return start_x, start_y, end_x, end_y


class Agent(nn.Module):
    def __init__(self, num_channels, num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor, frames=1):
        super(Agent, self).__init__()
        
        self.num_glimpses = num_glimpses
        self.retina = Retina(num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor)
        num_layers = get_num_layers(glimpse_h)
        if num_layers == 1: 
            k1, s1 = get_kernels_and_strides(glimpse_h)
            final_input_h, final_input_w = conv2d_size_out(glimpse_h, k1, s1), conv2d_size_out(glimpse_w, k1, s1)
            print(f"latent_size: {final_input_h}x{final_input_w}")
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(num_channels * frames * num_glimpses * num_patches, 64, kernel_size=(k1, k1), stride=(s1, s1))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * final_input_h * final_input_w, 512)),
                nn.ReLU(),
            )
        elif num_layers == 2: 
            k1, s1, k2, s2 = get_kernels_and_strides(glimpse_h)
            first_conv_h, first_conv_w = conv2d_size_out(glimpse_h, k1, s1), conv2d_size_out(glimpse_w, k1, s1)
            final_input_h, final_input_w = conv2d_size_out(first_conv_h, k2, s2), conv2d_size_out(first_conv_w, k2, s2)
            print(f"1st_conv_size: {first_conv_h}x{first_conv_w}")
            print(f"latent_size: {final_input_h}x{final_input_w}")
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(num_channels * frames * num_glimpses * num_patches, 32, kernel_size=(k1, k1), stride=(s1, s1))),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=(k2, k2), stride=(s2, s2))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * final_input_h * final_input_w, 512)),
                nn.ReLU(),
            )
        else:
            k1, s1, k2, s2, k3, s3 = get_kernels_and_strides(glimpse_h)
            first_conv_h, first_conv_w = conv2d_size_out(glimpse_h, k1, s1), conv2d_size_out(glimpse_w, k1, s1)
            second_conv_h, second_conv_w = conv2d_size_out(first_conv_h, k2, s2), conv2d_size_out(first_conv_w, k2, s2)
            final_input_h, final_input_w = conv2d_size_out(second_conv_h, k3, s3), conv2d_size_out(second_conv_w, k3, s3)
            print(f"1st_conv_size: {first_conv_h}x{first_conv_w}")
            print(f"2nd_conv_size: {second_conv_h}x{second_conv_w}")
            print(f"latent_size: {final_input_h}x{final_input_w}")
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(num_channels * frames * num_glimpses * num_patches, 32, kernel_size=(k1, k1), stride=(s1, s1))),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=(k2, k2), stride=(s2, s2))),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=(k3, k3), stride=(s3, s3))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * final_input_h * final_input_w, 512)),
                nn.ReLU(),
            )

        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor_mean = layer_init(nn.Linear(128, np.prod(envs.action_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, prev_loc, done, save_glimpses=False):
        if args.grayscale:
            x = x / 255.0
        phi, bboxes = self.retina.foveate(x, prev_loc, save_glimpses)
        hidden = self.network(phi)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        # print("LSTM Batch size:", batch_size)
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state, bboxes

    def get_action(self, x, lstm_state, prev_loc, done, action=None, loc=None, save_glimpses=False):
        hidden, lstm_state, bboxes = self.get_states(x, lstm_state, prev_loc, done, save_glimpses=save_glimpses)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if loc is None:
            loc = torch.FloatTensor(x.shape[0], args.num_glimpses, 2).uniform_(-1, 1).to(device)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), loc, lstm_state, action_std, bboxes

    def get_value(self, x, lstm_state, prev_loc, done):
        hidden, _, _ = self.get_states(x, lstm_state, prev_loc, done)
        return self.critic(hidden)

agent = Agent(envs.observation_space.shape[0],
              args.num_glimpses,
              args.patch_h, 
              args.patch_w, 
              args.num_patches, 
              args.glimpse_scale,
        ).to(device)

pytorch_total_params = sum(p.numel() for p in agent.parameters())
print(f'Model parameters: {pytorch_total_params}')

optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
prev_locs = torch.zeros((args.num_steps, args.num_envs, args.num_glimpses) + (2,)).to(device)
locs = torch.zeros((args.num_steps, args.num_envs, args.num_glimpses) + (2,)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
best_episodic_return = float('-inf')
best_avg_return = float('-inf')
is_best = False
global_step = 0
total_episodes = 0
returns_queue = deque(maxlen=args.avg_episodes)
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
next_lstm_state = (
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
)  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
num_updates = args.total_timesteps // args.batch_size

if args.capture_video:
    video_name = f"videos/{experiment_name}/video_{total_episodes}.mp4"
    recorder = iio.get_writer(video_name, fps=10)
    locs_heatmap = torch.zeros(envs.observation_space.shape[1], envs.observation_space.shape[2])

for update in range(1, num_updates + 1):
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step], next_lstm_state, prev_locs[step], next_done).flatten()
            action, logproba, _, loc, next_lstm_state, action_std, bboxes = agent.get_action(obs[step], next_lstm_state, prev_locs[step], next_done)

        actions[step] = action
        logprobs[step] = logproba
        if step < args.num_steps - 1:
            prev_locs[step + 1] = loc
        locs[step] = loc

        if args.capture_video and total_episodes % args.record_rate == 0:
            locs_heatmap = update_loc_heatmap(locs_heatmap, locs[step][0], args)
            frame = next_obs[0].clone().to(torch.uint8)
            frame = draw_bounding_boxes(frame, bboxes[0, :], width=1, colors=(255, 0, 0))
            frame = frame.permute(1, 2, 0)
            recorder.append_data(frame.numpy())

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)

        is_best = False
        some_ep_finished = False
        for info in infos:
            if 'episode' in info.keys():
                some_ep_finished = True
                total_episodes += 1
                returns_queue.append(info["episode"]["r"])
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                for j in range(action_std.shape[-1]):
                    writer.add_scalar(f"std/std_{j}", action_std[0][j], global_step)
                    
                avg_return = sum(returns_queue) / args.avg_episodes

                if len(returns_queue) == args.avg_episodes and avg_return >= best_avg_return:
                    best_avg_return = avg_return
                    is_best = True
                        
                if info["episode"]["r"] >= best_episodic_return:
                    best_episodic_return = info["episode"]["r"]
                
                if args.capture_video and total_episodes % args.record_rate == 0:
                    recorder.close()
                    wandb.log({"video": wandb.Video(video_name)})
                    
                    fig = px.imshow(locs_heatmap)
                    wandb.log({"charts/loc_heatmap": fig})
                    locs_heatmap = torch.zeros(envs.observation_space.shape[1], envs.observation_space.shape[2])

                # the next episode will be captured (if statement needs to be after episodes_counter increment in order to prepare the recorder)
                if args.capture_video and total_episodes % args.record_rate == 0:
                    video_name = f"videos/{experiment_name}/video_{total_episodes}.mp4"
                    recorder = iio.get_writer(video_name, fps=10)

        if some_ep_finished:
            writer.add_scalar(f"charts/episodes", total_episodes, global_step)
            if len(returns_queue) == args.avg_episodes:
                writer.add_scalar(f"charts/avg_return_last_{args.avg_episodes}_eps", sum(returns_queue) / args.avg_episodes, global_step)

            save_checkpoint(
                    {   "global_step": global_step,
                        "model_state": agent.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "best_episodic_return": best_episodic_return,
                        "best_avg_return": best_avg_return,
                        "total_episodes": total_episodes,
                    }, is_best)
            is_best = False

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device), next_lstm_state, locs[step], next_done).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.action_space.shape)
    b_locs = locs.reshape((-1, args.num_glimpses) + (2,))
    b_prev_locs = prev_locs.reshape((-1, args.num_glimpses) + (2,))

    b_dones = dones.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizaing the policy and value network
    assert args.num_envs == 1 # To use more envs, initial_lstm_state indexing in lines 728 and 745 will have to be added
    # envsperbatch = args.num_envs // args.n_minibatch
    target_agent = Agent(envs.observation_space.shape[0],
                         args.num_glimpses,
                         args.patch_h, 
                         args.patch_w, 
                         args.num_patches, 
                         args.glimpse_scale,
                    ).to(device)
    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        target_agent.load_state_dict(agent.state_dict())
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, newlogproba, entropy, _, _, _, _ = agent.get_action(
                b_obs[minibatch_ind], 
                initial_lstm_state, # here
                b_prev_locs[minibatch_ind],
                b_dones[minibatch_ind],
                b_actions[minibatch_ind],
                b_locs[minibatch_ind],
            )
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(
                b_obs[minibatch_ind], 
                initial_lstm_state, # here
                b_prev_locs[minibatch_ind],
                b_dones[minibatch_ind]
            ).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions[minibatch_ind])[1]).mean() > args.target_kl:
                agent.load_state_dict(target_agent.state_dict())
                break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

envs.close()

# Loading the best model
test_seed = int(time.time())
test_envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i, test=True) for i in range(args.num_envs)]), device)

load_model_path = os.path.join("ckpt", f"{experiment_name}_model_best.pth.tar")
ckpt = torch.load(load_model_path)
print("Loading weights from model", experiment_name, f"with average return over {args.avg_episodes} episodes of", ckpt["best_avg_return"])
print("The best episodic return during training was", ckpt["best_episodic_return"])
agent = Agent(test_envs.observation_space.shape[0],
              args.num_glimpses,
              args.patch_h, 
              args.patch_w, 
              args.num_patches, 
              args.glimpse_scale,
        ).to(device)
agent.load_state_dict(ckpt["model_state"]) 
optimizer.load_state_dict(ckpt["optim_state"])

# Testing the best model
agent.eval()
total_episodes = 0
total_return = 0
test_global_step = 0
loc = torch.zeros((args.num_envs, args.num_glimpses) + (2,)).to(device)
next_obs = test_envs.reset().to(device)
next_done = torch.zeros(args.num_envs).to(device)
next_lstm_state = (
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
)

if args.capture_video:
    test_video_name = f"videos/{experiment_name}/test_video_{total_episodes}.mp4"
    test_recorder = iio.get_writer(test_video_name, fps=10)
    locs_heatmap = torch.zeros(envs.observation_space.shape[1], envs.observation_space.shape[2])

save_glimpses = True
while total_episodes < args.eval_episodes:
    test_global_step += 1 * args.num_envs
    with torch.no_grad():
        action, _, _, loc, next_lstm_state, _, bboxes = agent.get_action(next_obs, next_lstm_state, loc, next_done, save_glimpses=save_glimpses)
    
    next_obs, _, done, info = test_envs.step(action)
    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
    if args.capture_video:
        locs_heatmap = update_loc_heatmap(locs_heatmap, loc[0], args)
    
        frame = next_obs[0].clone().to(torch.uint8)
        frame = draw_bounding_boxes(frame, bboxes[0, :], width=1, colors=(255, 0, 0))
        frame = frame.permute(1, 2, 0)
        test_recorder.append_data(frame.numpy())

    writer.add_scalar(f"test/episodes", total_episodes, test_global_step)

    for item in info:
        if "episode" in item.keys():
            if total_episodes % 10 == 0 and args.capture_video:
                save_glimpses = False
                test_recorder.close()
                wandb.log({"test_video": wandb.Video(test_video_name)})
            
                fig = px.imshow(locs_heatmap)
                wandb.log({"test/loc_heatmap": fig})

                test_video_name = f"videos/{experiment_name}/test_video_{total_episodes + 1}.mp4"
                test_recorder = iio.get_writer(test_video_name, fps=10)
                locs_heatmap = torch.zeros(envs.observation_space.shape[1], envs.observation_space.shape[2])

            total_episodes += 1
            print(f"total_episodes={total_episodes}")
            total_return += item["episode"]["r"]
            writer.add_scalar(f"test/episodic_return", item["episode"]["r"], total_episodes)
            writer.add_scalar(f"test/episodic_length", item["episode"]["l"], total_episodes)

            if total_episodes == 100:
                break

writer.add_text("test/avg_reward", f"Testing: the average return over {args.eval_episodes} was {total_return / args.eval_episodes}")
print(f"Testing: the average return over {args.eval_episodes} was {total_return / args.eval_episodes}")
writer.close()
