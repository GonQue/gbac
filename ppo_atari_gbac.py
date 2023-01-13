import argparse
from collections import deque
import os
import random
import shutil
import time
from distutils.util import strtobool

import gym
from gym.wrappers import (
    RecordEpisodeStatistics, 
    RecordVideo, 
    GrayScaleObservation, 
    ResizeObservation, 
    FrameStack
)
import imageio.v2 as iio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import draw_bounding_boxes, make_grid
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import plotly.express as px
from utils.ram_utils import get_num_layers, get_kernels_and_strides
from utils.truncated_normal import TruncatedNormal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--add-comment', type=str, default="",
                        help='Add comment to the name')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
                        help='the if of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--loc-lr', type=float, default=3e-5,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the environment')
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
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")     
    # Since it is None, wandb will use the default entity which is the username
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")    
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")        
    parser.add_argument('--record-rate', type=int, default=100,
                        help="number of episodes between each saved video of gameplay (with glimpses) and each heatmap")         
    
    # Algorithm specific arguments
    parser.add_argument('--resize', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help="weather to resize the game frame to 84x84 or keep the original size") 
    parser.add_argument('--num-envs', type=int, default=8, 
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128, 
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Use GAE (General Advantage Estimation) for advantage computation")  
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95, 
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4, 
                        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4, 
                        help='the K epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization") 
    parser.add_argument('--clip-coef', type=float, default=0.1, 
                        help='the surrogate clipping coefficient')
    parser.add_argument('--loc-clip-coef', type=float, default=0.2, 
                        help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles weather or not to use a clipped loss for the value function, as per the paper")
    parser.add_argument('--ent-coef', type=float, default=0.01, 
                        help='coefficient of the entropy')
    parser.add_argument('--vf-coef', type=float, default=0.5, 
                        help='coefficient of the value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, 
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None, # openai spinning up default: 0.015
                        help='the target KL divergence threshold')
    parser.add_argument('--diff-lr', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help="Toggles weather or not to use different learning rates for the action and location networks")

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
    parser.add_argument('--loc-hidden', type=int, default=256, 
                        help='hidden size of loc fc')
    parser.add_argument('--glimpse-hidden', type=int, default=512, 
                        help='hidden size of glimpse fc')
    parser.add_argument('--lstm-hidden', type=int, default=128, 
                        help='hidden size of the lstm')
    parser.add_argument('--loc-variance', type=float, default=0.05, 
                        help='variance of the distribution to choose locations')
    args = parser.parse_args()
    if args.patch_size != 0:
        args.patch_h = args.patch_size
        args.patch_w = args.patch_size

    # Default Rollouts Data will be num_envs * num_steps = 4 * 128 = 512, therefore batch_size = 512 
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    full_image_str = "_F" if not args.resize else ""
    args.exp_name = f"{args.gym_id}{full_image_str}__{args.filename}__{args.num_patches}" 
    if len(args.add_comment) != 0:
      args.exp_name += f"_{args.add_comment}"
    return args  


def make_env(gym_id, seed, idx, capture_video, resize, run_name, test=False):
    def thunk():
        if gym_id[-2:] == "v5":
            env = gym.make("ALE/" + gym_id)
        else: 
            env = gym.make(gym_id)

        env = RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                videos_folder = f"videos/Test_{run_name}" if test else f"videos/{run_name}"
                env = RecordVideo(env, videos_folder, episode_trigger=lambda t: t % 1000 == 0)
        if gym_id[-2:] == "v4":
            env = NoopResetEnv(env, noop_max=30) # This wrapper adds stochasticity to the environment (From Revisiting the ALE paper)
            env = MaxAndSkipEnv(env, skip=4) # Technique introduced to save computational time (From DQN Mnih paper)
            env = EpisodicLifeEnv(env) # Other technique used in the DQN paper
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env) # Nobody knows where this wrapper comes from
            env = ClipRewardEnv(env) # DQN paper
        if resize:
            env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env)
        env = FrameStack(env, 1) # Can help the agent identify the velocity of moving objects
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

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


def save_checkpoint(state, is_best):
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")

    ckpt_path = "ckpt/" + run_name + "_ckpt.pth.tar"
    torch.save(state, ckpt_path)
    if is_best:
        best_path = "ckpt/" + run_name + "_model_best.pth.tar"
        shutil.copyfile(ckpt_path, best_path)


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


class GlimpseNetwork(nn.Module):
    def __init__(self, h_g, h_l, num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor, num_channels):
        super(GlimpseNetwork, self).__init__()

        self.num_glimpses = num_glimpses
        self.retina = Retina(num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor)
        num_layers = get_num_layers(glimpse_h)
        if num_layers == 1: 
            k1, s1 = get_kernels_and_strides(glimpse_h)
            final_input_h, final_input_w = conv2d_size_out(glimpse_h, k1, s1), conv2d_size_out(glimpse_w, k1, s1)
            print(f"latent_size: {final_input_h}x{final_input_w}")
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(num_glimpses * num_patches, 64, kernel_size=(k1, k1), stride=(s1, s1))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * final_input_h * final_input_w, h_g)),
                nn.ReLU(),
            )
        elif num_layers == 2: 
            k1, s1, k2, s2 = get_kernels_and_strides(glimpse_h)
            first_conv_h, first_conv_w = conv2d_size_out(glimpse_h, k1, s1), conv2d_size_out(glimpse_w, k1, s1)
            final_input_h, final_input_w = conv2d_size_out(first_conv_h, k2, s2), conv2d_size_out(first_conv_w, k2, s2)
            print(f"1st_conv_size: {first_conv_h}x{first_conv_w}")
            print(f"latent_size: {final_input_h}x{final_input_w}")
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(num_glimpses * num_patches, 32, kernel_size=(k1, k1), stride=(s1, s1))),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=(k2, k2), stride=(s2, s2))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * final_input_h * final_input_w, h_g)),
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
                layer_init(nn.Conv2d(num_glimpses * num_patches, 32, kernel_size=(k1, k1), stride=(s1, s1))),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=(k2, k2), stride=(s2, s2))),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=(k3, k3), stride=(s3, s3))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * final_input_h * final_input_w, h_g)),
                nn.ReLU(),
            )

        # glimpse layer
        # D_in = num_patches * glimpse_h * glimpse_w * num_channels * num_glimpses
        # self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 2 * num_glimpses
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev, save_glimpses=False):
        # generate glimpse phi from image x
        phi, bboxes = self.retina.foveate(x, l_t_prev, save_glimpses)
        phi_out = self.network(phi)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return phi_out, g_t, bboxes


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, h_g, h_l, num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor, num_channels, hidden_size):
        super(Agent, self).__init__()
        
        self.num_glimpses = num_glimpses
        self.sensor = GlimpseNetwork(h_g, h_l, num_glimpses, glimpse_h, glimpse_w, num_patches, scaling_factor, num_channels)
        self.glimpse_lstm = nn.LSTM(h_g, hidden_size)
        for name, param in self.glimpse_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.loc_lstm = nn.LSTM(h_g + h_l, hidden_size)
        for name, param in self.loc_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # std=0.01 ensures the layers parameters will have similar scalar values, and
        # as a result the probability of taking each action will be similar
        print(envs.single_action_space.n)
        self.actor = layer_init(nn.Linear(hidden_size, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)    
        self.locator = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size // 2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size // 2, 2 * self.num_glimpses)),
            nn.Tanh(),
        )
    
    def get_states(self, x, glimpse_lstm_state, loc_lstm_state, prev_loc, done, save_glimpses=False):
        # image size: torch.Size([8, 1, 84, 84])
        # print("image size:", x.shape)
        # l_t = torch.FloatTensor(x.shape[0], 2).uniform_(-1, 1).to(device)
        glimpse_hidden, loc_hidden, bboxes = self.sensor(x / 255.0, prev_loc, save_glimpses)

        # LSTM logic
        batch_size = glimpse_lstm_state[0].shape[1]
        done = done.reshape((-1, batch_size))

        glimpse_hidden = glimpse_hidden.reshape((-1, batch_size, self.glimpse_lstm.input_size))
        new_glimpse_hidden = []
        for h, d in zip(glimpse_hidden, done):
            h, glimpse_lstm_state = self.glimpse_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * glimpse_lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * glimpse_lstm_state[1],
                ),
            )
            new_glimpse_hidden += [h]
        new_glimpse_hidden = torch.flatten(torch.cat(new_glimpse_hidden), 0, 1)

        loc_hidden = loc_hidden.reshape((-1, batch_size, self.loc_lstm.input_size))
        new_loc_hidden = []
        for h, d in zip(loc_hidden, done):
            h, loc_lstm_state = self.loc_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * loc_lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * loc_lstm_state[1],
                ),
            )
            new_loc_hidden += [h]
        new_loc_hidden = torch.flatten(torch.cat(new_loc_hidden), 0, 1)
        # print("new_hidden:", new_hidden.shape)
        # print("lstm_state:", lstm_state[0].shape)
        return new_glimpse_hidden, glimpse_lstm_state, new_loc_hidden, loc_lstm_state, bboxes
    
    # Critic's inference
    def get_value(self, x, glimpse_lstm_state, loc_lstm_state, prev_loc, done):
        hidden, _, _, _, _ = self.get_states(x, glimpse_lstm_state, loc_lstm_state, prev_loc, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, glimpse_lstm_state, loc_lstm_state, prev_loc, done, action=None, loc=None, save_glimpses=False):
        # print("x:", x.shape)
        # print("lstm_state:", lstm_state[0].shape)
        glimpse_hidden, glimpse_lstm_state, loc_hidden, loc_lstm_state, bboxes = self.get_states(x, glimpse_lstm_state, loc_lstm_state, prev_loc, done, save_glimpses=save_glimpses) 
        action_logits = self.actor(glimpse_hidden) # logits are unnormalized action probabilities
        action_probs = Categorical(logits=action_logits) # Categorical dist is essentially a softmax operation to get the action probs dist
        if action is None:
            action = action_probs.sample()

        mu = self.locator(loc_hidden.detach())
        # mu = self.locator(loc_hidden)
        mu = mu.view(-1, self.num_glimpses, 2)
        loc_probs = TruncatedNormal(mu, args.loc_variance, -1, 1) # 0.05 from RAM code
        if loc is None:
            # When using rsample() we can still backpropagate using the reparameterization trick,
            # when using sample(), the computation graph is cut off
            loc = loc_probs.rsample().detach()
            # loc = loc_probs.sample()

            done_reshaped = torch.zeros_like(loc).to(device)
            done_reshaped[:, 0, :] = done.unsqueeze(-1).expand(-1, 2)
            # Reset location if we get a done (The next glimpse will be in the center because the agent
            # is starting a new game)
            # loc = torch.clamp(loc, -1, 1) * (1.0 - done_reshaped)
            loc = loc * (1.0 - done_reshaped)

            loc_log_probs = loc_probs.log_prob(loc)
            loc_log_probs = torch.sum(loc_log_probs, dim=-1)

            # Tensor with random values that will be 0 if there is not a done
            # rand_locs = torch.rand_like(loc).clamp(-1, 1).to(device)
            # loc = loc + (rand_locs * done_reshaped) # since the coordinates that got reseted are 0, they will have a random value

        else:
            loc_log_probs = loc_probs.log_prob(loc)
            loc_log_probs = torch.sum(loc_log_probs, dim=-1)

        # print("loc_log_probs:", loc_log_probs)
        return action, action_probs.log_prob(action), action_probs.entropy(), loc, loc_log_probs, loc_probs.entropy.sum(-1), self.critic(glimpse_hidden), glimpse_lstm_state, loc_lstm_state, bboxes

           
if __name__ == "__main__":
    args = parse_args()
    full_image_str = "F" if not args.resize else ""
    run_name = f"{args.gym_id}__{args.filename}__{args.num_glimpses}g_{args.patch_h}x{args.patch_w}{full_image_str}_{args.glimpse_scale}s_{args.num_patches}p"  
    if len(args.add_comment) != 0:
      run_name += f"_{args.add_comment}" 
    run_name += f"__{args.seed}__{int(time.time())}"  
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

    device = torch.device("cuda" if torch.cuda.is_available and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, args.resize, run_name)
    for i in range(args.num_envs)])

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    # print("envs.single_action_space.n", envs.single_action_space.n)
    agent = Agent(envs, 
                  args.glimpse_hidden, 
                  args.loc_hidden, 
                  args.num_glimpses,
                  args.patch_h, 
                  args.patch_w, 
                  args.num_patches, 
                  args.glimpse_scale, 
                  envs.single_observation_space.shape[0],
                  args.lstm_hidden,
                ).to(device)

    pytorch_total_params = sum(p.numel() for p in agent.parameters())
    print(f'Model parameters: {pytorch_total_params}')
    # print(agent)
    # os._exit()
    if args.diff_lr:
        params = [{'params' : [p[1] for p in agent.named_parameters() if not p[0].startswith("loc") and not p[0].startswith("sensor.fc2")]},	
                  {'params' : [p[1] for p in agent.named_parameters() if p[0].startswith("loc") or p[0].startswith("sensor.fc2")], 'lr': args.loc_lr}]
    else:
        params = agent.parameters()
        
    # Original implementation of PPO uses eps=1e-5, instead of the pytorch default 1e-8
    optimizer = optim.Adam(params, lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    prev_locs = torch.zeros((args.num_steps, args.num_envs, args.num_glimpses) + (2,)).to(device)
    # prev_locs = torch.rand((args.num_steps, args.num_envs, args.num_glimpses) + (2,)).clamp(-1, 1).to(device)
    locs = torch.zeros((args.num_steps, args.num_envs, args.num_glimpses) + (2,)).to(device)
    loc_logprobs = torch.zeros((args.num_steps, args.num_envs, args.num_glimpses)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    best_episodic_return = float('-inf')
    best_avg_return = float('-inf')
    is_best = False
    global_step = 0
    episodes_counter = 0
    total_episodes = 0
    returns_queue = deque(maxlen=args.avg_episodes)
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_glimpse_lstm_state = (
        torch.zeros(agent.glimpse_lstm.num_layers, args.num_envs, agent.glimpse_lstm.hidden_size).to(device),
        torch.zeros(agent.glimpse_lstm.num_layers, args.num_envs, agent.glimpse_lstm.hidden_size).to(device),
    ) # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    next_loc_lstm_state = (
        torch.zeros(agent.loc_lstm.num_layers, args.num_envs, agent.loc_lstm.hidden_size).to(device),
        torch.zeros(agent.loc_lstm.num_layers, args.num_envs, agent.loc_lstm.hidden_size).to(device),
    )
    num_updates = args.total_timesteps // args.batch_size
    # print(num_updates)
    # print("next_obs.shape", next_obs.shape)
    # print("agent.get_value(next_obs)", agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    # print()
    # print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))
    if args.capture_video:
        video_name = f"videos/{run_name}/video_{episodes_counter}.mp4"
        recorder = iio.get_writer(video_name, fps=10)
        locs_heatmap = torch.zeros(envs.single_observation_space.shape[1], envs.single_observation_space.shape[2])

    # num_next_obs = 0
    for update in range(1, num_updates + 1):
        initial_glimpse_lstm_state = (next_glimpse_lstm_state[0].clone(), next_glimpse_lstm_state[1].clone())
        initial_loc_lstm_state = (next_loc_lstm_state[0].clone(), next_loc_lstm_state[1].clone())
        # prev_locs[0] = torch.FloatTensor(args.num_envs, 2).uniform_(-1, 1).to(device)
        # Annealing the learning rate if instructed to do so.
        if args.anneal_lr:
            # frac is 1 at the first update and will linearly decrease to 0 at the end of the updates
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            if args.diff_lr:
                loc_lrnow = frac * args.loc_lr
                optimizer.param_groups[1]["lr"] = loc_lrnow
        
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # During the rollout, we don't need to catch any gradients
            with torch.no_grad():
                # prev_loc = locs[step - 1] if step != 0 else initial_loc
                # num_next_obs += 1
                action, action_logprob, _, loc, loc_logprob, _, value, next_glimpse_lstm_state, next_loc_lstm_state, bboxes = agent.get_action_and_value(next_obs, next_glimpse_lstm_state, next_loc_lstm_state, prev_locs[step], next_done)
                values[step] = value.flatten()
            actions[step] = action
            action_logprobs[step] = action_logprob
            if step < args.num_steps - 1:
                prev_locs[step + 1] = loc
            locs[step] = loc
            loc_logprobs[step] = loc_logprob

            if args.capture_video and episodes_counter % args.record_rate == 0:
                locs_heatmap = update_loc_heatmap(locs_heatmap, locs[step][0], args)
                for x in next_obs[0]:
                    frame = x.clone().to(torch.uint8).unsqueeze(dim=0)
                    frame = draw_bounding_boxes(frame, bboxes[0, :], width=1, colors=(255, 0, 0))
                    frame = frame.permute(1, 2, 0)
                    # fig = px.imshow(frame.numpy())
                    # fig.write_image("frame.png")
                    recorder.append_data(frame.numpy())

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            is_best = False
            some_ep_finished = False
            for j, item in enumerate(info):
                if "episode" in item.keys():
                    some_ep_finished = True
                    total_episodes += 1
                    returns_queue.append(item["episode"]["r"])
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    avg_return = sum(returns_queue) / args.avg_episodes

                    if len(returns_queue) == args.avg_episodes and avg_return >= best_avg_return:
                        best_avg_return = avg_return
                        is_best = True
                        
                    if item["episode"]["r"] >= best_episodic_return:
                        best_episodic_return = item["episode"]["r"]

                    if j == 0 and args.capture_video and episodes_counter % args.record_rate == 0:
                        recorder.close()
                        wandb.log({"video": wandb.Video(video_name)})
                        
                        fig = px.imshow(locs_heatmap)
                        wandb.log({"charts/loc_heatmap": fig})
                        # print("HEATMAP:", locs_heatmap.sum())
                        # print("next_obs total:", num_next_obs)
                        # num_next_obs = 0
                        locs_heatmap = torch.zeros(envs.single_observation_space.shape[1], envs.single_observation_space.shape[2])
                    if j == 0:
                        episodes_counter += 1    
                        
                    # the next episode will be captured (if statement needs to be after episodes_counter increment in order to prepare the recorder)
                    if j == 0 and args.capture_video and episodes_counter % args.record_rate == 0:
                        video_name = f"videos/{run_name}/video_{episodes_counter}.mp4"
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
                    }, is_best)
                is_best = False

        
        # ALGO LOGIC: advantage calculation
        # bootstrap value if not done
        with torch.no_grad():
            # print("next value")
            next_value = agent.get_value(next_obs, next_glimpse_lstm_state, next_loc_lstm_state, locs[step], next_done).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    # returns = sum of discounted rewards
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_action_logprobs = action_logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_loc_logprobs = loc_logprobs.reshape((-1, args.num_glimpses))
        b_locs = locs.reshape((-1, args.num_glimpses) + (2,))
        b_prev_locs = prev_locs.reshape((-1, args.num_glimpses) + (2,))

        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        # Log Variable: clipfracs measures how often the clipped objective is actually triggered
        action_clipfracs = []
        loc_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel() # be really careful about the index
                # mb_inds[mb_inds < 0] = initial_loc[0]
                # print("small s:", b_locs[mb_inds[0]].shape)
                # Training
                # were we also pass the minibatch actions, therefore the agent won't sample new ones
                # print("training")
                # print(mb_inds)
                # print(mb_inds - 1)
                _, newactionlogprob, action_entropy, _, newloclogprob, loc_entropy, newvalue, _, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    (initial_glimpse_lstm_state[0][:, mbenvinds], initial_glimpse_lstm_state[1][:, mbenvinds]),
                    (initial_loc_lstm_state[0][:, mbenvinds], initial_loc_lstm_state[1][:, mbenvinds]),
                    b_prev_locs[mb_inds],
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    b_locs[mb_inds],
                ) 
                action_logratio = newactionlogprob - b_action_logprobs[mb_inds]
                action_ratio = action_logratio.exp()
                # print("newloclogprob:", newloclogprob)
                # print("newloclogprob:", newloclogprob.shape)
                # print("b_loc_logprobs:", b_loc_logprobs[mb_inds].shape)
                loc_logratio = (newloclogprob - b_loc_logprobs[mb_inds]).mean()
                loc_ratio = loc_logratio.exp()

                with torch.no_grad():
                    # Log Variables
                    # KL-Divergence helps us understand how aggresively the policy updates
                    old_approx_kl = (-action_logratio).mean()
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # according to the link, this is a better estimator
                    action_approx_kl = ((action_ratio - 1) - action_logratio).mean()
                    loc_approx_kl = ((loc_ratio - 1) - loc_logratio).mean()
                    action_clipfracs += [((action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    loc_clipfracs += [((loc_ratio - 1.0).abs() > args.loc_clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # 1e-8 is a small scalar value to prevent division by 0 error
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                action_pg_loss1 = -mb_advantages * action_ratio
                action_pg_loss2 = -mb_advantages * torch.clamp(action_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                action_pg_loss = torch.max(action_pg_loss1, action_pg_loss2).mean()

                loc_pg_loss1 = -mb_advantages * loc_ratio
                loc_pg_loss2 = -mb_advantages * torch.clamp(loc_ratio, 1 - args.loc_clip_coef, 1 + args.loc_clip_coef)
                loc_pg_loss = torch.max(loc_pg_loss1, loc_pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # Usual loss implementation: mean_squared_error
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # entropy is the measure of the level of caos in the action probability distribution
                # intuitively, maximizing entropy would encourage the agent to explore more
                action_entropy_loss = action_entropy.mean()
                loc_entropy_loss = loc_entropy.mean()
                # The idea is to minimize the policy loss and the value loss but maximize the entropy loss
                loss = action_pg_loss + loc_pg_loss - args.ent_coef * action_entropy_loss + v_loss * args.vf_coef
                # loss = action_pg_loss + loc_pg_loss - args.ent_coef * action_entropy_loss - args.ent_coef * loc_entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad() # resets gradients
                loss.backward() # backward pass
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step() # gradient descent
            
            # Early stopping (if enabled)
            if args.target_kl is not None:
                if action_approx_kl > args.target_kl:
                    break

        # Log Variable: 
        # explained_variance tells us if the value function is a good indicator of the returns
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", action_pg_loss.item(), global_step)
        writer.add_scalar("losses/loc_loss", loc_pg_loss.item(), global_step)
        writer.add_scalar("losses/action_entropy", action_entropy_loss.item(), global_step)
        writer.add_scalar("losses/loc_entropy", loc_entropy_loss.item(), global_step)
        writer.add_scalar("losses/action_approx_kl", action_approx_kl.item(), global_step)
        writer.add_scalar("losses/loc_approx_kl", loc_approx_kl.item(), global_step)
        writer.add_scalar("losses/action_clipfrac", np.mean(action_clipfracs), global_step)
        writer.add_scalar("losses/loc_clipfrac", np.mean(loc_clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    envs.close()

    # Loading the best model
    test_seed = int(time.time())
    test_envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, test_seed + i, i, args.capture_video, args.resize, run_name, test=True)
    for i in range(args.num_envs)])

    load_model_path = os.path.join("ckpt", f"{run_name}_model_best.pth.tar")
    ckpt = torch.load(load_model_path)
    print("Loading weights from model", run_name, f"with average return over {args.avg_episodes} episodes of", ckpt["best_avg_return"])
    print("The best episodic return during training was", ckpt["best_episodic_return"], )
    agent = Agent(test_envs, 
                  args.glimpse_hidden, 
                  args.loc_hidden, 
                  args.num_glimpses,
                  args.patch_h, 
                  args.patch_w, 
                  args.num_patches, 
                  args.glimpse_scale, 
                  envs.single_observation_space.shape[0],
                  args.lstm_hidden,
                ).to(device)
    agent.load_state_dict(ckpt["model_state"]) 
    optimizer.load_state_dict(ckpt["optim_state"])

    # Testing the best model
    agent.eval()
    total_episodes = 0
    total_return = 0
    test_global_step = 0
    loc = torch.zeros((args.num_envs, args.num_glimpses) + (2,)).to(device)
    next_obs = torch.Tensor(test_envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_glimpse_lstm_state = (
        torch.zeros(agent.glimpse_lstm.num_layers, args.num_envs, agent.glimpse_lstm.hidden_size).to(device),
        torch.zeros(agent.glimpse_lstm.num_layers, args.num_envs, agent.glimpse_lstm.hidden_size).to(device),
    ) # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    next_loc_lstm_state = (
        torch.zeros(agent.loc_lstm.num_layers, args.num_envs, agent.loc_lstm.hidden_size).to(device),
        torch.zeros(agent.loc_lstm.num_layers, args.num_envs, agent.loc_lstm.hidden_size).to(device),
    )
    if args.capture_video:
        test_video_name = f"videos/{run_name}/test_video_{episodes_counter}.mp4"
        test_recorder = iio.get_writer(test_video_name, fps=10)
        locs_heatmap = torch.zeros(envs.single_observation_space.shape[1], envs.single_observation_space.shape[2])

    save_glimpses = True
    while total_episodes < args.eval_episodes:
        test_global_step += 1 * args.num_envs
        with torch.no_grad():
            action, _, _, loc, _, _, _, next_glimpse_lstm_state, next_loc_lstm_state, bboxes = agent.get_action_and_value(
                next_obs, next_glimpse_lstm_state, next_loc_lstm_state, loc, next_done, save_glimpses=save_glimpses)
        
        next_obs, _, done, info = test_envs.step(action.cpu().numpy())
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        if args.capture_video:
            locs_heatmap = update_loc_heatmap(locs_heatmap, loc[0], args)
            for x in next_obs[0]:
                frame = x.clone().to(torch.uint8).unsqueeze(dim=0)
                frame = draw_bounding_boxes(frame, bboxes[0, :], width=1, colors=(255, 0, 0))
                frame = frame.permute(1, 2, 0)
                test_recorder.append_data(frame.numpy())

        writer.add_scalar(f"test/episodes", total_episodes, test_global_step)

        for i, item in enumerate(info):
            if "episode" in item.keys():
                total_episodes += 1
                print(f"total_episodes={total_episodes}")
                total_return += item["episode"]["r"]
                writer.add_scalar(f"test/episodic_return", item["episode"]["r"], total_episodes)
                writer.add_scalar(f"test/episodic_length", item["episode"]["l"], total_episodes)
                
                if i == 0:
                    save_glimpses = False
                    if args.capture_video:
                        test_recorder.close()
                        wandb.log({"test_video": wandb.Video(test_video_name)})
                    
                        fig = px.imshow(locs_heatmap)
                        wandb.log({"test/loc_heatmap": fig})

                        test_video_name = f"videos/{run_name}/test_video_{episodes_counter}.mp4"
                        test_recorder = iio.get_writer(test_video_name, fps=10)
                        locs_heatmap = torch.zeros(envs.single_observation_space.shape[1], envs.single_observation_space.shape[2])

                if total_episodes == 100:
                    break


    writer.add_text("test/avg_reward", f"Testing: the average return over {args.eval_episodes} was {total_return / args.eval_episodes}")
    print(f"Testing: the average return over {args.eval_episodes} was {total_return / args.eval_episodes}")
    writer.close()