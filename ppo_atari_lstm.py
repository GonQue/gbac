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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

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
    
    # Algorithm specific arguments
    parser.add_argument('--resize', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
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
    args = parser.parse_args()

    # Default Rollouts Data will be num_envs * num_steps = 4 * 128 = 512, therefore batch_size = 512 
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    full_image_str = "_F" if not args.resize else ""
    args.exp_name = f"{args.gym_id}__{args.filename}{full_image_str}" 
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        input_h = envs.single_observation_space.shape[1]
        input_w = envs.single_observation_space.shape[2]
        if args.resize:
            k1_h, s1_h, k2_h, s2_h, k3_h, s3_h = 8, 4, 4, 2, 3, 1
            k1_w, s1_w, k2_w, s2_w, k3_w, s3_w = 8, 4, 4, 2, 3, 1
        else:
            k1_h, s1_h, k2_h, s2_h, k3_h, s3_h = 15, 5, 5, 4, 3, 1
            k1_w, s1_w, k2_w, s2_w, k3_w, s3_w = 15, 5, 5, 3, 3, 1

        first_conv_h, first_conv_w = conv2d_size_out(input_h, k1_h, s1_h), conv2d_size_out(input_w, k1_w, s1_w)
        second_conv_h, second_conv_w = conv2d_size_out(first_conv_h, k2_h, s2_h), conv2d_size_out(first_conv_w, k2_w, s2_w)
        final_input_h, final_input_w = conv2d_size_out(second_conv_h, k3_h, s3_h), conv2d_size_out(second_conv_w, k3_w, s3_w)
        print(f"1st_conv_size: {first_conv_h}x{first_conv_w}")
        print(f"2nd_conv_size: {second_conv_h}x{second_conv_w}")
        print(f"latent_size: {final_input_h}x{final_input_w}")
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, kernel_size=(k1_h, k1_w), stride=(s1_h, s1_w))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=(k2_h, k2_w), stride=(s2_h, s2_w))),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=(k3_h, k3_w), stride=(s3_h, s3_w))),
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

        # std=0.01 ensures the layers parameters will have similar scalar values, and
        # as a result the probability of taking each action will be similar
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)    
    
    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
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
        return new_hidden, lstm_state
    
    # Critic's inference
    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done) 
        logits = self.actor(hidden) # logits are unnormalized action probabilities
        probs = Categorical(logits=logits) # Categorical dist is essentially a softmax operation to get the action probs dist
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

           
if __name__ == "__main__":
    args = parse_args()
    full_image_str = "_F" if not args.resize else ""
    run_name = f"{args.gym_id}__{args.filename}{full_image_str}" 
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

    agent = Agent(envs).to(device)
    pytorch_total_params = sum(p.numel() for p in agent.parameters())
    print(f'Model parameters: {pytorch_total_params}')
    # print(agent)
    # Original implementation of PPO uses eps=1e-5, instead of the pytorch default 1e-8
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
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
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    ) # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    num_updates = args.total_timesteps // args.batch_size
    # print(num_updates)
    # print("next_obs.shape", next_obs.shape)
    # print("agent.get_value(next_obs)", agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    # print()
    # print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the learning rate if instructed to do so.
        if args.anneal_lr:
            # frac is 1 at the first update and will linearly decrease to 0 at the end of the updates
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # During the rollout, we don't need to catch any gradients
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            is_best = False
            some_ep_finished = False
            for item in info:
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
        
        # ALGO LOGIC: advantage calculation
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
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
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel() # be really careful about the index
                # Training
                # were we also pass the minibatch actions, therefore the agent won't sample new ones
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                ) 
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Log Variables
                    # KL-Divergence helps us understand how aggresively the policy updates
                    old_approx_kl = (-logratio).mean()
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # according to the link, this is a better estimator
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # 1e-8 is a small scalar value to prevent division by 0 error
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                entropy_loss = entropy.mean()
                # The idea is to minimize the policy loss and the value loss but maximize the entropy loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad() # resets gradients
                loss.backward() # backward pass
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step() # gradient descent
            
            # Early stopping (if enabled)
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Log Variable: 
        # explained_variance tells us if the value function is a good indicator of the returns
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
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
    print("The best episodic return during training was", ckpt["best_episodic_return"])
    agent = Agent(test_envs).to(device)
    agent.load_state_dict(ckpt["model_state"]) 
    optimizer.load_state_dict(ckpt["optim_state"])

    # Testing the best model
    agent.eval()
    total_episodes = 0
    total_return = 0
    test_global_step = 0
    next_obs = torch.Tensor(test_envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )
    while total_episodes < args.eval_episodes:
        test_global_step += 1 * args.num_envs
        with torch.no_grad():
            action, _, _, _, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
        
        next_obs, _, done, info = test_envs.step(action.cpu().numpy())
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        writer.add_scalar(f"test/episodes", total_episodes, test_global_step)

        for item in info:
            if "episode" in item.keys():
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