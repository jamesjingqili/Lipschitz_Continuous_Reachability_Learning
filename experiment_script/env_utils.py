import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from LCRL.env import DummyVectorEnv
from LCRL.exploration import GaussianNoise
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic
import LCRL.reach_rl_gym_envs as reach_rl_gym_envs

from gymnasium.vector import SyncVectorEnv
from copy import deepcopy
from gymnasium.vector.utils import concatenate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ra_droneracing_Game-v6')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=40000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--total-episodes', type=int, default=160)
    parser.add_argument('--step-per-epoch', type=int, default=40000)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--control-net', type=int, nargs='*', default=[512, 512, 512, 512]) # for control policy
    parser.add_argument('--disturbance-net', type=int, nargs='*', default=[512, 512, 512, 512]) # for disturbance policy
    parser.add_argument('--critic-net', type=int, nargs='*', default=[512, 512, 512, 512]) # for critic net
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--continue-training-logdir', type=str, default=None)
    parser.add_argument('--continue-training-epoch', type=int, default=None)
    parser.add_argument('--actor-gradient-steps', type=int, default=1)
    parser.add_argument('--is-game-baseline', type=bool, default=False) # True -> classical approximated reach-avoid Bellman equation, False -> our new Reach-RL Bellman equation
    parser.add_argument('--target-update-freq', type=int, default=400)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--actor-activation', type=str, default='ReLU')
    parser.add_argument('--critic-activation', type=str, default='ReLU')
    parser.add_argument('--kwargs', type=str, default='{}')
    args = parser.parse_known_args()[0]
    return args    

def get_env_and_policy(args, epoch_id=100, pretrained=True):
    env = gym.make(args.task)
    # check if the environment has control and disturbance actions:
    assert hasattr(env, 'action1_space') and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    args.action1_shape = env.action1_space.shape or env.action1_space.n
    args.action2_shape = env.action2_space.shape or env.action2_space.n
    args.max_action1 = env.action1_space.high[0]
    args.max_action2 = env.action2_space.high[0]

    # if pretrained == False:
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model

    if args.actor_activation == 'ReLU':
        actor_activation = torch.nn.ReLU
    elif args.actor_activation == 'Tanh':
        actor_activation = torch.nn.Tanh
    elif args.actor_activation == 'Sigmoid':
        actor_activation = torch.nn.Sigmoid
    elif args.actor_activation == 'SiLU':
        actor_activation = torch.nn.SiLU

    if args.critic_activation == 'ReLU':
        critic_activation = torch.nn.ReLU
    elif args.critic_activation == 'Tanh':
        critic_activation = torch.nn.Tanh
    elif args.critic_activation == 'Sigmoid':
        critic_activation = torch.nn.Sigmoid
    elif args.critic_activation == 'SiLU':
        critic_activation = torch.nn.SiLU

    if args.critic_net is not None:
        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.critic_net,
            activation=critic_activation,
            concat=True,
            device=args.device
        )
    else:
        critic_net = Net(
            args.state_shape,
            args.action_shape,
            hidden_sizes=args.hidden_sizes,
            activation=critic_activation,
            concat=True,
            device=args.device
        )

    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    critic1 = Critic(critic_net, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(critic_net, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)    
    if args.control_net is None:
        args.control_net = args.hidden_sizes
    if args.disturbance_net is None:
        args.disturbance_net = args.hidden_sizes
    if args.critic_net is None:
        args.critic_net = args.hidden_sizes
    # import pdb; pdb.set_trace()
    log_path = None

    if args.is_game_baseline:
        from LCRL.policy import reach_avoid_game_DDPGPolicy_annealing as DDPGPolicy
        print("DDPG under the Reach-Avoid annealed Bellman equation has been loaded!")
    else:
        from LCRL.policy import reach_avoid_game_DDPGPolicy as DDPGPolicy
        print("DDPG under Reach-RL Bellman equation has been loaded!")
    actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor1 = Actor(
        actor1_net, args.action1_shape, max_action=args.max_action1, device=args.device
    ).to(args.device)
    actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)
    actor2_net = Net(args.state_shape, hidden_sizes=args.disturbance_net, activation=actor_activation, device=args.device)
    actor2 = Actor(
        actor2_net, args.action2_shape, max_action=args.max_action1, device=args.device
    ).to(args.device)
    actor2_optim = torch.optim.Adam(actor2.parameters(), lr=args.actor_lr)

    # policy = DDPGPolicy(
    # critic,
    # critic_optim,
    # tau=args.tau,
    # gamma=args.gamma,
    # exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    # reward_normalization=args.rew_norm,
    # estimation_step=args.n_step,
    # action_space=env.action_space,
    # actor1=actor1,
    # actor1_optim=actor1_optim,
    # actor2=actor2,
    # actor2_optim=actor2_optim,
    # actor_gradient_steps=args.actor_gradient_steps,
    # )
    if args.is_game_baseline:
        log_path = os.path.join(args.logdir, args.task, 'baseline_ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_a2_{}_{}_gamma_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net),
        args.control_net[0],
        len(args.control_net),
        args.disturbance_net[0],
        len(args.disturbance_net),
        args.gamma)
    )
    else:
        log_path = os.path.join(args.logdir, args.task, 'ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_a2_{}_{}_gamma_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net),
        args.control_net[0],
        len(args.control_net),
        args.disturbance_net[0],
        len(args.disturbance_net),
        args.gamma)
    )

    if pretrained == False:
        log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
                args.exploration_noise, 
                args.actor_lr, 
                args.critic_lr, 
                args.batch_size,
                args.step_per_epoch,
                args.kwargs,
                args.seed
            )
    else:
        # NOTE! if you want to use the pre-trained model, you can set the path to the model here:
        # otherwise, you can skip this cell and move to the next cell
        log_path = "Lipschitz_Continuous_Reachability_Learning/experiment_script/pretrained_neural_networks/ra_droneracing_Game-v6/ddpg_reach_avoid_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_8_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0"

    policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space,
        actor1=actor1,
        actor1_optim=actor1_optim,
        actor2=actor2,
        actor2_optim=actor2_optim,
        actor_gradient_steps=args.actor_gradient_steps,
        )
    
    if os.path.exists(log_path):
        policy.load_state_dict(torch.load(log_path+'/epoch_id_{}/policy.pth'.format(epoch_id), map_location=torch.device('cpu')))
        print("\nPolicy loaded!")
    else:
        # print("log_path does not exist!")
        raise(Exception("Policy log path does not exist!"))

    return env, policy

from LCRL.data import Batch
def find_a(state, policy):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    return act

def find_a_batch(states, policy):
    tmp_obs = np.array(states)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy() #.flatten()
    return act

def evaluate_V(state, policy):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy.critic_old(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()

def evaluate_V_batch(states, policy):
    """
    Evaluate V function for a batch of states.
    
    Args:
        states: Can be a single state or batch of states
                - Single state: array-like of shape (state_dim,)
                - Batch of states: array-like of shape (batch_size, state_dim)
    
    Returns:
        Value function outputs:
        - Single state: scalar value
        - Batch: array of shape (batch_size,)
    """
    states = np.array(states)
    
    # Handle both single state and batch of states
    if states.ndim == 1:
        # Single state - add batch dimension
        tmp_obs = states.reshape(1, -1)
        single_state = True
    else:
        # Already a batch of states
        tmp_obs = states
        single_state = False
    
    # Create batch with proper info structure for each state
    batch_size = tmp_obs.shape[0]
    tmp_batch = Batch(
        obs=tmp_obs, 
        info=Batch({} if batch_size == 1 else [{} for _ in range(batch_size)])
    )
    
    # Get actions for the batch
    actions = policy(tmp_batch, model="actor_old").act
    
    # Evaluate critic for the batch
    values = policy.critic_old(tmp_batch.obs, actions)
    values = values.cpu().detach().numpy()
    
    # Return appropriate format
    if single_state:
        return values.flatten()[0]  # Return scalar for single state
    else:
        return values #.flatten()     # Return array for batch


class NoResetSyncVectorEnv(SyncVectorEnv):
    def step_wait(self):
        """Step without automatically resetting environments that are done."""
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            obs, reward, terminated, truncated, info = env.step(action)

            # Keep done flags but do NOT reset
            self._rewards[i] = reward
            self._terminateds[i] = terminated
            self._truncateds[i] = truncated

            observations.append(obs)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )








