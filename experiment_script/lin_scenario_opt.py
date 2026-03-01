import sys 
sys.path.append(
    '/Users/sampada/Documents/Research/Bayesian_Optimization/code/bayes_opt_calibration/')

import os
import gymnasium as gym
import numpy as np
import scipy
import torch
import time

from LCRL.env import DummyVectorEnv
from LCRL.exploration import GaussianNoise
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic

from Lipschitz_Continuous_Reachability_Learning import experiment_script
from experiment_script.env_utils import NoResetSyncVectorEnv, evaluate_V_batch, find_a_batch, find_a, get_args, get_env_and_policy

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm

def make_new_env(args):
    return gym.make(args.task)

# Code drawn from Albert Lin (https://github.com/albertklin/conformal_prediction_verification/blob/main/run_verification.py)
# Lemma 5, Lin & Bansal 2024
def find_eps_lemma5(N, k, beta):
    # N: number of calibration points
    # k: number of violations in calibration set
    # eps: safety violation parameter
    # beta: confidence parameter
    eps_candidates = np.linspace(start=0, stop=1, num=int(1e3))
    beta_candidates = scipy.special.bdtr(k, N, eps_candidates)
    eps = eps_candidates[np.argmax(beta_candidates <= beta)]
    return eps

def compute_min_scenarios_alex(epsilon, delta, d):
    """
    Compute the minimum number of scenarios needed to ensure feasibility with given epsilon and delta.
    """
    # num = int(np.ceil((math.exp(1) / (epsilon*(math.exp(1)-1)))*(np.log(1/delta) + d*(d+1)/2 + d)))
    # num = torch.tensor(num, device=device)
    # num = int((2 / epsilon) * (np.log(1 / delta) + (d-1)* np.log(2)))
    num = int((2 / epsilon) * (np.log(1 / delta) + 1))
    return num

def sample_init_cond(N, alpha, policy, rng, model_dim, range_x, 
                    ego_setting, adversary_setting):
    """
    Sample N initial conditions in state space that satisfy the reach-avoid constraints.
    """
    print("Sampling initial conditions")
    ego_vx, ego_vy, ego_z, ego_vz = ego_setting
    ad_x, ad_vx, ad_y, ad_vy, ad_z, ad_vz = adversary_setting

    init_cond_final = []
    V_values_final = []
    have_sufficient = False
    try_N = N * 10

    while not have_sufficient:
        # If model dim is 2
        x01 = rng.uniform(range_x[0][0], range_x[0][1], size=(try_N, 1))
        ego_vx1 = np.full((try_N, 1), ego_vx)
        y01 = rng.uniform(range_x[2][0], range_x[2][1], size=(try_N, 1))
        ego_vy1 = np.full((try_N, 1), ego_vy)
        z01 = np.full((try_N, 1), ego_z)
        ego_vz1 = np.full((try_N, 1), ego_vz)

        ad_x1 = np.full((try_N, 1), ad_x)
        ad_vx1 = np.full((try_N, 1), ad_vx)
        ad_y1 = np.full((try_N, 1), ad_y)
        ad_vy1 = np.full((try_N, 1), ad_vy)
        ad_z1 = np.full((try_N, 1), ad_z)
        ad_vz1 = np.full((try_N, 1), ad_vz)

        if model_dim == 2: pass    
        elif model_dim == 3:
            z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(try_N, 1))   
        elif model_dim == 4:
            ego_vx1 = rng.uniform(range_x[1][0], range_x[1][1], size=(try_N, 1))
            z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(try_N, 1))
        elif model_dim == 6:
            ego_vx1 = rng.uniform(range_x[1][0], range_x[1][1], size=(try_N, 1))
            ego_vy1 = rng.uniform(range_x[3][0], range_x[3][1], size=(try_N, 1))
            z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(try_N, 1))
            ego_vz1 = rng.uniform(range_x[5][0], range_x[5][1], size=(try_N, 1))
        elif model_dim == 12:
            ego_vx1 = rng.uniform(range_x[1][0], range_x[1][1], size=(try_N, 1))
            ego_vy1 = rng.uniform(range_x[3][0], range_x[3][1], size=(try_N, 1))
            z01 = rng.uniform(range_x[4][0], range_x[4][1], size=(try_N, 1))
            ego_vz1 = rng.uniform(range_x[5][0], range_x[5][1], size=(try_N, 1))
            ad_x1 = rng.uniform(range_x[6][0], range_x[6][1], size=(try_N, 1))
            ad_vx1 = rng.uniform(range_x[7][0], range_x[7][1], size=(try_N, 1))
            ad_y1 = rng.uniform(range_x[8][0], range_x[8][1], size=(try_N, 1))
            ad_vy1 = rng.uniform(range_x[9][0], range_x[9][1], size=(try_N, 1))
            ad_z1 = rng.uniform(range_x[10][0], range_x[10][1], size=(try_N, 1))
            ad_vz1 = rng.uniform(range_x[11][0], range_x[11][1], size=(try_N, 1))
        else:
            raise NotImplementedError
        
        init_cond = np.hstack((x01, ego_vx1,
                                y01, ego_vy1,
                                z01, ego_vz1,
                                ad_x1, ad_vx1,
                                ad_y1, ad_vy1,
                                ad_z1, ad_vz1))
        
        # Check if the sampled initial conditions satisfy the reach-avoid constraints
        # Use evaluate_V to check the value function
        V_values = evaluate_V_batch(init_cond, policy)
        # Append init cond where V_values > alpha
        valid_indices = np.where(V_values > alpha)[0]
        init_cond_final.append(init_cond[valid_indices])
        V_values_final.append(V_values[valid_indices])
        
        # if len(init_cond_final) >= N:
        total_samps = sum(arr.shape[0] for arr in init_cond_final)
        if total_samps >= N:
            init_cond_final = np.vstack(init_cond_final)[:N]
            V_values_final = np.hstack(V_values_final)[:N]
            have_sufficient = True

    return init_cond_final, V_values_final

def sample_noise(N, horizon, epsilon_d):
    """
    Sample N noise vectors in disturbance space.
    """
    # Sample noise vectors uniformly in the disturbance space
    noise = np.random.uniform(-epsilon_d, epsilon_d, size=(N, horizon, 3))
    return noise


def reach_avoid_measure_vectorized(envv, horizon, init_cond_final, V_values, policy, args):
    """
    Measure the reach-avoid performance of the system given the initial conditions and their corresponding V values.
    This is a vectorized version for efficiency.
    """
    num_samples = init_cond_final.shape[0]
    # print(f"Number of samples for vectorized reach-avoid measure: {num_samples}")
    n_dim = envv.observation_space.shape[0]
    reach_avoid_measures = np.zeros(num_samples)

    envs = NoResetSyncVectorEnv([lambda: make_new_env(args) for _ in range(num_samples)])
    
    for i, state in enumerate(init_cond_final):
        envs.envs[i].reset(options={'initial_state': state})

    rewards = np.zeros((num_samples, horizon))
    constraints = np.zeros((num_samples, horizon))
    states = np.array([env.state for env in envs.envs])
    state_trajs = np.zeros((num_samples, n_dim, horizon+1))
    state_trajs[:, :, 0] = states

    for t in range(horizon):
        current_states = np.array([env.state for env in envs.envs])
        acts = find_a_batch(current_states, policy)
        actions = np.concatenate((acts[:, :3], np.zeros((num_samples, 3))), axis=1)  # assuming no noise in action for now
        states, rew, done, _, info = envs.step(actions)
        state_trajs[:, :, t+1] = states
        rewards[:, t] = rew * args.gamma**t
        constraints[:, t] = info['constraint'] * args.gamma**t

    min_constraints = np.minimum.accumulate(constraints, axis=1)
    reach_avoid_measures = np.max(np.minimum(rewards, min_constraints), axis=1)
    state_trajs_iterative = state_trajs
    return reach_avoid_measures, state_trajs_iterative
    

def reach_avoid_measure(env, horizon, init_cond_final, V_values):
    """
    Measure the reach-avoid performance of the system given the initial conditions and their corresponding V values.
    """
    num_samples = init_cond_final.shape[0]
    n_dim = env.observation_space.shape[0]
    reach_avoid_measures = np.zeros(num_samples)
    
    for i in range(num_samples):
        state = init_cond_final[i]
        V_value = V_values[i]
        
        # Reset the environment with the sampled initial condition
        options = {'initial_state': state}
        env.reset(options=options)
        rewards = np.zeros(horizon)
        constraints = np.zeros(horizon)
        
        # Simulate the environment for the given horizon
        for t in range(horizon):
            act = find_a(state)
            action = np.concatenate((act[:3], np.zeros(3)))  # assuming no noise in action for now 9/30/2025

            next_state, rew, _, _, info = env.step(action)
            cstrt = info['constraint']
            rewards[t] = rew
            constraints[t] = cstrt
           
            state = next_state

        # Want max (over t){ min(rewards[t], min(over t){ constraints[t] }) }
        reach_avoid_measures[i] = np.max(np.minimum(rewards, np.minimum.accumulate(constraints)))

    return reach_avoid_measures

def get_new_alpha(env, init_cond_final, V_values_final, alpha, horizon, policy, args) : #, noise):
    """
    Get a new alpha value based on the sampled initial conditions and their corresponding V values
    and reach avoid measures.
    """
    print("Getting new alpha")
    # reach_avoid_measures = reach_avoid_measure(env, horizon, noise, init_cond_final, V_values_final)
    reach_avoid_measures, state_trajs_iterative = reach_avoid_measure_vectorized(env, horizon, init_cond_final, V_values_final, policy, args)

    # import pdb; pdb.set_trace()

    num_safety_violations = np.sum(reach_avoid_measures < 0)

    if np.any(reach_avoid_measures < 0):
        # If any reach-avoid measure is negative, we need to adjust alpha
        new_alpha = np.max(V_values_final[reach_avoid_measures < 0])
    else:
        # If all reach-avoid measures are non-negative, we can keep the current alpha
        new_alpha = alpha

    return new_alpha, state_trajs_iterative, num_safety_violations

def solve_iterative_method(env, eps, delt, M, horizon, policy, dim, args,
                            rng, range_x, ego_setting, adv_setting, 
                            alpha_init=np.inf):
    """
    Solve the iterative method for reach-avoid certification.
    """
    alpha = alpha_init
    # N = compute_min_scenarios_alex(eps, delt, d=12)
    N = compute_min_scenarios_alex(eps, delt, d=dim)
    print("N: ", N)
    
    start_time = time.time()
    total_num_samples = 0
    total_num_safety_violations = 0
    for j in range(M):
        init_cond_final, V_values_final = sample_init_cond(N, alpha, policy, rng, dim,
                                                            range_x, ego_setting, 
                                                            adv_setting)
        print(init_cond_final.shape)
        total_num_samples += len(init_cond_final)
        # noise = sample_noise(N, horizon, epsilon_d)
        new_alpha, state_traj_iterative, num_safety_violations = \
            get_new_alpha(env, init_cond_final, V_values_final, alpha, 
                        horizon, policy, args) #, noise)
        total_num_safety_violations += num_safety_violations
        # epss = find_eps_lemma5(total_num_samples, total_num_safety_violations, delt)
        # print(f"Iteration {j+1}/{M}, Robust safety strength epsilon: {epss}")
        # print("N: ", total_num_samples, "k: ", total_num_safety_violations)
        if new_alpha == alpha:
            print(f"Converged at iteration {j+1}/{M}, alpha: {alpha:.4f}")
            break
        alpha = new_alpha
        print(f"Iteration {j+1}/{M}, New alpha: {alpha:.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time

    return alpha, total_time, state_traj_iterative, total_num_samples, \
            total_num_safety_violations

def robust_scenario_opt(N, alpha, delt, policy, dim, env, args,
                        rng, horizon, range_x, ego_setting, adv_setting):
    # Given N and given a volume (alpha level), find the safety level epsilon.
    init_cond_final, V_values_final = sample_init_cond(N, alpha, policy, rng, dim,
                                                            range_x, ego_setting, 
                                                            adv_setting)
    total_num_samples = len(init_cond_final)
    reach_avoid_measures, _ = reach_avoid_measure_vectorized(env, \
                            horizon, init_cond_final, V_values_final, policy, args)
    num_safety_violations = np.sum(reach_avoid_measures < 0)

    epss = find_eps_lemma5(total_num_samples, num_safety_violations, delt)

    return epss, total_num_samples, num_safety_violations

def visualize_set(alpha, epsilon_x, policy, slice = None):
    """
    Visualize the reach-avoid set.
    """
    # pass
    x = np.arange(-0.9, 0.9, 0.1)
    y = np.arange(-2.6, 0.0, 0.1)
    X, Y = np.meshgrid(x, y)
    size_x = len(x)
    size_y = len(y)
    V = np.zeros((X.shape))

    if slice is None:
        slice = {
            'ego_vx': 0.0,
            'ego_vy': 0.7,
            'ego_z': 0.0,
            'ego_vz': 0.0,
            'ad_x': 0.4,
            'ad_vx': 0.0,
            'ad_y': -2.2,
            'ad_vy': 0.3,
            'ad_z': 0.0,
            'ad_vz': 0.0
        }
    # slice of other states fixed
    ego_vx = slice['ego_vx']
    ego_vy = slice['ego_vy']
    ego_z = slice['ego_z']
    ego_vz = slice['ego_vz']

    ad_x = slice['ad_x']
    ad_vx = slice['ad_vx']
    ad_y = slice['ad_y']
    ad_vy = slice['ad_vy']
    ad_z = slice['ad_z']
    ad_vz = slice['ad_vz']

    # create one matrix of states
    H, W = X.shape
    states = np.empty((H, W, 12))
    states[:, :, 0] = X
    states[:, :, 1] = ego_vx
    states[:, :, 2] = Y
    states[:, :, 3] = ego_vy
    states[:, :, 4] = ego_z
    states[:, :, 5] = ego_vz
    states[:, :, 6] = ad_x
    states[:, :, 7] = ad_vx
    states[:, :, 8] = ad_y
    states[:, :, 9] = ad_vy
    states[:, :, 10] = ad_z
    states[:, :, 11] = ad_vz
    states_reshaped = states.reshape(-1, 12)

    V_values = evaluate_V_batch(states_reshaped, policy)
    V = V_values.reshape(H, W)
    V_flipped = np.flipud(V)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(V_flipped, annot=False, cmap=cm.coolwarm_r, alpha=0.9, ax=ax2, cbar=True,
                vmin=np.min(V_flipped), vmax=np.max(V_flipped))
    ax2.set_title('Scenario Iterative certified reach-avoid set', fontsize=16)
    ax2.set_xlabel('Ego x position', fontsize=14)
    ax2.set_ylabel('Ego y position', fontsize=14)
    
    x_V_interval = 30
    y_V_interval = 60

    x_ticks = np.arange(0, size_x, x_V_interval)
    y_ticks = np.arange(0, size_y, y_V_interval)
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)

    ax2.set_xticklabels(np.round(x[::x_V_interval], 2))
    ax2.set_yticklabels(np.round(y[::-y_V_interval]+0.02, 1))

    contours2 = ax2.contour((X+0.9)*10, (Y+2.6)*10, V_flipped, levels=[alpha], colors='black', linestyles='dashed')
    ax2.clabel(contours2, inline=True, fontsize=8, fmt="%.2f")

    # plt.show() #(block=True)
    # plt.pause(0.1)

    # save figure
    folder = "figs"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig2.savefig(os.path.join(folder, f"lin_scenario_opt_alpha_{alpha:.2f}.png"))

    print(f"Visualization saved for alpha: {alpha:.4f} to {os.path.join(folder, f'lin_scenario_opt_alpha_{alpha:.2f}.png')}")

    return



def main(visualize=False):
    args = get_args()
    env, policy = get_env_and_policy(args)

    confidence = 0.97
    eps = 1 - confidence  # epsilon
    delt = 1e-10 # delta
    M = 7 # max iterations
    horizon = 30

    alpha, total_time, state_traj_iterative, _, _ = \
        solve_iterative_method(env, eps, delt, M, horizon, policy, \
                            args, alpha_init=-np.inf)
    print(f"Final alpha: {alpha:.4f}, Total time: {total_time:.2f} seconds")
    if visualize:
        epsilon_x = 0.1 # coarseness of grid for visualization
        visualize_set(alpha, epsilon_x, policy)

if __name__ == "__main__":
    main(visualize=True)

