import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))

import sim
# simRemoteApi.start(19999)

import random
import time
import datetime
import pickle
from collections import deque
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from dqn_agent import Agent
from Poppy_Ergo.env import PoppyEnv


def get_joint_handles(clientID):
    _, m1 = sim.simxGetObjectHandle(clientID, "m1", sim.simx_opmode_blocking)
    _, m2 = sim.simxGetObjectHandle(clientID, "m2", sim.simx_opmode_blocking)
    _, m3 = sim.simxGetObjectHandle(clientID, "m3", sim.simx_opmode_blocking)
    _, m4 = sim.simxGetObjectHandle(clientID, "m4", sim.simx_opmode_blocking)
    _, m5 = sim.simxGetObjectHandle(clientID, "m5", sim.simx_opmode_blocking)
    _, m6 = sim.simxGetObjectHandle(clientID, "m6", sim.simx_opmode_blocking)
    return [m1, m2, m3, m4, m5, m6]


def print_init_pos(clientID, joint_handles):
    # print positions
    for i, handle in enumerate(joint_handles):
        print(f"m{i}_pos", sim.simxGetJointPosition(clientID, handle, sim.simx_opmode_streaming)[1])


def get_train_dir(parent_dir):
    i = 0
    while True:
        train_dir = os.path.join(parent_dir, f'Training_{i}')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            break
        i += 1
    return train_dir

def train(agent, env, writer, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        time.sleep(0.5)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                time.sleep(0.5)
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        writer.add_scalar("epsilon", eps, global_step=i_episode)
        writer.add_scalar("mean scores", np.mean(scores_window), global_step=i_episode)
        
        # if i_episode % 100 == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #     break
    return scores



def conect_and_load(port, ip):

    clientID = sim.simxStart(ip,port,True,True,5000,5) # Connect to CoppeliaSim
    scene_path = os.path.join('..', 'vrep-scene', 'poppy_two_target_scene.ttt')
    print('-'*5, 'Scene path:', scene_path, '-'*5)
    sim.simxLoadScene(clientID, scene_path, 0xFF, sim.simx_opmode_blocking)
    return clientID


if __name__ == "__main__":
    
    print ('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    
    clientID=conect_and_load(19997, '127.0.0.1')
    
    print('-'*10, 'Starting the simulation', '-'*10)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    time.sleep(1)

    if clientID!=-1:
        print ('Connected to remote API server')
        time.sleep(2)

        joint_handles = get_joint_handles(clientID)
        env = PoppyEnv(clientID, joint_handles)
        
        # reset positions
        state = env.reset()
        print_init_pos(clientID, joint_handles)
        
        # Hyperparams
        # HIDDEN_UNITS = (64, 32)
        # NETWORK_LR = 0.01
        eps_start=1.0
        eps_end=0.01
        eps_decay=0.995
        NUM_EPISODES = 1    #number of episodes to train
        MAX_T = 1000
        SEED = 0

        parent_dir = 'Training_Files'
        train_dir = get_train_dir(parent_dir)
        
        
        # save hyperparameters
        hyperparams = { 
                        'num_episodes': NUM_EPISODES,
                        'max_t': MAX_T,
                        'seed': SEED,
                        'eps_start': eps_start,
                        'eps_end': eps_end,
                        'eps_decay': eps_decay
        }

        with open(f'{train_dir}/params.pickle', 'wb') as f:
            pickle.dump(hyperparams, f)

        # Initialise the agents
        action_size = 6 # left, right, no movement
        state_size = 3 # 3 Degrees of freedom
        agent = Agent(state_size, action_size, SEED)

        # logs
        writer = SummaryWriter(os.path.join(train_dir, 'summary'))
    
        # train
        train(agent, env, writer, n_episodes=NUM_EPISODES)
        # stopping the simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        print('-'*10, 'Stopped the simulation', '-'*10)
        time.sleep(2)
        sim.simxFinish(clientID)

        #save the agent's q-network for testing
        agent.save(train_dir)
        print('-'*10, 'Program finished', '-'*10)
