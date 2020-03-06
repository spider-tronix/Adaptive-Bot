import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))

import sim
# simRemoteApi.start(19999)

import argparse
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

def parse_joint_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault_joints", default=0, type=list, help="Faulty joints")
    return parser.parse_args()

def get_train_dir(parent_dir):
    i = 0
    while True:
        train_dir = os.path.join(parent_dir, f'Training_{i}')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            break
        i += 1
    return train_dir

def train(agent, env, writer, train_dir, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.996):
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
        print('\rEpisode {}\tScore: {}\tepsilon: {:.2f}'.format(i_episode, score, eps))

        writer.add_scalar("epsilon", eps, global_step=i_episode)
        writer.add_scalar("mean scores", np.mean(scores_window), global_step=i_episode)
        
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #save the agent's q-network for testing
            agent.save(train_dir, i_episode)
            print('-'*10, 'Agent saved', '-'*10)

    return scores


def print_info(info):
    print('-'*5,'\t', info, '\t', '-'*5)


def conect_and_load(port, ip):
    clientID = sim.simxStart(ip,port,True,True,5000,5) # Connect to CoppeliaSim
    scene_path = os.path.join('vrep-scene', 'poppy_two_target_pos_z.ttt')
    print('-'*5, 'Scene path:', scene_path, '-'*5)
    sim.simxLoadScene(clientID, scene_path, 0xFF, sim.simx_opmode_blocking)
    return clientID


if __name__ == "__main__":
    args = parse_joint_args()

    print('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    
    clientID=conect_and_load(19997, '127.0.0.1')
    
    print('Starting the simulation')
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    time.sleep(1)

    if clientID!=-1:
        print('Connected to remote API server')
        time.sleep(2)
        
        _, green_box = sim.simxGetObjectHandle(clientID, "Green_Box", sim.simx_opmode_blocking)
        _, red_box = sim.simxGetObjectHandle(clientID, "Red_Box", sim.simx_opmode_blocking)        
        env = PoppyEnv(clientID, args.fault_joints, green_box, red_box)
        
        eps_start=1.0
        eps_end=0.01
        eps_decay=0.996
        # EPS DECAY RATE CHANGED
        
        NUM_EPISODES = 1000    #number of episodes to train
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
        agent = Agent(env.state_size, env.action_size, SEED)

        # load trained agent
        # agent.load('Training_Files_2/Training_9/target_network_final')
        
        # logs
        writer = SummaryWriter(os.path.join(train_dir, 'summary'))
    
        # train
        time1 = datetime.datetime.now()
        train(agent, env, writer, train_dir, n_episodes=NUM_EPISODES, eps_decay=eps_decay)
        time2 = datetime.datetime.now()
        print_info(f'Time taken for {NUM_EPISODES} is {time2-time1}')

        # stopping the simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        print('Stopped the simulation')
        time.sleep(1)
        sim.simxFinish(clientID)

        #save the agent's q-network for testing
        agent.save(train_dir, 'final')
        print('Program finished')
