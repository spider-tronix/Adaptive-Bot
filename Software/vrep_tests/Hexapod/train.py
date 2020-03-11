import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))

import sim


import argparse
import random
import time
import datetime
import pickle
from collections import deque
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from Hexapod.dqn_agent import Agent
from Hexapod.hexapod_env import HexapodEnv

def parse_joint_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=19997, type=int, help="Port at which CoppeliaSim is running")
    parser.add_argument("--ip", default='127.0.0.1', type=str, help="IP to connect to CoppeliaSim")
    parser.add_argument("--num_episodes", default=1000, type=int, help="Number of training episodes")
    parser.add_argument("--max_t", default=10000, type=int, help="Time steps per episode")
    parser.add_argument("--eps_decay", default=0.996, type=float, help="Controls exploration")
    parser.add_argument("--faulty_joints", default=-1, nargs='*', help="Faulty joints")
    parser.add_argument("--load_dir", default=None, type=str, help="Path to load model")
    # parser.add_argument("--num_layers", default=3, type=int, help="Number of Layers in Q-Network")
    # parser.add_argument("--hidden_size", default=[256, 128, 64], nargs='*', type=int, help="Number of activation units in hidden layers")
    parser.add_argument("--seed", default=0, type=int, help="To Maintain Reproducibility")
    return parser.parse_args()


def get_train_dir(faulty_joints):
    # Make Parent dir according to faulty joints
    if faulty_joints == -1:
        parent_dir = os.path.join('Training_Files', 'All_Joints')
    else:
        parent_dir = os.path.join('Training_Files', 'Joints_' + '_'.join(list(faulty_joints)))
    if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
    # Create training dir inside parent directory
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
        time.sleep(0.1)
        score = 0
        time1 = datetime.datetime.now()
        for _ in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                time.sleep(0.2)
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tScore: {}\tepsilon: {:.2f}\t {}'.format(i_episode, score, eps, datetime.datetime.now() - time1))

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
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID = sim.simxStart(ip,port,True,True,5000,5) # Connect to CoppeliaSim
    scene_path = os.path.join('vrep_scene', 'hexapod_scene.ttt')
    print('-'*5, 'Scene path:', scene_path, '-'*5)
    sim.simxLoadScene(clientID, scene_path, 0xFF, sim.simx_opmode_blocking)
    return clientID


if __name__ == "__main__":
    args = parse_joint_args()

    print('Program started')
    clientID=conect_and_load(args.port, args.ip)
    
    print('Starting the simulation')
    sim.simxSynchronous(clientID, 1)   # enable synchoronous mode
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    sim.simxSynchronousTrigger(clientID)
    time.sleep(1)
    
    if clientID!=-1:
        print('Connected to remote API server')
        time.sleep(1)
               
        env = HexapodEnv(clientID, args.faulty_joints)
        
        print_info(f"Eps Decay Rate is {args.eps_decay}")

        train_dir = get_train_dir(args.faulty_joints)
        
        # save hyperparameters
        hyperparams = { 
                        'num_episodes': args.num_episodes,
                        'max_t': args.max_t,
                        'seed': args.seed,
                        'eps_decay': args.eps_decay,
        }

        with open(f'{train_dir}/params.pickle', 'wb') as f:
            pickle.dump(hyperparams, f)

        # Initialise the agents
        agent = Agent(env.state_size, env.action_size, args.seed)

        # load trained agent
        if args.load_dir is not None:
            agent.load(args.load_dir)
        
        # logs
        writer = SummaryWriter(os.path.join(train_dir, 'summary'))
    
        # train
        train(agent, env, writer, train_dir, n_episodes=args.num_episodes, max_t=args.max_t, eps_decay=args.eps_decay)
        
        # stopping the simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        print('Stopped the simulation')
        time.sleep(1)
        sim.simxFinish(clientID)

        #save the agent's q-network for testing
        agent.save(train_dir, 'final')
        print('Program finished')
