import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))

import sim
# simRemoteApi.start(19999)

import datetime
import time
import numpy as np

import torch

from dqn_agent import Agent
from Poppy_Ergo.env import PoppyEnv



def test(agent, env, n_episodes=10, max_t=1000):
    
    scores = []                        # list containing scores from each episode
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        time.sleep(0.5)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps=0)
            print(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                time.sleep(0.5)
                break 
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score))
    print('--Mean Score {:.2f}--: '.format(np.mean(scores)))
    return scores



def print_info(info):
    print('-'*5,'\t', info, '\t', '-'*5)


def open_scene():
    robot = int(input("Choose Robot to test:\n1.Poppy Ergo (Hand Robot)\n2.Hexapod\n"))
    if robot == 1:  # Poppy Ergo
        broken_links = int(input("Choose Configuration:\n1.All joints working" \
                                                    "\n2.Faults in Joint 2 and 3" \
                                                    "\n3.Faults in Joint 4 and 5\n"))
        if broken_links == 1:
            return robot, 0
        elif broken_links == 2:
            return robot, [2,3]
        elif broken_links == 3:
            return robot [4,5]
        else:
            print("Choose between given options only")
            quit()
    elif robot == 2:
        print("Hexapod not implemented")
    else:
        print("Choose between 1 and 2 only")
        quit()

def conect_and_load(port, ip):
    clientID = sim.simxStart(ip,port,True,True,5000,5) # Connect to CoppeliaSim
    scene_path = os.path.join('vrep-scene', 'poppy_two_target_pos_z.ttt')
    print('-'*5, 'Scene path:', scene_path, '-'*5)
    sim.simxLoadScene(clientID, scene_path, 0xFF, sim.simx_opmode_blocking)
    return clientID


if __name__ == "__main__":
    
    print_info('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    
    robot, broken_links = open_scene()

    clientID=conect_and_load(19998, '127.0.0.2')
    
    print('-'*10, 'Starting the simulation', '-'*10)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    time.sleep(1)


    if clientID!=-1:
        print ('Connected to remote API server')
        time.sleep(2)

        _, green_box = sim.simxGetObjectHandle(clientID, "Green_Box", sim.simx_opmode_blocking)
        _, red_box = sim.simxGetObjectHandle(clientID, "Red_Box", sim.simx_opmode_blocking)        
        env = PoppyEnv(clientID, broken_links, green_box, red_box)
        
        agent = Agent(env.state_size, env.action_size, seed=0)

        # load agent q-network
        agent.load('Training_Files/Training_1/target_network_50')
        
        # test
        test(agent, env, n_episodes=5)
         
        # stopping the simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        print('Stopped the simulation')
        time.sleep(1)
        sim.simxFinish(clientID)
        print('Program finished')

        
