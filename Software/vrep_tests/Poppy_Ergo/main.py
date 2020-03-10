import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))

import sim
# simRemoteApi.start(19999)
import pickle
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
        for _ in range(max_t):
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
                                                    "\n2.Faults in Joint 3" \
                                                    "\n3.Faults in Joint 2 and 4"
                                                    "\n4.Faults in Joint 1, 3 and 5"\
                                                    "\n5.Faults in Joints 1, 2 and 4\n"))

        
        if broken_links == 1:
            model_dir = "All_Joints"
            faulty_joints = -1
        elif broken_links == 2:
            model_dir = "Joints_3"
            faulty_joints = [3] 
        elif broken_links == 3:
            model_dir = "Joints_2_4"
            faulty_joints = [2, 4]
        elif broken_links == 4:
            model_dir = "Joints_1_3_5"
            faulty_joints = [1,3,5] 
        elif broken_links == 5:
            model_dir = "Joints_1_2_4"
            faulty_joints = [1,2,4] 
        else:
            print("Choose between given options only")
            quit()

        model_path = os.path.join("Trained_Agents", model_dir, "target_network_final")
        if not os.path.exists(model_path):
            print("Agent not trained for chose configuration")
            quit()
        return faulty_joints, model_path
    elif robot == 2:
        print("Hexapod not implemented")
        quit()
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
    
    faulty_joints, model_dir = open_scene()

    clientID=conect_and_load(19997, '127.0.0.1')
    
    print('-'*10, 'Starting the simulation', '-'*10)
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    time.sleep(1)


    if clientID!=-1:
        print ('Connected to remote API server')
        time.sleep(2)

        env = PoppyEnv(clientID, faulty_joints)
        
        with open(f'{model_dir}/params.pickle', 'rb') as f: 
            params = pickle.load(f)

        agent = Agent(env.state_size, env.action_size, params.num_layers, 
                                        params.hidden_size, params.seed)

        # load agent q-network
        agent.load(model_dir)
        
        # test
        test(agent, env, n_episodes=1)
         
        # stopping the simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        print('Stopped the simulation')
        time.sleep(1)
        sim.simxFinish(clientID)
        print('Program finished')

        
