import gym
from gym import spaces
import numpy as np
import random

import sim
import time
from math import sin, cos, radians, pi

#### action map ########
# 0 - no movement
# 1 - move left by 1 degree
# 2 - move right by 1 degree

#### possible spaces ########
# -2.6179075241088867
# -1.5772767066955566
# -2.617981433868408

ANGLE_DISCRETIZATION = 5

class PoppyEnv(gym.Env):
    
    def __init__(self, clientID, green_box, red_box):
        self.num_joints = 6
        self.clientID = clientID
        self.green_box = green_box
        self.red_box = red_box
        self.action_map = {0:0, 1:ANGLE_DISCRETIZATION, 2:-ANGLE_DISCRETIZATION}
        
        self.action_size = self.num_joints*3        # 0, left, right
        self.state_size = self.num_joints + 2       # 6 DOF + (x, y) box cords

        self._setup_spaces()
        self._get_joint_handles()
        self._init_pos()

    def _get_joint_handles(self):
        self.joint_handles = []
        for i in range(1, self.num_joints+1):
            _, handle = sim.simxGetObjectHandle(self.clientID, f"m{i}", sim.simx_opmode_blocking)
            self.joint_handles.append(handle)
        

    def _init_pos(self):
        # print positions
        for i, handle in enumerate(self.joint_handles, start=1):
            print(f"m{i}_pos", sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_streaming)[1])


    def _setup_spaces(self):
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(low=np.array([-5.0, -5.0, -2.62, -2.62, -2.62, -2.62, -2.62, -2.62]), 
                                            high=np.array([5.0, 5.0, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62]),
                                            dtype=np.float32)

    def step(self, action):
            assert self.action_space.contains(action)
            
            handle = self.joint_handles[int(action/3)]
            value = self.action_map[action%3]

            #for angle, handle in zip(action, self.joint_handles)        
            cur_pos = sim.simxGetJointPosition(self.clientID,handle,sim.simx_opmode_buffer)[1]
            sim.simxSetJointTargetPosition(self.clientID,handle,cur_pos+value*(3.14/180),sim.simx_opmode_oneshot)

            # give reward for make it straight
            reward, done = self.detect_collision()
            return self.get_state(), reward, done, {}
    
    def detect_collision(self):
        reward = 0
        done = False
        
        emptyBuff = bytearray()
        result,collision_state,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(self.clientID,
                                                            "remoteApiCommandServer",
                                                            sim.sim_scripttype_childscript,
                                                            "collision_with_box",[],[],[],
                                                            emptyBuff,sim.simx_opmode_blocking)

        if result == sim.simx_return_ok:
            if collision_state[0] == 1:  # Collided with green box
                reward = 100
                done = True
            elif collision_state[1] == 1:  # collided with red box
                reward = -100
                done = True
            return reward, done
        else:
            raise NotImplementedError
        
    def get_state(self):
        state = [self.green_box_cords[0], self.green_box_cords[1]]
        # Add servo angles
        for handle in self.joint_handles:
            state.append(sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_buffer)[1])
        #print("state", state)
        return np.array(state)

    def reset(self):
        for handle in self.joint_handles:
            sim.simxSetJointTargetPosition(self.clientID, handle, 0, 
                                            sim.simx_opmode_oneshot)
        # Change Box positions
        rand_angle = random.randint(0, 359) 
        x, y = self.get_new_pos(rand_angle)
        sim.simxSetObjectPosition(self.clientID, self.green_box, self.joint_handles[0], 
                                    (x, y, 0), sim.simx_opmode_oneshot)
        sim.simxSetObjectPosition(self.clientID, self.red_box, self.joint_handles[0], 
                                    (-x, -y, 0), sim.simx_opmode_oneshot)
        # set state variable
        self.green_box_cords = (x, y)
        return self.get_state()
    
    @staticmethod
    def get_new_pos(angle):
        r = 0.20
        theta_rad = radians(angle)
        return (r*cos(theta_rad), r*sin(theta_rad))