import gym
from gym import spaces
import numpy as np
import random

import sim
import simConst
import time
from math import sin, cos, radians, pi

# RemoteAPI constants
streaming = sim.simx_opmode_streaming
blocking = sim.simx_opmode_blocking
buffer = sim.simx_opmode_buffer
oneshot = sim.simx_opmode_oneshot

ANGLE_DISCRETIZATION = 10
GOAL_DISTANCE = 0.2
DISTANCE_TO_WALK = 0.5

class HexapodEnv(gym.Env):
    
    def __init__(self, clientID, faulty_joints, test=True):
        self.num_joints = 12
        self.clientID = clientID
        if faulty_joints != -1:
            self.faulty_joints = [int(joint) for joint in faulty_joints]
        else:
            self.faulty_joints = []
        self.action_map = {0:0, 1:ANGLE_DISCRETIZATION, 2:-ANGLE_DISCRETIZATION}
        
        self.action_size = self.num_joints*3        # 0, left, right
        self.state_size = self.num_joints   # Current Pos of each servo

        self.test = test
        self._setup_spaces()
        self._get_joint_handles()
        self._init_pos()

    def _get_joint_handles(self):
        self.hexapod_handle = sim.simxGetObjectHandle(self.clientID, 'hexapod', blocking)[1] 
        print("Hexapod handle", self.hexapod_handle)       
        self.joint_handles = {} # get joint handles
        for i in range(6):
            for j in range(2):
                joint = f"hexa_joint{j+1}_{i}"
                self.joint_handles[joint] =  sim.simxGetObjectHandle(self.clientID, joint, blocking)[1]

    def _init_pos(self):
        # Set to streaming mode
        sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, streaming)
        for handle in self.joint_handles.values():
            sim.simxGetJointPosition(self.clientID, handle, streaming)
        time.sleep(1)

        # Get the positions now
        self.init_pos = sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, buffer)[1]
        print("Hexapod init Pos(Relative to world): ", self.init_pos)
        #self.init_joint_pos = []
        # for joint, handle in self.joint_handles.items():
        #     pos = sim.simxGetJointPosition(self.clientID, handle, buffer)[1]
        #     self.init_joint_pos.append(pos)
        #     print(joint, 'Joint Pos: ', pos)
        
        # Make the faulty joints invisible
        if self.test:
            for joint in self.faulty_joints:
                handle = list(self.joint_handles.values())[joint]
                sim.simxSetModelProperty(self.clientID, handle, simConst.sim_modelproperty_not_visible, blocking)
        print(f"Faulty joints {self.faulty_joints}")

        
    def _setup_spaces(self):
        self.action_space = spaces.Discrete(self.action_size)
        # self.observation_space = spaces.Box(low=np.array([-5.0, -5.0, -2.62, -2.62, -2.62, -2.62, -2.62, -2.62]), 
        #                                     high=np.array([5.0, 5.0, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62]),
        #                                     dtype=np.float32)

    def reset(self):        
        
        # reset objects dynamically in lua script
        emptyBuff = bytearray()
        result,collision_state,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(self.clientID,
                                                            "hexapod",
                                                            sim.sim_scripttype_childscript,
                                                            "reset_robot",[],[],[],
                                                            emptyBuff,blocking)

        for handle in self.joint_handles.values():
            sim.simxSetJointTargetPosition(self.clientID, handle, 0, blocking)
        self.prev_pos = sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, buffer)[1]
        return np.zeros(self.state_size)     # hope coppeliaSim set them properly


    def step(self, action):
        assert self.action_space.contains(action)
        
        joint = int(action/3)
        handle = list(self.joint_handles.values())[joint]
        value = self.action_map[action%3]
        
        if joint in self.faulty_joints:
            value = 0 
        cur_pos = sim.simxGetJointPosition(self.clientID,handle,buffer)[1]
        sim.simxSetJointTargetPosition(self.clientID,handle,cur_pos+value*(3.14/180),oneshot)

        reward, done = self.calc_reward()
        state = []
        for handle in self.joint_handles.values():
            state.append(sim.simxGetJointPosition(self.clientID, handle, buffer)[1])

        return np.array(state), reward, done, {}
    

    def calc_reward(self):
        reward = 0
        done = False
        cur_pos = sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, buffer)[1]

        if cur_pos[2] - self.prev_pos[2] > 0.001:
            reward = 1
        elif cur_pos[2] - self.prev_pos[2] < -0.001:
            reward = -1
        self.prev_pos = cur_pos   # update prev pos

        if cur_pos[2] > DISTANCE_TO_WALK:
            done = True
            reward += 100        
        return reward, done

    
    