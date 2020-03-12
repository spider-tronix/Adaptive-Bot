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

ANGLE_DISCRETIZATION = 30
GOAL_DISTANCE = 0.2
DISTANCE_TO_WALK = 0.5
MAX_ANGLE = 30*(3.14/180)

emptyBuff = bytearray()   # used as parameter to remoteAPI function

class HexapodEnv(gym.Env):
    
    def __init__(self, clientID, faulty_joints, test=True):
        self.num_joints = 6
        self.clientID = clientID
        if faulty_joints != -1:
            self.faulty_joints = [int(joint) for joint in faulty_joints]
        else:
            self.faulty_joints = []
        self.action_map = {0:ANGLE_DISCRETIZATION, 1:-ANGLE_DISCRETIZATION}
        
        self.action_size = self.num_joints*2        # left, right
        self.state_size = self.num_joints   # Current Pos of each servo

        self.test = test
        self._setup_spaces()
        self._get_joint_handles()
        self._init_pos()



    def _get_joint_handles(self):
        self.hexapod_handle = sim.simxGetObjectHandle(self.clientID, 'hexapod', blocking)[1]     
        self.joint_handles = {} # get joint handles
        for i in range(6):
            for j in range(1):
                joint = f"hexa_joint{j+1}_{i}"
                self.joint_handles[joint] =  sim.simxGetObjectHandle(self.clientID, joint, blocking)[1]

        # list of 3 dicts
        self.links = []  # organise them according to link positions
        for j in range(3):
            handles = {}
            for i in range(6):
                joint = f"hexa_joint{j+1}_{i}"
                handles[joint] =  sim.simxGetObjectHandle(self.clientID, joint, blocking)[1]
            self.links.append(handles)
        


    def _init_pos(self):
        # Set to streaming mode
        sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, streaming)
        for handle in self.joint_handles.values():
            sim.simxGetJointPosition(self.clientID, handle, streaming)
        time.sleep(1)

        # Get the positions now
        self.init_pos = sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, buffer)[1]
        print("Hexapod init Pos(Relative to world): ", self.init_pos)
        
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
        result,collision_state,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(self.clientID,
                                                            "hexapod",
                                                            sim.sim_scripttype_childscript,
                                                            "reset_robot",[],[],[],
                                                            emptyBuff,blocking)

        for link, angle in zip(self.links, [0, -30, 120]):
            for handle in link.values():
                sim.simxSetJointTargetPosition(self.clientID, handle, angle*(3.14/180), oneshot)
                sim.simxSynchronousTrigger(self.clientID)
        
        self.prev_pos = sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, buffer)[1]
        self.state = np.ones(self.state_size)*120*(3.14/180)
        return self.state     # hope coppeliaSim set them properly


    def step(self, action):
        assert self.action_space.contains(action)
        
        joint = int(action/2)
        handle = list(self.joint_handles.values())[joint]
        parent_handle = list(self.links[1].values())[joint]
        value = self.action_map[action%2]
        
        if joint in self.faulty_joints:
            value = 0 
        
        cur_pos = sim.simxGetJointPosition(self.clientID,handle,buffer)[1]
        new_pos = cur_pos + value*(3.14/180)
        if np.abs(new_pos) <= MAX_ANGLE:   # Limiting it from rotating beyond |30| degree
            sim.simxSetJointTargetPosition(self.clientID,parent_handle,-45*(3.14/180),oneshot)
            sim.simxSynchronousTrigger(self.clientID)
            sim.simxSetJointTargetPosition(self.clientID,handle,new_pos,oneshot)
            sim.simxSynchronousTrigger(self.clientID)
            sim.simxSetJointTargetPosition(self.clientID,parent_handle,-30*(3.14/180),oneshot)
            sim.simxSynchronousTrigger(self.clientID)
            
            # update state, reward, done
            reward, done = self.calc_reward()
            self.state[joint] = new_pos
        else:
            reward = -1
            done = False

        return self.state, reward, done, {}
        

    def calc_reward(self):
        reward = 0
        done = False
        cur_pos = sim.simxGetObjectPosition(self.clientID, self.hexapod_handle, -1, buffer)[1]

        if cur_pos[2] - self.prev_pos[2] > 0.001:
            reward = 1
        self.prev_pos = cur_pos   # update prev pos

        if cur_pos[2] > DISTANCE_TO_WALK:
            done = True
            reward += 100        
        return reward, done

    
    