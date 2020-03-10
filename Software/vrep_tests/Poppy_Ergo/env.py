import gym
from gym import spaces
import numpy as np
import random

import sim
import time
from math import sin, cos, radians, pi

# RemoteAPI constants
streaming = sim.simx_opmode_streaming
blocking = sim.simx_opmode_blocking
buffer = sim.simx_opmode_buffer
oneshot = sim.simx_opmode_oneshot

ANGLE_DISCRETIZATION = 10
BOX_DISTANCE = 0.2

class PoppyEnv(gym.Env):
    
    def __init__(self, clientID, faulty_joints):
        self.num_joints = 6
        self.clientID = clientID
        if faulty_joints != -1:
            self.faulty_joints = [int(joint) for joint in faulty_joints]
        else:
            self.faulty_joints = []
        self.action_map = {0:0, 1:ANGLE_DISCRETIZATION, 2:-ANGLE_DISCRETIZATION}
        
        self.action_size = self.num_joints*3        # 0, left, right
        self.state_size = self.num_joints + 3       # 6 DOF + (x, y, z) box distance cords

        self._setup_spaces()
        self._get_joint_handles()
        self._init_pos()

    def _get_joint_handles(self):
        self.green_box = sim.simxGetObjectHandle(self.clientID, "Green_Box", blocking)[1]
        self.red_box = sim.simxGetObjectHandle(self.clientID, "Red_Box", blocking)[1]  
        self.lamp_end = sim.simxGetObjectHandle(self.clientID, "lamp_end", blocking)[1]  
        
        self.joint_handles = [] # get joint handles
        for i in range(1, self.num_joints+1):
            _, handle = sim.simxGetObjectHandle(self.clientID, f"m{i}", blocking)
            self.joint_handles.append(handle)
        
    def _init_pos(self):
        # print positions
        for i, handle in enumerate(self.joint_handles, start=1):
            print(f"m{i}_pos", sim.simxGetJointPosition(self.clientID, handle, streaming)[1])
        print(f"Faulty joints {self.faulty_joints}")
        print("Distance from Green Box: " ,sim.simxGetObjectPosition(self.clientID, self.lamp_end, self.green_box, streaming)[1])

    def _setup_spaces(self):
        self.action_space = spaces.Discrete(self.action_size)
        # self.observation_space = spaces.Box(low=np.array([-5.0, -5.0, -2.62, -2.62, -2.62, -2.62, -2.62, -2.62]), 
        #                                     high=np.array([5.0, 5.0, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62]),
        #                                     dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        
        joint = int(action/3)
        handle = self.joint_handles[joint]
        value = self.action_map[action%3]
        
        if joint in self.faulty_joints:
            value = 0 

        #for angle, handle in zip(action, self.joint_handles)        
        cur_pos = sim.simxGetJointPosition(self.clientID,handle,buffer)[1]
        sim.simxSetJointTargetPosition(self.clientID,handle,cur_pos+value*(3.14/180),oneshot)

        # give reward for touching the box
        reward, done = self.detect_collision()

        box_distance = sim.simxGetObjectPosition(self.clientID, self.lamp_end, self.green_box, buffer)[1]
        state = [d for d in box_distance]
        for handle in self.joint_handles:
            state.append(sim.simxGetJointPosition(self.clientID, handle, buffer)[1])

        return np.array(state), reward, done, {}
    
    def detect_collision(self):
        reward = 0
        done = False
        
        emptyBuff = bytearray()
        result,collision_state,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(self.clientID,
                                                            "remoteApiCommandServer",
                                                            sim.sim_scripttype_childscript,
                                                            "collision_with_box",[],[],[],
                                                            emptyBuff,blocking)

        if result == sim.simx_return_ok:
            if collision_state[0] == 1:  # Collided with green box
                reward = 100
                done = True
            elif collision_state[1] == 1:  # collided with red box
                reward = -100
                done = True
            return reward, done
        else:
            print("Error in Detecting Collision")
            quit()
        
  
  
    def reset(self):        
        for handle in self .joint_handles:
            sim.simxSetJointTargetPosition(self.clientID, handle, 0, blocking)
        # Change Box positions
        rand_angle = 0 # radians(random.randint(0, 359))        
        box_x, box_y = (BOX_DISTANCE*cos(rand_angle), BOX_DISTANCE*sin(rand_angle))
        sim.simxSetObjectPosition(self.clientID, self.green_box, self.joint_handles[0], (box_x, box_y, 0), oneshot)
        sim.simxSetObjectPosition(self.clientID, self.red_box, self.joint_handles[0], (-box_x, -box_y, 0), oneshot)
        
        box_distance = sim.simxGetObjectPosition(self.clientID, self.lamp_end, self.green_box, buffer)[1]
        state = [d for d in box_distance]
        for handle in self.joint_handles:
            state.append(sim.simxGetJointPosition(self.clientID, handle, buffer)[1])
        return np.array(state)

    