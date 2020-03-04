import gym
from gym import spaces
import numpy as np

import sim
import time

#### action map ########
# 0 - no movement
# 1 - move left by +ANGLE_DISCRETIZATION degree
# 2 - move right by -ANGLE_DISCRETIZATION degree

#### possible spaces ########
# -2.6179075241088867
# -1.5772767066955566
# -2.617981433868408

ANGLE_DISCRETIZATION = 5

class PoppyEnv(gym.Env):
    
    def __init__(self, clientID, servo_handles):
        self.clientID = clientID
        self.servo_handles = servo_handles
        self.action_map = {0:0, 1:ANGLE_DISCRETIZATION, 2:-ANGLE_DISCRETIZATION}
        self._setup_spaces()

    def _setup_spaces(self):
        # action space = NUM_SERVOS * 3
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([-2.62, -1.58, -2.62]), 
                                            high=np.array([2.62, 1.58, 2.62]),
                                            dtype=np.float32)


    def step(self, action):
        assert self.action_space.contains(action)
        
        handle = self.servo_handles[int(action/3)]
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
        state = []
        for handle in self.servo_handles:
            state.append(sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_buffer)[1])
        return np.array(state)

    def reset(self):
        for handle in self.servo_handles:
            sim.simxSetJointTargetPosition(self.clientID, handle, 0, 
                                            sim.simx_opmode_oneshot)
        return self.get_state()
        