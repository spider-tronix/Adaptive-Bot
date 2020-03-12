#!/usr/bin/env python
# license removed for brevity

import rospy 
import time
from std_msgs.msg import String,Float64

import gym
from gym import spaces
import numpy as np
import random

class Aragog:
    def __init__(self):
	self.left_revolute_bottom_angle = 10 * 3.14 / 180.0
    	self.left_revolute_top_angle = 80 * 3.14 / 180.0
    	self.right_revolute_bottom_angle = 10 * 3.14 / 180.0
    	self.right_revolute_top_angle = 80 * 3.14 / 180.0
    	self.middle_revolute_left_angle = 0 * 3.14 / 180.0
    	self.middle_revolute_right_angle = 30 * 3.14 / 180.0
        self.ros_init()

    def ros_init(self):
        rospy.init_node('Aragog_script',anonymous=True)
        self.left_revolute = rospy.Publisher("Aragog/joint1_position_controller/command",Float64)
        self.middle_revolute = rospy.Publisher("Aragog/joint2_position_controller/command",Float64)
        self.right_revolute = rospy.Publisher("Aragog/joint3_position_controller/command",Float64)

    
    def move_forward_init(self):
        self.middle_revolute.publish(self.middle_revolute_left_angle)
        time.sleep(1)
        self.left_revolute.publish(self.left_revolute_top_angle)
        time.sleep(1)
        self.right_revolute.publish(self.right_revolute_top_angle)
        time.sleep(1)

    def move_forward(self):
        self.middle_revolute.publish(self.middle_revolute_left_angle)
        time.sleep(2)
        self.left_revolute.publish(self.left_revolute_top_angle)
        time.sleep(1)
        self.right_revolute.publish(self.right_revolute_top_angle)
        time.sleep(3)

        self.middle_revolute.publish(self.middle_revolute_right_angle)
        time.sleep(2)
        self.left_revolute.publish(self.left_revolute_bottom_angle)
        time.sleep(0.5)
        self.right_revolute.publish(self.right_revolute_top_angle)
        time.sleep(3)

ANGLE_DISCRETIZATION = 5
GOAL_DISTANCE = 0.2
DISTANCE_TO_WALK = 0.8
MAX_ANGLE = 50 * 3.14 / 180.0

class AragogEnv(gym,Env):

    def __init__ ( self,test = True):
        self.num_joints = 3
        self.action_map = {0:ANGLE_DISCRETIZATION, 1:-ANGLE_DISCRETIZATION}

        self.action_size = self.num_joints*2
        self.state_size = self.num_joints

        self.test = test
        self._setup_spaces()

        self._get_handles()


    def _get_handles(self):
        rospy.init_node('Aragog_script',anonymous=True)
        self.left_revolute = rospy.Publisher("Aragog/joint1_position_controller/command",Float64)
        self.middle_revolute = rospy.Publisher("Aragog/joint2_position_controller/command",Float64)
        self.right_revolute = rospy.Publisher("Aragog/joint3_position_controller/command",Float64)

    def step(self,action):
        cur_pos = # get current joint pos
        new_pos = cur_pos + self.action_map(action[0])*(3.14/180)
        self.middle_revolute.publish(new_pos)
        time.sleep(0.5)
        
        cur_pos = # get current joint pos2
        new_pos = cur_pos + self.action_map(action[0])*(3.14/180)
        
        self.left_revolute.publish(new_pos)
        time.sleep(0.5)
        
        cur_pos = # get current joint pos
        new_pos = cur_pos + self.action_map(action[0])*(3.14/180)
        
        self.right_revolute.publish(new_pos)
        time.sleep(0.5)

        new_state = # get new joint angles
        reward = # distance from origin * some_constant
        done = # if reached goal
        return distance_from_origin

    def reset(self):
        #make the bot go to the centre and set the angles to zero




if __name__ == '__main__':
    Aragog = Aragog()

    Aragog.move_forward_init()

    while True:
        Aragog.move_forward()

        



