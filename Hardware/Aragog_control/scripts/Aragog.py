#!/usr/bin/env python
# license removed for brevity

import rospy 
import time
from std_msgs.msg import String,Float64

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

if __name__ == '__main__':
    Aragog = Aragog()

    Aragog.move_forward_init()

    while True:
        Aragog.move_forward()

        



