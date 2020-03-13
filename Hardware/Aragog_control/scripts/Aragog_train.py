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
        cur_pos = # get current joint pos 0
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
	pass

