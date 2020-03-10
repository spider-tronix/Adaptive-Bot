import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))


import sim
import time

# simRemoteApi.start(19999)


print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
print("Client ID", clientID)

# Getting joint names
joints = [[] for _ in range(6)]
for i in range(6):
    for j in range(3):
        joints[i].append(f"hexa_joint{j+1}_{i}")
print("JOINTS", joints)

# Getting Handle
joint_handles = {}
for i in range(6):
    for j in range(3):
        joint = joints[i][j]
        joint_handles[joint] =  sim.simxGetObjectHandle(clientID, joint, sim.simx_opmode_blocking)[1]

# Getting initial positions
for joint, handle in joint_handles.items():
    print(joint, sim.simxGetJointPosition(clientID, handle, sim.simx_opmode_streaming)[1])

# define params
angle1 = 20
angle2 = 20
def raise_leg(clientID, leg_num):
    handle = joint_handles[joints[leg_num-1][1]]
    sim.simxSetJointTargetPosition(clientID,handle, angle1*(3.14/180), sim.simx_opmode_oneshot)
    time.sleep(0.01)
    

def lower_leg(clientID, leg_num):
    handle = joint_handles[joints[leg_num-1][1]]
    sim.simxSetJointTargetPosition(clientID,handle, -angle1*(3.14/180), sim.simx_opmode_oneshot)
    time.sleep(0.01)
    
def mv_forward_leg(clientID, leg_num):
    handle = joint_handles[joints[leg_num-1][0]]
    sim.simxSetJointTargetPosition(clientID,handle, angle2*(3.14/180), sim.simx_opmode_oneshot)
    time.sleep(0.01)
    

def mv_backward_leg(clientID, leg_num):
    handle = joint_handles[joints[leg_num-1][0]]
    sim.simxSetJointTargetPosition(clientID,handle, -angle2*(3.14/180), sim.simx_opmode_oneshot)
    time.sleep(0.01)
    
    
time.sleep(0.5)
flip = 0
if clientID != -1:
    print ('Connected to remote API server')

    # hexapod part
    driveBackStartTime=-99000
    motorSpeeds = []
    
    
    while (sim.simxGetConnectionId(clientID)!=-1):
    
        sensorTrigger=0
        if flip == 0:
            #raise legs 1, 3, 5
            raise_leg(clientID,1)
            raise_leg(clientID,3)
            raise_leg(clientID,5)
        
        elif flip == 1:
            mv_forward_leg(clientID,1)
            mv_forward_leg(clientID,3)
            mv_forward_leg(clientID,5)
            mv_backward_leg(clientID,2)
            mv_backward_leg(clientID,4)
            mv_backward_leg(clientID,6)
        
        elif flip == 2:
            lower_leg(clientID,1)
            lower_leg(clientID,3)
            lower_leg(clientID,5)
        
        elif flip == 3:
            raise_leg(clientID,2)
            raise_leg(clientID,4)
            raise_leg(clientID,6)
        
        elif flip == 4:
            mv_backward_leg(clientID,1)
            mv_backward_leg(clientID,3)
            mv_backward_leg(clientID,5)
            mv_forward_leg(clientID,2)
            mv_forward_leg(clientID,4)
            mv_forward_leg(clientID,6)
        
        elif flip == 5:
            lower_leg(clientID,2)
            lower_leg(clientID,4)
            lower_leg(clientID,6)
        
        flip += 1
        if flip > 5:
            flip = 0

        #Stabilize the last joint in each leg
        for leg_num in range(6):    
            handle = joint_handles[joints[leg_num][2]]
            sim.simxSetJointTargetPosition(clientID,handle, 90*(3.14/180), sim.simx_opmode_oneshot)
    

        #extApi_sleepMs(5)
        time.sleep(0.1)
        print("Moved")
    
    sim.simxFinish(clientID)


