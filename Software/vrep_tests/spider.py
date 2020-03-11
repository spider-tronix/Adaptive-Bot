
import os
import sys
from os.path import abspath, dirname
sys.path.append(os.path.abspath('..'))

import time

import sim

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
print(clientID)
_, handle1 = sim.simxGetObjectHandle(clientID, "left_revolute", sim.simx_opmode_blocking)
_, handle2 = sim.simxGetObjectHandle(clientID, "right_revolute", sim.simx_opmode_blocking)
_, handle3 = sim.simxGetObjectHandle(clientID, "middle_revolute", sim.simx_opmode_blocking)
print(sim.simxGetJointPosition(clientID, handle1, sim.simx_opmode_oneshot_wait))



sim.simxSetJointTargetPosition(clientID, handle1, 60*(3.14/180), sim.simx_opmode_oneshot_wait)
time.sleep(2)
sim.simxSetJointTargetPosition(clientID, handle1, 90*(3.14/180), sim.simx_opmode_oneshot_wait)
time.sleep(2)
sim.simxSetJointTargetPosition(clientID, handle1, 60*(3.14/180), sim.simx_opmode_oneshot_wait)
time.sleep(2)

sim.simxFinish(-1) # just in case, close all opened connections
