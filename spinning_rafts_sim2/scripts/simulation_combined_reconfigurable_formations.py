"""
This is for the simulation of many rafts
The maximum characters per line is set to be 120.
"""
# import glob
import os
import sys
import shelve
import platform
import datetime

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
# from scipy.integrate import RK45
from scipy.integrate import solve_ivp
from scipy.spatial import Voronoi as scipyVoronoi
# import scipy.io
from scipy.spatial import distance as scipy_distance

parallel_mode = 0

if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
    projectDir = "D:\\simulationFolder\\spinning_rafts_sim2"
elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
    projectDir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
else:
    projectDir = os.getcwd()

if projectDir != os.getcwd():
    os.chdir(projectDir)

if parallel_mode == 1:
    import functions_spinning_rafts as fsr
else:
    import scripts.functions_spinning_rafts as fsr

scriptDir = os.path.join(projectDir, "scripts")
capSym6Dir = os.path.join(projectDir, '2019-05-13_capillaryForceCalculations-sym6')
capSym4Dir = os.path.join(projectDir, '2019-03-29_capillaryForceCalculations')
dataDir = os.path.join(projectDir, 'data')
if not os.path.isdir(dataDir):
    os.mkdir('data')


# %% load capillary force and torque
os.chdir(capSym6Dir)

shelveName = 'capillaryForceAndTorque_sym6'
shelveDataFileName = shelveName + '.dat'
listOfVariablesToLoad = ['eeDistanceCombined', 'forceCombinedDistancesAsRowsAll360',
                         'torqueCombinedDistancesAsRowsAll360']

if not os.path.isfile(shelveDataFileName):
    print('the capillary data file is missing')

tempShelf = shelve.open(shelveName)
capillaryEEDistances = tempShelf['eeDistanceCombined']  # unit: m
capillaryForcesDistancesAsRowsLoaded = tempShelf['forceCombinedDistancesAsRowsAll360']  # unit: N
capillaryTorquesDistancesAsRowsLoaded = tempShelf['torqueCombinedDistancesAsRowsAll360']  # unit: N.m

# further data treatment on capillary force profile
# insert the force and torque at eeDistance = 1um as the value for eedistance = 0um.
capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
capillaryForcesDistancesAsRows = np.concatenate(
    (capillaryForcesDistancesAsRowsLoaded[:1, :], capillaryForcesDistancesAsRowsLoaded), axis=0)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRowsLoaded[:1, :], capillaryTorquesDistancesAsRowsLoaded), axis=0)

# add angle=360, the same as angle = 0
capillaryForcesDistancesAsRows = np.concatenate(
    (capillaryForcesDistancesAsRows, capillaryForcesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRows, capillaryTorquesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)

# correct for the negative sign of the torque
capillaryTorquesDistancesAsRows = - capillaryTorquesDistancesAsRows

# some extra treatment for the force matrix
# note the sharp transition at the peak-peak position (45 deg): only 1 deg difference,
# the force changes from attraction to repulsion. consider replacing values at eeDistance = 0, 1, 2,
# with values at eeDistance = 5um.
nearEdgeSmoothingThres = 1  # unit: micron; if 1, then it is equivalent to no smoothing.
for distanceToEdge in np.arange(nearEdgeSmoothingThres):
    capillaryForcesDistancesAsRows[distanceToEdge, :] = capillaryForcesDistancesAsRows[nearEdgeSmoothingThres, :]
    capillaryTorquesDistancesAsRows[distanceToEdge, :] = capillaryTorquesDistancesAsRows[nearEdgeSmoothingThres, :]

# select a cut-off distance below which all the attractive force (negative-valued) becomes zero,
# due to raft wall-wall repulsion
capAttractionZeroCutoff = 0
mask = np.concatenate((capillaryForcesDistancesAsRows[:capAttractionZeroCutoff, :] < 0,
                       np.zeros((capillaryForcesDistancesAsRows.shape[0] - capAttractionZeroCutoff,
                                 capillaryForcesDistancesAsRows.shape[1]), dtype=int)),
                      axis=0)
capillaryForcesDistancesAsRows[mask.nonzero()] = 0

# set capillary force = 0 at 0 distance
# capillaryForcesDistancesAsRows[0,:] = 0

# realign the first peak-peak direction with an angle = capillaryPeakOffset from the x-axis.
capillaryPeakOffset = 0
capillaryForcesDistancesAsRows = np.roll(capillaryForcesDistancesAsRows, capillaryPeakOffset,
                                         axis=1)  # 45 is due to original data
capillaryTorquesDistancesAsRows = np.roll(capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis=1)

capillaryForceAngleAveraged = capillaryForcesDistancesAsRows[1:, :-1].mean(axis=1)  # starting from 1 um to 1000 um
capillaryForceMaxRepulsion = capillaryForcesDistancesAsRows[1:, :-1].max(axis=1)
capillaryForceMaxRepulsionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmax(axis=1)
capillaryForceMaxAttraction = capillaryForcesDistancesAsRows[1:, :-1].min(axis=1)
capillaryForceMaxAttractionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmin(axis=1)

# %% magnetic force and torque calculation:
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m

# from the data 2018-09-28, 1st increase:
# (1.4e-8 A.m**2 for 14mT), (1.2e-8 A.m**2 for 10mT), (0.96e-8 A.m**2 for 5mT), (0.78e-8 A.m**2 for 1mT)
# from the data 2018-09-28, 2nd increase:
# (1.7e-8 A.m**2 for 14mT), (1.5e-8 A.m**2 for 10mT), (1.2e-8 A.m**2 for 5mT), (0.97e-8 A.m**2 for 1mT)
magneticMomentOfOneRaft = 1e-8  # unit: A.m**2

orientationAngles = np.arange(0, 361)  # unit: degree;
orientationAnglesInRad = np.radians(orientationAngles)

magneticDipoleEEDistances = np.arange(0, 10001) / 1e6  # unit: m

radiusOfRaft = 1.5e-4  # unit: m

magneticDipoleCCDistances = magneticDipoleEEDistances + radiusOfRaft * 2  # unit: m

# magDpEnergy = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: J
magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N.m

for index, d in enumerate(magneticDipoleCCDistances):
    # magDpEnergy[index, :] = \
    #     miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 3)
    magDpForceOnAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 4)
    magDpForceOffAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (2 * np.cos(orientationAnglesInRad) *
                                                   np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 4)
    magDpTorque[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (3 * np.cos(orientationAnglesInRad) *
                                               np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 3)

# magnetic force at 1um(attractionZeroCutoff) should have no attraction, due to wall-wall repulsion.
# Treat it similarly as capillary cutoff
# attractionZeroCutoff = 0  # unit: micron
# mask = np.concatenate((magDpForceOnAxis[:attractionZeroCutoff, :] < 0,
#                        np.zeros((magDpForceOnAxis.shape[0] - attractionZeroCutoff, magDpForceOnAxis.shape[1]),
#                                 dtype=int)), axis=0)
# magDpForceOnAxis[mask.nonzero()] = 0
#
# magDpMaxRepulsion = magDpForceOnAxis.max(axis=1)
# magDpForceAngleAverage = magDpForceOnAxis[:, :-1].mean(axis=1)

# set on-axis magnetic force = 0 at 0 distance
# magDpForceOnAxis[0,:] = 0

# %% lubrication equation coefficients:
RforCoeff = 150.0  # unit: micron
stepSizeForDist = 0.1
lubCoeffScaleFactor = 1 / stepSizeForDist
eeDistancesForCoeff = np.arange(0, 15 + stepSizeForDist, stepSizeForDist, dtype='double')  # unit: micron

eeDistancesForCoeff[0] = 1e-10  # unit: micron

x = eeDistancesForCoeff / RforCoeff  # unit: 1

lubA = x * (-0.285524 * x + 0.095493 * x * np.log(x) + 0.106103) / RforCoeff  # unit: 1/um

lubB = ((0.0212764 * (- np.log(x)) + 0.157378) * (- np.log(x)) + 0.269886) / (
        RforCoeff * (- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549)  # unit: 1/um

# lubC = ((-0.0212758 * (- np.log(x)) - 0.089656) * (- np.log(x)) + 0.0480911) / \
#        (RforCoeff ** 2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^2

# lubD = (0.0579125 * (- np.log(x)) + 0.0780201) / \
#        (RforCoeff ** 2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^2

lubG = ((0.0212758 * (- np.log(x)) + 0.181089) * (- np.log(x)) + 0.381213) / (
        RforCoeff ** 3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^3

lubC = - RforCoeff * lubG

# lubH = (0.265258 * (- np.log(x)) + 0.357355) / \
#        (RforCoeff ** 3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^3

# lubCoeffCombined = np.column_stack((lubA,lubB,lubC,lubD,lubG,lubH))

# %% check capillary and magnetic forces
# all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit

# check the dipole orientation and capillary orientation
# eeDistanceForPlotting = 70
# fig, ax = plt.subplots(ncols=2, nrows=1)
# ax[0].plot(capillaryForcesDistancesAsRows[eeDistanceForPlotting, :], 'o-',
#            label='capillary force')  # 0 deg is the peak-peak alignment - attraction.
# ax[0].plot(magDpForceOnAxis[eeDistanceForPlotting, :], 'o-',
#            label='magnetic force')  # 0 deg is the dipole-dipole attraction
# ax[0].set_xlabel('angle')
# ax[0].set_ylabel('force (N)')
# ax[0].legend()
# ax[0].set_title('force at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting,
#                                                                                          capillaryPeakOffset))
# ax[1].plot(capillaryTorquesDistancesAsRows[eeDistanceForPlotting, :], 'o-',
#            label='capillary torque')  # 0 deg is the peak-peak alignment - attraction.
# ax[1].plot(magDpTorque[eeDistanceForPlotting, :], 'o-',
#            label='magnetic torque')  # 0 deg is the dipole-dipole attraction
# ax[1].set_xlabel('angle')
# ax[1].set_ylabel('torque (N.m)')
# ax[1].legend()
# ax[1].set_title('torque at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting,
#                                                                                           capillaryPeakOffset))

# plot the various forces and look for the transition rps
# densityOfWater = 1e-15 # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
# raftRadius = 1.5e2 # unit: micron
# magneticFieldRotationRPS = 22
# omegaBField = magneticFieldRotationRPS * 2 * np.pi
# hydrodynamicRepulsion = densityOfWater * omegaBField ** 2 * raftRadius ** 7 * 1e-6 / \
#                         np.arange(raftRadius * 2 + 1, raftRadius * 2 + 1002) ** 3  # unit: N
# sumOfAllForces = capillaryForcesDistancesAsRows.mean(axis=1) + magDpForceOnAxis.mean(axis=1)[
#                                                                :1001] + hydrodynamicRepulsion
# fig, ax = plt.subplots(ncols = 1, nrows = 1)
# ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1), label = 'angle-averaged capillary force')
# ax.plot(magDpForceOnAxis.mean(axis = 1)[:1000], label = 'angle-averaged magnetic force')
# ax.plot(hydrodynamicRepulsion, label = 'hydrodynamic repulsion')
# ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1) + magDpForceOnAxis.mean(axis = 1)[:1001],
#         label = 'angle-avaraged sum of magnetic and capillary force')
# ax.set_xlabel('edge-edge distance (um)')
# ax.set_ylabel('Force (N)')
# ax.set_title('spin speed {} rps'.format(magneticFieldRotationRPS))
# ax.plot(sumOfAllForces, label = 'sum of angle-averaged magnetic and capillary forces and hydrodynamic force ')
# ax.legend()


# %% simulation of many rafts
if parallel_mode == 1:
#    numOfRafts = int(sys.argv[1])
#    spinSpeedStart = int(sys.argv[2])
#    spinSpeedStep = -1
#    spinSpeedEnd = spinSpeedStart + spinSpeedStep
#    f_y = -15
#    f_x = -30
    
    numOfRafts = 120
    spinSpeedStart = -15  # negative value is clockwise rotation
    spinSpeedEnd = -16
    spinSpeedStep = -1
    omega_stepout = 80
    
    m1 = 1.0
    m2 = 1.0 # unit: *1e-8 Am^2
    
    magmom = np.concatenate((m1*np.ones(int(numOfRafts/2)), m2*np.ones(numOfRafts - int(numOfRafts/2))))
    
    
    f_y = -1* int(sys.argv[1])
    f_x = -1* int(sys.argv[2])

#    Fy_Fx_list = ([-30, -0], [-0, -30]) # list of (fy, fx) to simulate
    
    B1_y_x_list = ([1,1], )
    B0_y_x_list = ([0,0], ) 
    
    Fy_Fx_list = ([f_y, f_x], ) # list of (fy, fx) to simulate
    
else:
    numOfRafts = 100
    spinSpeedStart = -15  # negative value is clockwise rotation
    spinSpeedEnd = -16
    spinSpeedStep = -1
    omega_stepout = 80
    
    m1 = 1 #1.0
    m2 = 1 #0.9 # unit: *1e-8 Am^2
    
    magmom = np.concatenate((m1*np.ones(int(numOfRafts/2)), m2*np.ones(numOfRafts - int(numOfRafts/2))))
    
#    B1_y_x_list = ([1,1], [1,1])
#    B0_y_x_list = ([0,0], [1,0]) 
#    Fy_Fx_list = ([-30, -0], [-0, -30]) # list of (fy, fx) to simulate
    
#    B1_y_x_list = ([1,1], [1,1])
#    B0_y_x_list = ([0,0], [0,0]) 
#    Fy_Fx_list = ([-30, -0], [-30, -30]) # list of (fy, fx) to simulate
##    
#    B1_y_x_list = ([1,1],)
#    B0_y_x_list = ([0,0],) 
#    Fy_Fx_list = ([-0, -60],)
    
#    B1_y_x_list = ([1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1])
#    B0_y_x_list = ([0,0], [0,0], [0,0], [1,0], [0,0], [0,0], [0,0], [0,0], [0,0]) 
#    Fy_Fx_list = ([-20, -20], [-30, 60], [-30, -31], [0, -30], [-30, 0], [-1, -0], [0, -70], [-70, -70], [-30, -60])
    
    B1_y_x_list = ([1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1])
    B0_y_x_list = ([1,0], [1,0], [0,0], [0,0], [1,0], [0,0], [0,0], [0,0]) 
    Fy_Fx_list = ([0, -30], [0, -40], [-30, -30], [-30, 60], [-30, -31], [0, -30], [-30, 0], [0, -30])
    
#    B1_y_x_list = ([1,1],)
#    B0_y_x_list = ([0,0],) 
#    Fy_Fx_list = ([-30, -60],)
    
#    B1_y_x_list = ([1,1], [1,1])
#    B0_y_x_list = ([0,0], [0,0]) 
#    Fy_Fx_list = ([0, -30], [0, -70])
    
#    B1_y_x_list = ([1,1], [1,1], [1,1], [1,1], [1,1])
#    B0_y_x_list = ([0,0], [0,0], [0,0], [0,0], [0,0]) 
#    Fy_Fx_list = ([-30, -30], [-30, -60], [-60, -61], [-30, 0], [-70, 0])
    
#    B1_y_x_list = ([1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1])
#    B0_y_x_list = ([0,0], [0,0], [0,0], [0,0], [1,0], [0,0], [0,0], [0,0]) 
#    Fy_Fx_list = ([-20, -20], [-30, 0], [-70, 0], [0, -30], [0, -30], [-30, -60], [-30, -31])
    
#    Fy_Fx_list = ([0, -30], [0, -60], [0,-90], [0,-20], [0, -10])
#   
#    Fy_Fx_list = ([0, -30], [-30, 0], [0, -10])
#    Fy_Fx_list = ([-30, -60], [-10, 0], [0, -70], [-30, -31], [-70, 0]) # list of (fy, fx) to simulate
    
#    Fy_Fx_list = ([-20, -10], [-30, -15], [-40, -20], [-100, -50], [-120, -20], [-0, -30], [-30, 0], [-15, -30], [-50, -100]) # list of (fy, fx) to simulate
    
#    Fy_Fx_list = ([-20, -70], [-40, -80], [-9.75, -60], [-160, -20]) # list of (fy, fx) to simulate
    
#    Fy_Fx_list = ([-15, -15], [-15, -16], [-15, -30], [-20, -21], [-20, -28], [-20,-50], 
#                  [-20, -55], [-20, -90], [-23, -85], [-23, -90]) # list of (fy, fx) to simulate
#    f_y = -20
#    f_x = -21

timeStepSize = 1e-3 # unit: s
numOfTimeSteps = 10000
timeTotal = timeStepSize * numOfTimeSteps

magneticFieldStrengthX_all = np.zeros(numOfTimeSteps)
magneticFieldStrengthY_all = np.zeros(numOfTimeSteps)
magneticFieldDirection1_all = np.zeros(numOfTimeSteps)
magneticFieldStrength1_all = np.zeros(numOfTimeSteps)
magneticFieldDirection1X_all = np.zeros(numOfTimeSteps)
magneticFieldDirection1Y_all = np.zeros(numOfTimeSteps)


os.chdir(dataDir)
now = datetime.datetime.now()

if parallel_mode == 1:
#    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
#                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's_' + str(m1) + 'm1_' + str(m2) + 'm2'
    
#    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
#                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's_' + str(m1) + 'm1_' + str(m2) + 'm2' + '_rotating' # rotating collective
#    
#    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
#                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's_' + str(m1) + 'm1_' + str(m2) + 'm2' + '_oscillating' # oscillating collective
#    
    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's_' + str(m1) + 'm1_' + str(m2) + 'm2' + '_static' # static collective
#    
#    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
#                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's_' + str(m1) + 'm1_' + str(m2) + 'm2' + '_Ychains' # Y chains
#    
#    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
#                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's_' + str(m1) + 'm1_' + str(m2) + 'm2' + '_GASPP' # GASPP
#    
    
else:
    outputFolderName = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(numOfRafts) + 'Rafts_' + \
                       'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's'

if not os.path.isdir(outputFolderName):
    os.mkdir(outputFolderName)
os.chdir(outputFolderName)

listOfVariablesToSave = ['arenaSize', 'numOfRafts', 'magneticFieldStrength', 'magneticFieldRotationRPS', 'omegaBField',
                         'timeStepSize', 'numOfTimeSteps',
                         'timeTotal', 'outputImageSeq', 'outputVideo', 'outputFrameRate', 'intervalBetweenFrames',
                         'raftLocations', 'raftOrientations', 'raftRadii', 'raftRotationSpeedsInRad',
                         # 'raftRelativeOrientationInDeg', # for debug use
                         'entropyByNeighborDistances', 'hexaticOrderParameterAvgs', 'hexaticOrderParameterAvgNorms',
                         'hexaticOrderParameterMeanSquaredDeviations', 'hexaticOrderParameterModuliiAvgs',
                         'hexaticOrderParameterModuliiStds',
                         'deltaR', 'radialRangeArray', 'binEdgesNeighborDistances',
                         'radialDistributionFunction', 'spatialCorrHexaOrderPara',
                         'spatialCorrHexaBondOrientationOrder',
                         # 'mag_dipole_force_on_axis_term', 'capillary_force_term', 'hydrodynamic_force_term',
                         # 'mag_dipole_force_off_axis_term', 'velocity_torque_coupling_term',
                         # 'velocity_mag_fd_torque_term', 'velocity_mag_fd_torque_coupling',
                         # 'wall_repulsion_term', 'stochastic_force_term', 'force_curvature_term',
                         # 'magnetic_field_torque_term', 'magnetic_dipole_torque_term',
                         # 'capillary_torque_term', 'stochastic_torque_term',
                         'currStepNum', 'currentFrameBGR', 'dfNeighbors', 'dfNeighborsAllFrames',
                         'cc', 'ch', 'cm', 'tb', 'tm', 'tc', 'omega_stepout', 'magneticFieldRotationRPSY', 'magneticFieldRotationRPSX', 'stdOfAngleNoise']

# constants of proportionality
cm = 1  # coefficient for the magnetic force term
cc = 1 #0.5*1  # coefficient for the capillary force term
ch = 1  # coefficient for the hydrodynamic force term
tb = 1  # coefficient for the magnetic field torque term
tm = 1  # coefficient for the magnetic dipole-dipole torque term
tc = 1  # coefficient for the capillary torque term
forceDueToCurvature = 0 # 5e-10  # 5e-9 #1e-10 # unit: N
wallRepulsionForce = 1e-7  # unit: N

unitVectorX = np.array([1, 0])
unitVectorY = np.array([0, 1])

if numOfRafts > 2:
    arenaSize = 1.2e4#1.2e4 #1.5e4
else:
    arenaSize = 5e3  # unit: micron
R = raftRadius = 1.5e2  # unit: micron (will come back and revise this duplication)
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])
# cutoffDistance = 100000  # unit: micron. Above which assume that the rafts do not interact.
# radiusOfCurvatureFreeCenter = 10 * raftRadius # unit: micron

# all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit
canvasSizeInPixel = int(1000)  # unit: pixel
scaleBar = arenaSize / canvasSizeInPixel  # unit: micron/pixel

densityOfWater = 1e-15  # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
miu = 1e-15  # dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2
piMiuR = np.pi * miu * raftRadius  # unit: N.s/um

magneticFieldStrength = 10e-3  # 10e-3 # unit: T
#magneticFieldStrengthX1 = 1  # aplitude of oscillating magnetic field along X-axis # time magneticFieldStrength # unit: T
#magneticFieldStrengthY1 = 1  # amplitude of oscillating magentic field along Y-axis # time magneticFieldStrength # unit: T
#magneticFieldStrengthX0 = 0  # constant magnetic field along X-axis # time magneticFieldStrength # unit: T
#magneticFieldStrengthY0 = 0  # constant magnetic field along Y-axis #time magneticFieldStrength # unit: T

magneticFieldDirection_array = np.zeros(numOfTimeSteps)
initialPositionMethod = 2  # 1 -random positions, 2 - fixed initial position,
# 3 - starting positions are the last positions of the previous spin speeds, 4 - starting from initial positions taken from a saved simulation file
ccSeparationStarting = 400  # unit: micron
initialOrientation = 0 #0  # unit: deg
lastPositionOfPreviousSpinSpeeds = np.zeros((numOfRafts, 2))  # come back and check if these three can be removed.
lastOrientationOfPreviousSpinSpeeds = np.zeros((numOfRafts, 2))
lastOmegaOfPreviousSpinSpeeds = np.zeros(numOfRafts)
firstSpinSpeedFlag = 1

lubEqThreshold = 15  # unit micron,
stdOfFluctuationTerm = 0.00
stdOfTorqueNoise = 0 #1e-15 #0  # 1e-12 # unit: N.m
stdOfAngleNoise = 0 #90 #1 # unit: degrees

deltaR = 1
radialRangeArray = np.arange(2, 100, deltaR)
binEdgesNeighborDistances = list(np.arange(2, 10, 0.5)) + [100]

outputImageSeq = 0
outputVideo = 1
outputFrameRate = 10.0
intervalBetweenFrames = int(10)  # unit: steps
blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype='int32') * 255

solverMethod = 'RK45'  # RK45,RK23, Radau, BDF, LSODA


def funct_drdt_dalphadt(t, raft_loc_orient):
    """
    Two sets of ordinary differential equations that define dr/dt and dalpha/dt above and below the threshold value
    for the application of lubrication equations
    """
    #    raft_loc_orient = raftLocationsOrientations
    raft_loc = raft_loc_orient[0: numOfRafts * 2].reshape(numOfRafts, 2)  # in um
    raft_orient = raft_loc_orient[numOfRafts * 2: numOfRafts * 3]  # in deg

    drdt = np.zeros((numOfRafts, 2))  # unit: um
    raft_spin_speeds_in_rads = np.zeros(numOfRafts)  # in rad
    dalphadt = np.zeros(numOfRafts)  # unit: deg

    mag_dipole_force_on_axis_term = np.zeros((numOfRafts, 2))
    capillary_force_term = np.zeros((numOfRafts, 2))
    hydrodynamic_force_term = np.zeros((numOfRafts, 2))
    mag_dipole_force_off_axis_term = np.zeros((numOfRafts, 2))
    velocity_torque_coupling_term = np.zeros((numOfRafts, 2))
    velocity_mag_fd_torque_coupling = np.zeros((numOfRafts, 2))
    wall_repulsion_term = np.zeros((numOfRafts, 2))
    stochastic_force = np.zeros((numOfRafts, 2))
    curvature_force_term = np.zeros((numOfRafts, 2))
    boundary_force_term = np.zeros((numOfRafts, 2))

    magnetic_field_torque_term = np.zeros(numOfRafts)
    magnetic_dipole_torque_term = np.zeros(numOfRafts)
    capillary_torque_term = np.zeros(numOfRafts)
    stochastic_torque_term = np.zeros(numOfRafts)
    
#    ch /= max(1,abs(magneticFieldRotationRPSY))
    
#    print(t)
    magneticFieldDirection1X = magneticFieldDirectionX + (magneticFieldRotationRPSX * 360 * t) % 360
    magneticFieldDirection1Y = magneticFieldDirectionY + (magneticFieldRotationRPSY * 360 * t) % 360
    
#    magneticFieldDirection1 = magneticFieldDirection + (magneticFieldRotationRPS * 360 * t) % 360
#    magneticFieldStrengthX = np.cos(np.deg2rad(magneticFieldDirection1X))
#    magneticFieldStrengthY = np.sin(np.deg2rad(magneticFieldDirection1Y))
#    magneticFieldDirection1 = np.arctan2(magneticFieldStrengthY, magneticFieldStrengthX)*(180/np.pi) % 360
#    magneticFieldStrength1 = magneticFieldStrength*np.sqrt(magneticFieldStrengthX**2 + magneticFieldStrengthY**2)
    
    magneticFieldStrengthX = magneticFieldStrength*(magneticFieldStrengthX1*np.cos(np.deg2rad(magneticFieldDirection1X)) + magneticFieldStrengthX0)
    magneticFieldStrengthY = magneticFieldStrength*(magneticFieldStrengthY1*np.sin(np.deg2rad(magneticFieldDirection1Y)) + magneticFieldStrengthY0)
    magneticFieldDirection1 = np.arctan2(magneticFieldStrengthY, magneticFieldStrengthX)*(180/np.pi) % 360
    magneticFieldStrength1 = np.sqrt(magneticFieldStrengthX**2 + magneticFieldStrengthY**2)
#    magneticFieldStrength1 = magneticFieldStrength*np.sqrt(magneticFieldStrengthX**2 + magneticFieldStrengthY**2)
    
#    print("sto-Start")
    # stochastic torque term
#    stochastic_torque = omegaBField * np.random.normal(0, stdOfTorqueNoise, 1)
    stochastic_torque = omegaBField * np.random.normal(0, stdOfTorqueNoise, numOfRafts)
#     unit: N.m, assuming omegaBField is unitless
#    stochastic_torque_term = np.ones(numOfRafts) * stochastic_torque * 1e6 / (8 * piMiuR * R ** 2)
    stochastic_torque_term = stochastic_torque * 1e6 / (8 * piMiuR * R ** 2)
#     unit: 1/s assuming omegaBField is unitless.

#    print("sto-end")

    # loop for torques and calculate raft_spin_speeds_in_rads
    for raft_id in np.arange(numOfRafts):
        # raft_id = 0
        ri = raft_loc[raft_id, :]  # unit: micron

        # magnetic field torque:
#        magnetic_field_torque = \
#            magneticFieldStrength1 * magneticMomentOfOneRaft \
#            * np.sin(np.deg2rad(magneticFieldDirection1 - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque = \
            magneticFieldStrength1 * magneticMomentOfOneRaft * magmom[raft_id] \
            * np.sin(np.deg2rad(magneticFieldDirection1 - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque_term[raft_id] = tb * magnetic_field_torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        rji_ee_dist_smallest = R  # initialize

        for neighbor_id in np.arange(numOfRafts):
            if neighbor_id == raft_id:
                continue
            rj = raft_loc[neighbor_id, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_ee_dist = rji_norm - 2 * R  # unit: micron
            rji_unitized = rji / rji_norm  # unit: micron
            rji_unitized_cross_z = np.asarray((rji_unitized[1], -rji_unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi
                      - raft_orient[raft_id]) % 360  # unit: deg; assuming both rafts's orientations are the same

            # torque terms:
            # magnetic dipole-dipole torque
            if 10000 > rji_ee_dist >= lubEqThreshold:
                magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] \
                                                       + tm * magDpTorque[int(rji_ee_dist + 0.5), int(phi_ji + 0.5)] \
                                                       * 1e6 / (8 * piMiuR * R ** 2)
            elif lubEqThreshold > rji_ee_dist >= 0:
                magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] + \
                                                       tm * lubG[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                       * magDpTorque[int(rji_ee_dist + 0.5), int(phi_ji + 0.5)] \
                                                       * 1e6 / miu  # unit: 1/s
            elif rji_ee_dist < 0:
                magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] + tm * lubG[0] \
                                                       * magDpTorque[0, int(phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s
            # capillary torque
            if 1000 > rji_ee_dist >= lubEqThreshold:
                capillary_torque_term[raft_id] = capillary_torque_term[raft_id] + tc * \
                                                 capillaryTorquesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                 * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                capillary_torque_term[raft_id] = capillary_torque_term[raft_id] + tc * \
                                                 lubG[int(rji_ee_dist * lubCoeffScaleFactor)] * \
                                                 capillaryTorquesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                 * 1e6 / miu  # unit: 1/s
            elif rji_ee_dist < 0:
                capillary_torque_term[raft_id] = capillary_torque_term[raft_id] + tc * lubG[0] * \
                                                 capillaryTorquesDistancesAsRows[0, int(phi_ji + 0.5)] \
                                                 * 1e6 / miu  # unit: 1/s
            # change magnetic field torque only if the smallest edge-edge distance is below lubrication threshold
            if rji_ee_dist < lubEqThreshold and rji_ee_dist < rji_ee_dist_smallest:
                rji_ee_dist_smallest = rji_ee_dist
                if rji_ee_dist_smallest >= 0:
                    magnetic_field_torque_term[raft_id] = lubG[int(rji_ee_dist_smallest * lubCoeffScaleFactor)] \
                                                          * magnetic_field_torque * 1e6 / miu \
                                                          + lubC[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                          * magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                         int(phi_ji + 0.5)] / miu  # unit: 1/s
                elif rji_ee_dist_smallest < 0:
                    magnetic_field_torque_term[raft_id] = lubG[0] * magnetic_field_torque * 1e6 / miu \
                                                        + lubC[0] * magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                        int(phi_ji + 0.5)] / miu  # unit: 1/s

            # debug use:
        #            raftRelativeOrientationInDeg[neighbor_id, raftID, currentStepNum] = phi_ji

        # debug use
        #        capillaryTorqueTerm[raftID, currentStepNum] = capillary_torque_term[raftID]

        raft_spin_speeds_in_rads[raft_id] = \
            magnetic_field_torque_term[raft_id] + magnetic_dipole_torque_term[raft_id] \
            + capillary_torque_term[raft_id] + stochastic_torque_term[raft_id]
            
        # add stepout behaviour # w = wh - wh*(sqrt(1 - (wc/wh)^2))
        if abs(raft_spin_speeds_in_rads[raft_id]) > omega_stepout*2*np.pi:
#            print("stepped out_" + str(raft_spin_speeds_in_rads[raft_id]) + "_t_" + str(t))
#            raft_spin_speeds_in_rads[raft_id] = omega_stepout*2*np.pi*np.sign(raft_spin_speeds_in_rads[raft_id])
            raft_spin_speeds_in_rads[raft_id] = raft_spin_speeds_in_rads[raft_id] - raft_spin_speeds_in_rads[raft_id]*np.sqrt(1 - \
                                                    (abs(omega_stepout*2*np.pi)/abs(raft_spin_speeds_in_rads[raft_id]))**2)
                
#    print("Torque alculated")
                
    # loop for forces
    for raft_id in np.arange(numOfRafts):
        # raftID = 0
        ri = raft_loc[raft_id, :]  # unit: micron
        omegai = raft_spin_speeds_in_rads[raft_id]  # only reason to have separate loops for force and torque?
        
        # meniscus curvature force term
        if forceDueToCurvature != 0:
            ri_center = centerOfArena - ri
            #            ri_center_Norm = np.sqrt(ri_center[0]**2 + ri_center[1]**2)
            #            ri_center_Unitized = ri_center / ri_center_Norm
            curvature_force_term[raft_id, :] = forceDueToCurvature / (6 * piMiuR) * ri_center / (arenaSize / 2)

        # boundary lift force term
        if numOfRafts > 2:
            d_to_left = ri[0]
            d_to_right = arenaSize - ri[0]
            d_to_bottom = ri[1]
            d_to_top = arenaSize - ri[1]
            boundary_force_term[raft_id, :] = \
                1e-6 * densityOfWater * omegai ** 2 * R ** 7 / (6 * piMiuR) \
                * ((1 / d_to_left ** 3 - 1 / d_to_right ** 3) * unitVectorX
                   + (1 / d_to_bottom ** 3 - 1 / d_to_top ** 3) * unitVectorY)

        # magnetic field torque:
#        magnetic_field_torque = \
#            magneticFieldStrength1 * magneticMomentOfOneRaft \
#            * np.sin(np.deg2rad(magneticFieldDirection1 - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque = \
            magneticFieldStrength1 * magneticMomentOfOneRaft * magmom[raft_id] \
            * np.sin(np.deg2rad(magneticFieldDirection1 - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque_term[raft_id] = tb * magnetic_field_torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        for neighbor_id in np.arange(numOfRafts):
            if neighbor_id == raft_id:
                continue
            rj = raft_loc[neighbor_id, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_ee_dist = rji_norm - 2 * R  # unit: micron
            rji_unitized = rji / rji_norm  # unit: micron
            rji_unitized_cross_z = np.asarray((rji_unitized[1], -rji_unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raft_id]) % 360  # unit: deg;

            # force terms:
            omegaj = raft_spin_speeds_in_rads[neighbor_id]
            # need to come back and see how to deal with this. maybe you need to define it as a global variable.

            # magnetic dipole force on axis
            if 10000 > rji_ee_dist >= lubEqThreshold:
                mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] \
                                                            + cm * magDpForceOnAxis[int(rji_ee_dist + 0.5),
                                                                                    int(phi_ji + 0.5)] \
                                                            * rji_unitized / (6 * piMiuR)  # unit: um/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] \
                                                            + cm * lubA[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                            * magDpForceOnAxis[int(rji_ee_dist + 0.5),
                                                                               int(phi_ji + 0.5)] \
                                                            * rji_unitized / miu  # unit: um/s
            elif rji_ee_dist < 0:
                mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] \
                                                            + cm * lubA[0] * magDpForceOnAxis[0, int(phi_ji + 0.5)] \
                                                            * rji_unitized / miu  # unit: um/s
            # capillary force
            if 1000 > rji_ee_dist >= lubEqThreshold:
                capillary_force_term[raft_id, :] = capillary_force_term[raft_id, :] \
                                                   + cc * capillaryForcesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                         int(phi_ji + 0.5)] \
                                                   * rji_unitized / (6 * piMiuR)  # unit: um/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                capillary_force_term[raft_id, :] = capillary_force_term[raft_id, :] \
                                                   + cc * lubA[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                   * capillaryForcesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                    int(phi_ji + 0.5)] \
                                                   * rji_unitized / miu  # unit: um/s
            elif rji_ee_dist < 0:
                capillary_force_term[raft_id, :] = capillary_force_term[raft_id, :] \
                                                   + cc * lubA[0] * capillaryForcesDistancesAsRows[0,
                                                                                                   int(phi_ji + 0.5)] \
                                                   * rji_unitized / miu  # unit: um/s
            
            # hydrodynamic force (hydro force reduces for higher omega)
            if rji_ee_dist >= lubEqThreshold:
                hydrodynamic_force_term[raft_id, :] = hydrodynamic_force_term[raft_id, :] \
                                                      + ch * 1e-6 * densityOfWater * omegaj ** 2 * R ** 7 * rji \
                                                      / rji_norm ** 4 / (6 * piMiuR)  # unit: um/s;
            elif lubEqThreshold > rji_ee_dist >= 0:
                hydrodynamic_force_term[raft_id, :] = hydrodynamic_force_term[raft_id, :] \
                                                      + ch * lubA[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                      * (1e-6 * densityOfWater * omegaj ** 2 * R ** 7 / rji_norm ** 3) \
                                                      * rji_unitized / miu  # unit: um/s
            # wall repulsion term when two rafts overlap
            if rji_ee_dist < 0:
                wall_repulsion_term[raft_id, :] = wall_repulsion_term[raft_id, :] \
                                                  + wallRepulsionForce * (-rji_ee_dist / R) * rji_unitized \
                                                  / (6 * piMiuR)
            # magnetic dipole-dipole force, off-axis
            if 10000 > rji_ee_dist >= lubEqThreshold:
                mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / (6 * piMiuR)
            elif lubEqThreshold > rji_ee_dist >= 0:
                mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + lubB[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                             * magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / miu  # unit: um/s
            elif rji_ee_dist < 0:
                mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + lubB[0] * magDpForceOffAxis[0, int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / miu  # unit: um/s
            # velocity-torque coupling above the lub-threshold, velocity-mag-field coupling below lub-threshold
            if rji_ee_dist >= lubEqThreshold:
                velocity_torque_coupling_term[raft_id, :] = velocity_torque_coupling_term[raft_id, :] \
                                                            - R ** 3 * omegaj * rji_unitized_cross_z \
                                                            / (rji_norm ** 2)  # unit: um/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                velocity_mag_fd_torque_coupling[raft_id, :] = velocity_mag_fd_torque_coupling[raft_id, :] \
                                                              + lubC[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                              * (magnetic_field_torque \
                                                              + magDpTorque[0, int(phi_ji + 0.5)] \
                                                              + capillaryTorquesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)]) * 1e6 \
                                                                 * rji_unitized_cross_z / miu  # unit: um/s
            elif rji_ee_dist < 0:
                velocity_mag_fd_torque_coupling[raft_id, :] = velocity_mag_fd_torque_coupling[raft_id, :] \
                                                              + lubC[0] * (magnetic_field_torque \
                                                              + magDpTorque[0, int(phi_ji + 0.5)] \
                                                              + capillaryTorquesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)]) * 1e6 \
                                                              * rji_unitized_cross_z / miu  # unit: um/s
            # stochastic force term
            # if rji_ee_dist >= lubEqThreshold and currentStepNum > 1:
            #     prev_drdt = (raftLocations[raftID, currentStepNum, :]
            #                  - raftLocations[raftID, currentStepNum-1, :]) / timeStepSize
            #     stochastic_force[raftID, currentStepNum, :] = stochastic_force[raftID, currentStepNum, :] \
            #                                                  + np.sqrt(prev_drdt[0]**2 + prev_drdt[1]**2) \
            #                                                  * np.random.normal(0, stdOfFluctuationTerm, 1) \
            #                                                  * rji_unitized

        # update drdr and dalphadt
        drdt[raft_id, :] = \
            mag_dipole_force_on_axis_term[raft_id, :] + capillary_force_term[raft_id, :] \
            + hydrodynamic_force_term[raft_id, :] + wall_repulsion_term[raft_id, :] \
            + mag_dipole_force_off_axis_term[raft_id, :] + velocity_torque_coupling_term[raft_id, :] \
            + velocity_mag_fd_torque_coupling[raft_id, :] + stochastic_force[raft_id, :] \
            + curvature_force_term[raft_id, :] + boundary_force_term[raft_id, :]
            
    dalphadt = raft_spin_speeds_in_rads / np.pi * 180  # in deg

    drdt_dalphadt = np.concatenate((drdt.flatten(), dalphadt))
    
#    print("Force alculated")

    return drdt_dalphadt


# for forceDueToCurvature in np.array([0]):
#for magneticFieldRotationRPS in np.arange(spinSpeedStart, spinSpeedEnd, spinSpeedStep):
for index, f in enumerate(Fy_Fx_list):
    # negative magneticFieldRotationRPS means clockwise in rh coordinate,
    # positive magneticFieldRotationRPS means counter-clockwise
    # magneticFieldRotationRPS = -10 # unit: rps (rounds per seconds)
    magneticFieldRotationRPS = -15
    omegaBField = magneticFieldRotationRPS * 2 * np.pi  # unit: rad/s
#    magneticFieldRotationRPSY = magneticFieldRotationRPS*1
#    magneticFieldRotationRPSX = magneticFieldRotationRPS*3
    
    magneticFieldRotationRPSY = f[0]
    magneticFieldRotationRPSX = f[1]
    
    magneticFieldStrengthY1 = B1_y_x_list[index][0]
    magneticFieldStrengthX1 = B1_y_x_list[index][1]
    magneticFieldStrengthY0 = B0_y_x_list[index][0]
    magneticFieldStrengthX0 = B0_y_x_list[index][1]
    
    if magneticFieldRotationRPSY == 0 and magneticFieldStrengthY0 == 0:
        stdOfAngleNoise = 10 #10 # using stepout behaviour noise is not neccessary for GASPP
        ch = 0
        cc = 0
        tc = 0
#        numOfTimeSteps = 10000
       
    elif magneticFieldRotationRPSY == magneticFieldRotationRPSX:
        stdOfAngleNoise = 0
        ch = 1
        cc = 1
        tc = 1
#        numOfTimeSteps = 5000
    
    elif magneticFieldRotationRPSY == 0 or magneticFieldRotationRPSX == 0 :
        stdOfAngleNoise = 0
        ch = 1
        cc = 1
        tc = 1
#        numOfTimeSteps = 10000
       
    else:
        stdOfAngleNoise = 0
        ch = 1 #0.25
        cc = 1
        tc = 1
#        numOfTimeSteps = 10000
        
#    print("fy=" + str(magneticFieldRotationRPSY) + '_fx=' + str(magneticFieldRotationRPSX) + '_cc=' + str(cc) + '_ch=' + str(ch))
    
    # initialize key dataset
    raftLocations = np.zeros((numOfRafts, numOfTimeSteps, 2))  # in microns
    raftOrientations = np.zeros((numOfRafts, numOfTimeSteps))  # in deg
    raftRadii = np.ones(numOfRafts) * raftRadius  # in micron
    raftRotationSpeedsInRad = np.zeros((numOfRafts, numOfTimeSteps))  # in rad
    raftRelativeOrientationInDeg = np.zeros((numOfRafts, numOfRafts, numOfTimeSteps))
    #  in deg, (neighborID, raftID, frame#)

    # For debug use:
    # magDipoleForceOnAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # capillaryForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # hydrodynamicForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # magDipoleForceOffAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # velocityTorqueCouplingTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # velocityMagDpTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # wallRepulsionTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # stochasticTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # curvatureForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    # boundaryForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #
    # magneticFieldTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    # magneticDipoleTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    # capillaryTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    # stochasticTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))

    # initialize variables for order parameters:
    entropyByNeighborDistances = np.zeros(numOfTimeSteps)

    hexaticOrderParameterAvgs = np.zeros(numOfTimeSteps, dtype=np.csingle)
    hexaticOrderParameterAvgNorms = np.zeros(numOfTimeSteps)
    hexaticOrderParameterMeanSquaredDeviations = np.zeros(numOfTimeSteps, dtype=np.csingle)
    hexaticOrderParameterModuliiAvgs = np.zeros(numOfTimeSteps)
    hexaticOrderParameterModuliiStds = np.zeros(numOfTimeSteps)

    radialDistributionFunction = np.zeros((numOfTimeSteps, len(radialRangeArray)))  # pair correlation function: g(r)
    spatialCorrHexaOrderPara = np.zeros((numOfTimeSteps, len(radialRangeArray)))
    # spatial correlation of hexatic order paramter: g6(r)
    spatialCorrHexaBondOrientationOrder = np.zeros((numOfTimeSteps, len(radialRangeArray)))
    # spatial correlation of bond orientation parameter: g6(r)/g(r)

    dfNeighbors = pd.DataFrame(columns=['frameNum', 'raftID', 'hexaticOrderParameter',
                                        'neighborDistances', 'neighborDistancesAvg'])

    dfNeighborsAllFrames = pd.DataFrame(columns=['frameNum', 'raftID', 'hexaticOrderParameter',
                                                 'neighborDistances', 'neighborDistancesAvg'])

    currStepNum = 0
    # initialize rafts positions: 1 - random positions, 2 - fixed initial position,
    # 3 - starting positions are the last positions of the previous spin speeds
    if initialPositionMethod == 1:
        # initialize the raft positions in the first frame, check pairwise ccdistance all above 2R
        paddingAroundArena = 5  # unit: radius
        ccDistanceMin = 2.5  # unit: radius
        raftLocations[:, currStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                             arenaSize - raftRadius * paddingAroundArena,
                                                             (numOfRafts, 2))
        raftsToRelocate = np.arange(numOfRafts)
        while len(raftsToRelocate) > 0:
            raftLocations[raftsToRelocate, currStepNum, :] = np.random.uniform(
                0 + raftRadius * paddingAroundArena,
                arenaSize - raftRadius * paddingAroundArena, (len(raftsToRelocate), 2))
            pairwiseDistances = scipy_distance.cdist(raftLocations[:, currStepNum, :],
                                                     raftLocations[:, currStepNum, :], 'euclidean')
            np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
            raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
            raftsToRelocate = np.unique(raftsToRelocate)
    elif initialPositionMethod == 2 or (initialPositionMethod == 3 and firstSpinSpeedFlag == 1):
        if numOfRafts == 2:
            raftLocations[0, currStepNum, :] = np.array([arenaSize / 2 + ccSeparationStarting / 2, arenaSize / 2])
            raftLocations[1, currStepNum, :] = np.array([arenaSize / 2 - ccSeparationStarting / 2, arenaSize / 2])
##            
#            raftLocations[0, currStepNum, :] = np.array([arenaSize / 2, arenaSize / 2 + ccSeparationStarting / 2])
#            raftLocations[1, currStepNum, :] = np.array([arenaSize / 2, arenaSize / 2 - ccSeparationStarting / 2])
##            
            raftOrientations[:, currStepNum] = initialOrientation
            raftRotationSpeedsInRad[:, currStepNum] = omegaBField
        else:
#            raftLocations[:, currStepNum, :] = fsr.square_spiral(numOfRafts, raftRadius * 2 + 100, centerOfArena)
            raftLocations[:, currStepNum, :] = fsr.hexagonal_spiral(numOfRafts, raftRadius * 2 + 100, centerOfArena)
        firstSpinSpeedFlag = 0
    elif initialPositionMethod == 3 and firstSpinSpeedFlag == 0:
#        raftLocations[0, currStepNum, :] = lastPositionOfPreviousSpinSpeeds[0, :]
#        raftLocations[1, currStepNum, :] = lastPositionOfPreviousSpinSpeeds[1, :]
        raftLocations[:, currStepNum, :] = lastPositionOfPreviousSpinSpeeds
        raftOrientations[:, currStepNum] = lastOrientationOfPreviousSpinSpeeds
        raftRotationSpeedsInRad[:, currStepNum] = lastOmegaOfPreviousSpinSpeeds
        
    elif initialPositionMethod == 4: # get initial positions from last frame of previously saved simulations.
        # the file to read the initial positions from 
        initilal_file = '/home/gardi/Research_IS/spinning_rafts_sim2/data/2021-09-13_00-08-42_100Rafts_timeStep0.001_total5.0s_transition successful/Simulation_RK45_100Rafts_Yx-70_Xx-70Hz_B0.01T_m1e-08Am2_capPeak0_curvF0_startPosMeth3_lubEqThres15_timeStep0.001_5.0s_7' 
        InitDataList = []
        variableListsForAllInitData = []
        tShel = shelve.open(initilal_file)
        variableListOfOneInitDataFile = list(tShel.keys())
        expDict = {}
        for key in tShel:
            try:
                expDict[key] = tShel[key]
            except TypeError:
                pass
        tShel.close()
        InitDataList.append(expDict)
        variableListsForAllInitData.append(variableListOfOneInitDataFile)
    
        raftLocations[:, currStepNum, :] = InitDataList[0]['raftLocations'][:,-1,:]
        raftOrientations[:, currStepNum] = InitDataList[0]['raftOrientations'][:,-1]
        raftRotationSpeedsInRad[:, currStepNum] = InitDataList[0]['raftRotationSpeedsInRad'][:,-1]
    
        
#        # load existing simulation data shelve file
#        os.chdir(dataDir)
#        resultFolders = next(os.walk(dataDir))[1]
#        resultFolders.sort()
#        
#        resultFolderID = -1 #-1  # last folder
#        os.chdir(resultFolders[resultFolderID])
#        
#        parts = resultFolders[resultFolderID].split('_')
#        
#        magneticFieldRotationRPSY_prev = 0 #Fy_Fx_list[index-1][0]
#        magneticFieldRotationRPSX_prev = -30 #Fy_Fx_list[index-1][1]
##       fileToRead = 'Simulation_RK45_{}Rafts_Yx{}_Xx{}Hz_B0.01T_m1e-08Am2_capPeak0_curvF0_startPosMeth2_lubEqThres15_timeStep0.001_10.0s'.format(numOfRafts, magneticFieldRotationRPSY, magneticFieldRotationRPSX)
#        fileToRead = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_Yx' \
#                     + str(magneticFieldRotationRPSY_prev).zfill(3) + '_Xx' + str(magneticFieldRotationRPSX_prev) + 'Hz_B' + str(magneticFieldStrength) \
#                     + 'T_m' + str(magneticMomentOfOneRaft) + 'Am2_capPeak' + str(capillaryPeakOffset) + '_curvF' \
#                     + str(forceDueToCurvature) + '_startPosMeth' + str(initialPositionMethod) + '_lubEqThres' \
#                     + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's'
#        shelfToRead = shelve.open(fileToRead, flag='r')
#        listOfVariablesInShelfToRead = list(shelfToRead.keys())
#        for key in shelfToRead:
#            globals()[key] = shelfToRead[key]
#        shelfToRead.close()

    # check and draw the initial position of rafts
    # currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
    #                                           np.int32(raftLocations[:, currentStepNum, :] / scaleBar),
    #                                           np.int64(raftRadii / scaleBar), numOfRafts)
    # currentFrameBGR = fsr.draw_cap_peaks_rh_coord(currentFrameBGR,
    #                                               np.int64(raftLocations[:, currentStepNum, :]/scaleBar),
    #                                               raftOrientations[:, currentStepNum], 6, capillaryPeakOffset,
    #                                               np.int64(raftRadii / scaleBar), numOfRafts)
    # currentFrameBGR = fsr.draw_raft_orientations_rh_coord(currentFrameBGR,
    #                                                       np.int64(raftLocations[:, currentStepNum, :]/scaleBar),
    #                                                       raftOrientations[:, currentStepNum],
    #                                                       np.int64(raftRadii/scaleBar), numOfRafts)
    # currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
    #                                              np.int64(raftLocations[:, currentStepNum, :]/scaleBar),
    #                                              numOfRafts)
    # plt.imshow(currentFrameBGR)

#    outputFileName = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_' \
#                     + str(magneticFieldRotationRPS).zfill(3) + 'rps_B' + str(magneticFieldStrength) \
#                     + 'T_m' + str(magneticMomentOfOneRaft) + 'Am2_capPeak' + str(capillaryPeakOffset) + '_curvF' \
#                     + str(forceDueToCurvature) + '_startPosMeth' + str(initialPositionMethod) + '_lubEqThres' \
#                     + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's'
    outputFileName = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_Yx' \
                     + str(magneticFieldRotationRPSY).zfill(3) + '_Xx' + str(magneticFieldRotationRPSX) + 'Hz_B' + str(magneticFieldStrength) \
                     + 'T_m' + str(magneticMomentOfOneRaft) + 'Am2_capPeak' + str(capillaryPeakOffset) + '_curvF' \
                     + str(forceDueToCurvature) + '_startPosMeth' + str(initialPositionMethod) + '_lubEqThres' \
                     + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's_' + str(index) 

    if outputVideo == 1:
        outputVideoName = outputFileName + '.mp4'
        fourcc = cv.VideoWriter_fourcc(*'DIVX')  # *'mp4v' worked for linux, *'DIVX', MJPG
        frameW, frameH, _ = blankFrameBGR.shape
        videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

    for currStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
        # currentStepNum = 0
#        magneticFieldRotationRPS
        magneticFieldDirectionY = (magneticFieldRotationRPSY * 360 * currStepNum * timeStepSize) % 360
        magneticFieldDirectionX = (magneticFieldRotationRPSX * 360 * currStepNum * timeStepSize) % 360
        magneticFieldDirection1X_all[currStepNum] = magneticFieldDirectionX
        magneticFieldDirection1Y_all[currStepNum] = magneticFieldDirectionY
        
        # for debugging the magnetic field profile
        
         #magneticFieldStrengthX = np.cos(np.deg2rad(magneticFieldDirectionX))
         #magneticFieldStrengthY = np.sin(np.deg2rad(magneticFieldDirectionY))
        magneticFieldStrengthX = magneticFieldStrength*(magneticFieldStrengthX1*np.cos(np.deg2rad(magneticFieldDirectionX)) + magneticFieldStrengthX0)
        magneticFieldStrengthY = magneticFieldStrength*(magneticFieldStrengthY1*np.sin(np.deg2rad(magneticFieldDirectionY)) + magneticFieldStrengthY0)
        magneticFieldDirection1 = np.arctan2(magneticFieldStrengthY, magneticFieldStrengthX)*(180/np.pi) % 360
        magneticFieldStrength1 = np.sqrt(magneticFieldStrengthX**2 + magneticFieldStrengthY**2)
#        
#        
        magneticFieldStrengthX_all[currStepNum] = magneticFieldStrengthX
        magneticFieldStrengthY_all[currStepNum] = magneticFieldStrengthY
#        
##        plt.plot(magneticFieldStrengthX_all[:currStepNum],magneticFieldStrengthY_all[:currStepNum], c='b',
##                 label="stepnum=" + str(currStepNum*timeStepSize))
##        plt.legend()
##        plt.savefig("plot")
##        plt.close("all")
#        
#        
#        magneticFieldDirection1 = np.arctan2(magneticFieldStrengthY, magneticFieldStrengthX)*(180/np.pi) % 360
#        magneticFieldStrength1 = magneticFieldStrength*np.sqrt(magneticFieldStrengthX**2 + magneticFieldStrengthY**2)
#        
#        magneticFieldDirection1_all[currStepNum] = magneticFieldDirection1
#        magneticFieldStrength1_all[currStepNum] = magneticFieldStrength1
        
#        magneticFieldDirection = np.arctan2(np.sin(magneticFieldDirectionY*np.pi/180), np.cos(magneticFieldDirectionX*np.pi/180))*180/np.pi % 360
        magneticFieldDirection = np.arctan2(magneticFieldStrengthY, magneticFieldStrengthX)*(180/np.pi) % 360
        
        magneticFieldDirection_array[currStepNum] = magneticFieldDirection
        
        raftLocationsOrientations = np.concatenate((raftLocations[:, currStepNum, :].flatten(),
                                                    raftOrientations[:, currStepNum]))

        sol = solve_ivp(funct_drdt_dalphadt, (0, timeStepSize), raftLocationsOrientations, method=solverMethod)
        
#        print("solver return")

        raftLocations[:, currStepNum + 1, :] = sol.y[0:numOfRafts * 2, -1].reshape(numOfRafts, 2)
        
        angleNoise = np.random.normal(0, stdOfAngleNoise, numOfRafts)
        raftOrientations[:, currStepNum + 1] = sol.y[numOfRafts * 2: numOfRafts * 3, -1] + angleNoise
        
#        # add noise in angle whenever the magnetic field direction changes (for GASPP, it changes by 180 degree)
#        if magneticFieldDirection_array[currStepNum] - magneticFieldDirection_array[currStepNum - 1] == 0:
#            raftOrientations[:, currStepNum + 1] = sol.y[numOfRafts * 2: numOfRafts * 3, -1]
#        else:
#            raftOrientations[:, currStepNum + 1] = sol.y[numOfRafts * 2: numOfRafts * 3, -1] + np.random.normal(0, stdOfAngleNoise, numOfRafts)
            


        if numOfRafts > 2:
            # Voronoi calculation:
            vor = scipyVoronoi(raftLocations[:, currStepNum, :])
            allVertices = vor.vertices
            neighborPairs = vor.ridge_points  # row# is the index of a ridge,
            # columns are the two point# that correspond to the ridge

            ridgeVertexPairs = np.asarray(vor.ridge_vertices)  # row# is the index of a ridge,
            # columns are two vertex# of the ridge

            pairwiseDistances = scipy_distance.cdist(raftLocations[:, currStepNum, :],
                                                     raftLocations[:, currStepNum, :], 'euclidean')

            # calculate hexatic order parameter and entropy by neighbor distances
            for raftID in np.arange(numOfRafts):
                # raftID = 0
                r_i = raftLocations[raftID, currStepNum, :]  # unit: micron

                # neighbors of this particular raft:
                ridgeIndices0 = np.nonzero(neighborPairs[:, 0] == raftID)
                ridgeIndices1 = np.nonzero(neighborPairs[:, 1] == raftID)
                ridgeIndices = np.concatenate((ridgeIndices0, ridgeIndices1), axis=None)
                neighborPairsOfOneRaft = neighborPairs[ridgeIndices, :]
                NNsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[neighborPairsOfOneRaft[:, 0] == raftID, 1],
                                               neighborPairsOfOneRaft[neighborPairsOfOneRaft[:, 1] == raftID, 0]))
                neighborDistances = pairwiseDistances[raftID, NNsOfOneRaft]

                # calculate hexatic order parameter of this one raft
                neighborLocations = raftLocations[NNsOfOneRaft, currStepNum, :]
                neighborAnglesInRad = np.arctan2(-(neighborLocations[:, 1] - r_i[1]),
                                                 (neighborLocations[:, 0] - r_i[0]))
                # negative sign to make angle in the right-handed coordinate

                raftHexaticOrderParameter = \
                    np.cos(neighborAnglesInRad * 6).mean() + np.sin(neighborAnglesInRad * 6).mean() * 1j

                dfNeighbors.at[raftID, 'frameNum'] = currStepNum
                dfNeighbors.at[raftID, 'raftID'] = raftID
                dfNeighbors.at[raftID, 'hexaticOrderParameter'] = raftHexaticOrderParameter
                dfNeighbors.at[raftID, 'neighborDistances'] = neighborDistances
                dfNeighbors.at[raftID, 'neighborDistancesAvg'] = neighborDistances.mean()

            # calculate order parameters for the current time step:
            hexaticOrderParameterList = dfNeighbors['hexaticOrderParameter'].tolist()
            neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())

            hexaticOrderParameterArray = np.array(hexaticOrderParameterList)
            hexaticOrderParameterAvgs[currStepNum] = hexaticOrderParameterArray.mean()
            hexaticOrderParameterAvgNorms[currStepNum] = np.sqrt(hexaticOrderParameterAvgs[currStepNum].real ** 2
                                                                 + hexaticOrderParameterAvgs[currStepNum].imag ** 2)
            hexaticOrderParameterMeanSquaredDeviations[currStepNum] = ((hexaticOrderParameterArray
                                                                        - hexaticOrderParameterAvgs[
                                                                            currStepNum]) ** 2).mean()
            hexaticOrderParameterModulii = np.absolute(hexaticOrderParameterArray)
            hexaticOrderParameterModuliiAvgs[currStepNum] = hexaticOrderParameterModulii.mean()
            hexaticOrderParameterModuliiStds[currStepNum] = hexaticOrderParameterModulii.std()

            count, _ = np.histogram(np.asarray(neighborDistancesList) / raftRadius, binEdgesNeighborDistances)
            entropyByNeighborDistances[currStepNum] = fsr.shannon_entropy(count)

            # g(r) and g6(r) for this frame
            for radialIndex, radialIntervalStart in enumerate(radialRangeArray):
                radialIntervalEnd = radialIntervalStart + deltaR
                # g(r)
                js, ks = np.logical_and(pairwiseDistances >= radialIntervalStart,
                                        pairwiseDistances < radialIntervalEnd).nonzero()
                count = len(js)
                density = numOfRafts / arenaSize ** 2
                radialDistributionFunction[currStepNum, radialIndex] = \
                    count / (2 * np.pi * radialIntervalStart * deltaR * density * ( numOfRafts - 1))
                # g6(r)
                sumOfProductsOfPsi6 = \
                    (hexaticOrderParameterArray[js] * np.conjugate(hexaticOrderParameterArray[ks])).sum().real
                spatialCorrHexaOrderPara[currStepNum, radialIndex] = \
                    sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
                # g6(r)/g(r)
                if radialDistributionFunction[currStepNum, radialIndex] != 0:
                    spatialCorrHexaBondOrientationOrder[currStepNum, radialIndex] = \
                        spatialCorrHexaOrderPara[currStepNum, radialIndex] / radialDistributionFunction[
                            currStepNum, radialIndex]

            dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors,ignore_index=True)

        # draw current frame
        if (outputImageSeq == 1 or outputVideo == 1) and (currStepNum % intervalBetweenFrames == 0):
            currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                                      np.int32(raftLocations[:, currStepNum, :] / scaleBar),
                                                      np.int64(raftRadii / scaleBar), numOfRafts)
            if numOfRafts == 2:
                currentFrameBGR = fsr.draw_b_field_in_rh_coord(currentFrameBGR, magneticFieldDirection)
            currentFrameBGR = fsr.draw_cap_peaks_rh_coord(currentFrameBGR,
                                                          np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                                          raftOrientations[:, currStepNum], 6, capillaryPeakOffset,
                                                          np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = fsr.draw_raft_orientations_rh_coord(currentFrameBGR,
                                                                  np.int64(
                                                                      raftLocations[:, currStepNum, :] / scaleBar),
                                                                  raftOrientations[:, currStepNum],
                                                                  np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                                         np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                                         numOfRafts)
            if numOfRafts == 2:
                vector1To2SingleFrame = raftLocations[1, currStepNum, :] - raftLocations[0, currStepNum, :]
                distanceSingleFrame = np.sqrt(vector1To2SingleFrame[0] ** 2 + vector1To2SingleFrame[1] ** 2)
                phase1To2SingleFrame = np.arctan2(vector1To2SingleFrame[1], vector1To2SingleFrame[0]) * 180 / np.pi
                currentFrameBGR = fsr.draw_frame_info(currentFrameBGR, currStepNum, distanceSingleFrame,
                                                      raftOrientations[0, currStepNum], magneticFieldDirection,
                                                      raftRelativeOrientationInDeg[0, 1, currStepNum])
            elif numOfRafts > 2:
                currentFrameBGR = fsr.draw_frame_info_many(currentFrameBGR, currStepNum,
                                                           hexaticOrderParameterAvgNorms[currStepNum],
                                                           hexaticOrderParameterModuliiAvgs[currStepNum],
                                                           entropyByNeighborDistances[currStepNum])

            if outputImageSeq == 1:
                outputImageName = outputFileName + str(currStepNum).zfill(7) + '.jpg'
                cv.imwrite(outputImageName, currentFrameBGR)
            if outputVideo == 1:
                videoOut.write(np.uint8(currentFrameBGR))

    if outputVideo == 1:
        videoOut.release()

    tempShelf = shelve.open(outputFileName)
    expDict = {}
    for key in listOfVariablesToSave:
        try:
            tempShelf[key] = globals()[key]
            expDict[key] = tempShelf[key]
        except TypeError:
            #
            # __builtins__, tempShelf, and imported modules can not be shelved.
            #
            # print('ERROR shelving: {0}'.format(key))
            pass
    tempShelf.close()
    
    # save .mat file for matlab
    outputDataFileName_mat = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_' + str(magneticFieldRotationRPSY).zfill(3) + '_Xx' \
                            + str(magneticFieldRotationRPSX) + str(forceDueToCurvature) + '_startPosMeth' + str(initialPositionMethod) + '_lubEqThres' \
                            + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's_' + '_matlab' 
           
    from scipy.io import savemat
    savemat(outputDataFileName_mat + ".mat", expDict)
    
    lastPositionOfPreviousSpinSpeeds[:,:] = raftLocations[:,currStepNum,:]
    lastOrientationOfPreviousSpinSpeeds = raftOrientations[:,currStepNum]
    lastOmegaOfPreviousSpinSpeeds[:] = raftRotationSpeedsInRad[:, currStepNum]
