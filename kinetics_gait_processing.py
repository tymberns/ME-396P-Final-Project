#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:54:35 2026

@author: tyberns
"""

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys

baseDir = os.getcwd()
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(baseDir)
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

session_id = "99d7534a-b172-4908-9d41-2fd47289be97"
trial_name = 'walk1'

case = '0' # Change this to compare across settings.
motion_type = 'walking'

time_window = [3.75, 5.25]
    
# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyze the results of the optimal control problem. If you
# solved the problem already, and only want to analyze/process the results, you
# can set solveProblem to False and run this script with analyzeResults set to
# True. This is useful if you do additional post-processing but do not want to
# re-run the problem.
analyzeResults = True

# Path to where you want the data to be downloaded.
dataFolder = os.path.join(baseDir, session_id)

# %% Setup. 
if not 'time_window' in locals():
    time_window = None
if not 'repetition' in locals():
    repetition = None
if not 'treadmill_speed' in locals():
    treadmill_speed = 0
if not 'contact_side' in locals():
    contact_side = 'all'
settings = processInputsOpenSimAD(baseDir, dataFolder, session_id, trial_name, 
                                  motion_type, time_window, repetition,
                                  treadmill_speed, contact_side)

# %% Simulation.
run_tracking(baseDir, dataFolder, session_id, settings, case=case, 
              solveProblem=solveProblem, analyzeResults=analyzeResults)

# %% Plots.
# To compare different cases, add to the cases list, eg cases=['0','1'].
plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, cases=[case])