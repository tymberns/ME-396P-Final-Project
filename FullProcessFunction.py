#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:11:32 2026

@author: tyberns
"""

import os
import sys
import pandas as pd
import numpy as np
from gait_analysis_edited import gait_analysis_edited as ga
from utilsOpenSimAD import processInputsOpenSimAD
from mainOpenSimAD import run_tracking
    
def FullProcess(sessionID, trialName):
    sys.path.append("./")
    sys.path.append("./ActivityAnalyses")
    # from utils import get_trial_id, download_trial
    # from utilsPlotting import plot_dataframe_with_shading
    
    # %% Helper Functions
    def read_mot(path):
        # Find the line where the header ends
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if line.strip().lower() == "endheader":
                    header_rows = i + 1
                    break
            else:
                raise ValueError(f"No 'endheader' found in {path}")
    
        df = pd.read_csv(path, sep=r"\s+", skiprows=header_rows)
        return df
    
    # GRF vector norm
    def force_magnitude(side):
        return np.sqrt(grfs[f'ground_force_{side}_vx'] ** 2 +
                       grfs[f'ground_force_{side}_vy'] ** 2 +
                       grfs[f'ground_force_{side}_vz'] ** 2)
    
    # Muscle group dataframe creation
    def group_df(abbrevs):
        df = muscles_df.loc[abbrevs].copy()
        df.index = df.index.map(muscle_names).str.title()
        df.index.name = 'Muscle Force'
        return df
    
    # %% Paths.
    baseDir = os.path.join(os.getcwd())
    
    scalar_names = {'gait_speed','stride_length','step_width','cadence',
                    'single_support_time','double_support_time','step_length_symmetry', 
                    'stance_time', 'swing_time'}
    
    # Select how many gait cycles you'd like to analyze. Select -1 for all gait
    # cycles detected in the trial.
    n_gait_cycles = -1 
    
    # Select lowpass filter frequency for kinematics data.
    filter_frequency = 6
    
    # %% Gait Kinematics Analysis
    # Get trial id from name.
    # trial_id = get_trial_id(sessionID,trial_name)    
    
    # Set session path.
    sessionDir = os.path.join(baseDir, f"OpenCapData_{sessionID}")
    
    # # Download data. Only works if logged into the same OpenCap account as the desired session
    # trialName = download_trial(trial_id,sessionDir,session_id=sessionID) 
    
    # Init gait analysis.
    gait_r = ga(
        sessionDir, trialName, leg='r',
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        n_gait_cycles=n_gait_cycles)
    gait_l = ga(
        sessionDir, trialName, leg='l',
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        n_gait_cycles=n_gait_cycles)
    
    timestart = gait_r.startTime
    timeend = gait_r.endTime
    
    # Compute scalars and get time-normalized kinematic curves.
    gaitResults = {}
    gaitResults['scalars_r'] = gait_r.compute_scalars(scalar_names)
    gaitResults['curves_r'] = gait_r.get_coordinates_normalized_time()
    gaitResults['scalars_l'] = gait_l.compute_scalars(scalar_names)
    gaitResults['curves_l'] = gait_l.get_coordinates_normalized_time()    
    
    hip_r_rom = gait_r.compute_rom('hip_flexion')
    hip_l_rom = gait_l.compute_rom('hip_flexion')
    knee_r_rom = gait_r.compute_rom('knee_angle')
    knee_l_rom = gait_l.compute_rom('knee_angle')
    ankle_r_rom = gait_r.compute_rom('ankle_angle')
    ankle_l_rom = gait_l.compute_rom('ankle_angle')
    
    hip_r_peakangle = gait_r.compute_peak_angle('hip_flexion')
    hip_l_peakangle = gait_l.compute_peak_angle('hip_flexion')
    knee_r_peakangle = knee_r_rom
    knee_l_peakangle = knee_l_rom
    ankle_r_peakangle = gait_r.compute_peak_angle('ankle_angle')
    ankle_l_peakangle = gait_l.compute_peak_angle('ankle_angle')
    
    CoM = gait_r.comValues()
    
    gaitResults['scalars_r']['hip_ROM'] = {'value': hip_r_rom[0], 'units': hip_r_rom[1]}
    gaitResults['scalars_l']['hip_ROM'] = {'value': hip_l_rom[0], 'units': hip_l_rom[1]}
    gaitResults['scalars_r']['knee_ROM'] = {'value': knee_r_rom[0], 'units': knee_r_rom[1]}
    gaitResults['scalars_l']['knee_ROM'] = {'value': knee_l_rom[0], 'units': knee_l_rom[1]}
    gaitResults['scalars_r']['ankle_ROM'] = {'value': ankle_r_rom[0], 'units': ankle_r_rom[1]}
    gaitResults['scalars_l']['ankle_ROM'] = {'value': ankle_l_rom[0], 'units': ankle_l_rom[1]}
    
    gaitResults['scalars_r']['hip_peak_angle'] = {'value': hip_r_peakangle[0], 'units': hip_r_peakangle[1]}
    gaitResults['scalars_l']['hip_peak_angle'] = {'value': hip_l_peakangle[0], 'units': hip_l_peakangle[1]}
    gaitResults['scalars_r']['knee_peak_angle'] = {'value': knee_r_peakangle[0], 'units': knee_r_peakangle[1]}
    gaitResults['scalars_l']['knee_peak_angle'] = {'value': knee_l_peakangle[0], 'units': knee_l_peakangle[1]}
    gaitResults['scalars_r']['ankle_peak_angle'] = {'value': ankle_r_peakangle[0], 'units': ankle_r_peakangle[1]}
    gaitResults['scalars_l']['ankle_peak_angle'] = {'value': ankle_l_peakangle[0], 'units': ankle_l_peakangle[1]}
    
    # %% Print scalar results as a dataframe.
    # Each scalar is stored as {'value': ..., 'units': ...}, so building a
    # DataFrame from the dict gives one column per metric; transpose so each
    # metric is a row.
    scalars_r_df = pd.DataFrame(gaitResults['scalars_r']).T
    scalars_l_df = pd.DataFrame(gaitResults['scalars_l']).T
    
    gait_metrics_df = pd.DataFrame({
        'Right': pd.to_numeric(scalars_r_df['value']),
        'Left': pd.to_numeric(scalars_l_df['value']),
        'Units': scalars_r_df['units'],
    }).round(2)
    gait_metrics_df.index.name = 'Metric'
    gait_metrics_df.index = gait_metrics_df.index.str.replace('_', ' ').str.upper()
    
    # %% Dynamics
    baseDir = os.getcwd()
    opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
    sys.path.append(baseDir)
    sys.path.append(opensimADDir)
    
    case = '0' # Change this to compare across settings.
    motion_type = 'walking'
    
    time_window = [timestart[0], timeend[0]]
        
    # Set to True to solve the optimal control problem.
    solveProblem = True
    # Set to True to analyze the results of the optimal control problem. If you
    # solved the problem already, and only want to analyze/process the results, you
    # can set solveProblem to False and run this script with analyzeResults set to
    # True. This is useful if you do additional post-processing but do not want to
    # re-run the problem.
    analyzeResults = True
    
    # Path to where you want the data to be downloaded.
    dataFolder = os.path.join(baseDir)
    
    # Setup 
    if not 'time_window' in locals():
        time_window = None
    if not 'repetition' in locals():
        repetition = None
    if not 'treadmill_speed' in locals():
        treadmill_speed = 0
    if not 'contact_side' in locals():
        contact_side = 'all'
    settings = processInputsOpenSimAD(baseDir, dataFolder, sessionID, trialName, 
                                      motion_type, time_window, repetition,
                                      treadmill_speed, contact_side)
    
    # Make sure the Dynamics output folder exists
    resultsDir = os.path.join(dataFolder, sessionID, 'OpenSimData', 'Dynamics', trialName)
    os.makedirs(resultsDir, exist_ok=True)
    
    # Simulation.
    run_tracking(baseDir, dataFolder, sessionID, settings, case=case, 
                  solveProblem=solveProblem, analyzeResults=analyzeResults)
    
    # Dynamics Dataframes
    sys.path.append("../")
    sys.path.append("../ActivityAnalyses")
    
    # %% Moments
    jointmoments = read_mot(f"{sessionID}/OpenSimData/Dynamics/{trialName}/kinetics_{trialName}_0.mot")
    jointmoments = jointmoments[['hip_flexion_r_moment', 'hip_adduction_r_moment', 
                                 'hip_rotation_r_moment', 'knee_angle_r_moment', 
                                 'ankle_angle_r_moment', 'hip_flexion_l_moment', 
                                 'hip_adduction_l_moment', 'hip_rotation_l_moment', 
                                 'knee_angle_l_moment', 'ankle_angle_l_moment']]
    maxmoments = jointmoments.abs().max()
    
    # Split names like 'hip_flexion_r_moment' into joint + side
    names = maxmoments.index.str.replace('_moment', '', regex=False)
    momentside = names.str[-1]      # 'r' or 'l'
    joint = names.str[:-2]
    
    moments_df = (
        maxmoments.set_axis(pd.MultiIndex.from_arrays([joint, momentside]))
               .unstack()
               .rename(columns={'r': 'Right (Nm)', 'l': 'Left (Nm)'})[['Right (Nm)', 'Left (Nm)']]
               .round(2)
    )
    moments_df.index = moments_df.index + ' Moment'
    moments_df.index = moments_df.index.str.replace('_', ' ').str.title()
    moments_df.index.name = 'Moment'
    
    # %% GRFs
    grfs = read_mot(f"{sessionID}/OpenSimData/Dynamics/{trialName}/GRF_resultant_{trialName}_0.mot")
    
    grf_df = pd.DataFrame(
        {'Right Foot (N)': [force_magnitude('right').max()],
         'Left Foot (N)':  [force_magnitude('left').max()]},
        index=['Peak Ground Reaction Force']
    ).round(2)
    grf_df.index.name = 'Force'
    
    # %% Muscles
    muscles = read_mot(f"{sessionID}/OpenSimData/Dynamics/{trialName}/forces_{trialName}_0.mot")
    muscles = muscles[[c for c in muscles.columns if c != 'time' and c.endswith(('_l', '_r'))]]
    maxmuscles = muscles.abs().max()
    
    muscle_names = {
        'bflh': 'biceps femoris long head', 'bfsh': 'biceps femoris short head',
        'semimem': 'semimembranosus',
        'gaslat': 'gastrocnemius lateralis', 'gasmed': 'gastrocnemius medialis',
        'soleus': 'soleus', 'tibant': 'tibialis anterior',
        'glmax1': 'gluteus maximus 1', 'glmax2': 'gluteus maximus 2', 'glmax3': 'gluteus maximus 3',
        'glmed1': 'gluteus medius 1', 'glmed2': 'gluteus medius 2', 'glmed3': 'gluteus medius 3',
        'glmin1': 'gluteus minimus 1', 'glmin2': 'gluteus minimus 2', 'glmin3': 'gluteus minimus 3',
        'psoas': 'psoas', 'recfem': 'rectus femoris',
        'vasint': 'vastus intermedius', 'vaslat': 'vastus lateralis', 'vasmed': 'vastus medialis',
    }
    
    # muscle_groups = {
    #     'Loading Response (Hip Flexors)':        ['glmax1', 'glmax2', 'glmax3'],
    #     'Stance Stability (Hip Abductors)':        ['glmed1', 'glmed2', 'glmed3', 'glmin1', 'glmin2', 'glmin3'],
    #     'Swing Initiation / Swing (Hip Flexors)':         ['psoas'],
    #     'Terminal Swing / Early Stance (Hamstrings)': ['bflh', 'bfsh', 'semimem'],
    #     'Weight Acceptance (Quadriceps)':          ['recfem', 'vasint', 'vaslat', 'vasmed'],
    #     'Push-Off (Plantarflexors)':               ['gaslat', 'gasmed', 'soleus'],
    #     'Initial Contact / Clearance (Dorsiflexors)': ['tibant'],
    # }
    
    # Split muscle names into muscle + side
    side = maxmuscles.index.str[-1]       # 'r' or 'l'
    muscle = maxmuscles.index.str[:-2]
    
    muscles_df = (
        maxmuscles.set_axis(pd.MultiIndex.from_arrays([muscle, side]))
                  .unstack()
                  .rename(columns={'r': 'Right (Nm)', 'l': 'Left (Nm)'})[['Right (Nm)', 'Left (Nm)']]
                  .round(2)
    )
    
    hip_extensors_df  = group_df(['glmax1', 'glmax2', 'glmax3'])
    hip_abductors_df  = group_df(['glmed1', 'glmed2', 'glmed3', 'glmin1', 'glmin2', 'glmin3'])
    hip_flexors_df    = group_df(['psoas'])
    hamstrings_df     = group_df(['bflh', 'bfsh', 'semimem'])
    quadriceps_df     = group_df(['recfem', 'vasint', 'vaslat', 'vasmed'])
    plantarflexors_df = group_df(['gaslat', 'gasmed', 'soleus'])
    dorsiflexors_df   = group_df(['tibant'])
    
    return gait_metrics_df, moments_df, grf_df, hip_extensors_df, hip_abductors_df, hip_flexors_df, hamstrings_df, quadriceps_df, plantarflexors_df, dorsiflexors_df