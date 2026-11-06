import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from gait_analysis_edited import gait_analysis_edited as ga
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
session_id = '99d7534a-b172-4908-9d41-2fd47289be97'
trial_name = 'walk1'

scalar_names = {'gait_speed','stride_length','step_width','cadence',
                'single_support_time','double_support_time','step_length_symmetry', 
                'stance_time', 'swing_time'}

# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = -1 

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# %% Gait analysis.
# Get trial id from name.
trial_id = get_trial_id(session_id,trial_name)    

# Set session path.
sessionDir = os.path.join(dataFolder, session_id)

# Download data.
trialName = download_trial(trial_id,sessionDir,session_id=session_id) 

# Init gait analysis.
gait_r = ga(
    sessionDir, trialName, leg='r',
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_gait_cycles=n_gait_cycles)
gait_l = ga(
    sessionDir, trialName, leg='l',
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_gait_cycles=n_gait_cycles)
    
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

# # %% Print scalar results.
# print('\nRight foot gait metrics:')
# print('-----------------')
# for key, value in gaitResults['scalars_r'].items():
#     rounded_value = round(value['value'], 2)
#     print(f"{key}: {rounded_value} {value['units']}")
    
# print('\nLeft foot gait metrics:')
# print('-----------------')
# for key, value in gaitResults['scalars_l'].items():
#     rounded_value = round(value['value'], 2)
#     print(f"{key}: {rounded_value} {value['units']}")

# # %% You can plot multiple curves, in this case we compare right and left legs.
# plot_dataframe_with_shading(
#     [gaitResults['curves_r']['mean'], gaitResults['curves_l']['mean']],
#     [gaitResults['curves_r']['sd'], gaitResults['curves_l']['sd']],
#     leg = ['r','l'],
#     xlabel = '% gait cycle',
#     title = 'kinematics (m or deg)',
#     legend_entries = ['right','left'])