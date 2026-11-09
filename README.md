# ME-396P-Final-Project
Repository for Engineering Application for Engineers Final Project

## To be able to run all Open Cap processing code:
Visit https://github.com/opencap-org/opencap-processing and follow the README

## To calculate gait metrics:
Download gait_analysis_edited.py and gait_metrics.py, and run gait_metrics.py from the same working directory as the files downloaded above

## To calculate gait kinetics, ground reaction forces, and muscle forces:
Download settingsOpenSimAD.py from this repo and replace the file with the same name in UtilsDynamicSimulations > OpenSimAD (alternatively, you can just open up that file and change the 'ipopt_tolerance' under the walking setup - line 204 - to 2)
Download and run kinetics_gait_processing.py in the same working directory
Note: This code will take some time to run, as it is an optimizer. However, this should only be 10-20 minutes for the length trials we are processing.

Note: There are small ##TODO lines in each gait_metrics and kinetics_gait_processing, as the current session id/file name and trail name are hardcoded in. However, this will need to change to read the names of the folder and trial name of the inputted folder if we want users to be able to simply drag and drop.
