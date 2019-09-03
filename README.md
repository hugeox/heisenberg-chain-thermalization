

### Directory structure

# code
All the code needed to run the simulation on the cluster is here. 

NB: One needs to load intelpython3 module (using  module load intelpython3) 
in order for this code to run at a reasonable speed.

# run_jobs
scripts to run the simulation using slurm

# test
Various test files and checks.
In file run_all, one can run a less demanding simulation on their workstation.

# cluster_data_analysis
code to analyse the cluster data is in fit_therm_times_analysis.py First the raw cluster data is turned into a dataset containing thermalizaion times and various other quantities. This is then cached.
The code computing thermalization times is in data_analysis_lib.py

