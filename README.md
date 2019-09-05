

### Directory structure

# code
All the code needed to run the simulation on the cluster is here. 

NB: One needs to load intelpython3 module (using  module load intelpython3) 
in order for this code to run at a reasonable speed.

# test
Various test files and checks - checks are assuming the presence of 
cluster results
In file run_all, one can run a less demanding simulation on their workstation.

# cluster_data_analysis
code to analyse the cluster data is in fit_therm_times_analysis.py
First the raw cluster data is turned into a dataset containing thermalizaion times and various other quantities. This is then cached.
One can also analyze the data using only the cached file.
N_dependence_analysis.py analyzes n_dependence of observables.
lyaps_fit_analysis.py analyzes low energy behaviour of lyapunov exponents for.
mc_trial.py analyzes the data using Monte Carlo simulation results for equilibirum values of observables at different magnetizations.

The code computing thermalization times is in data_analysis_lib.py

# data 
This is where all the cached pickles and other data go



