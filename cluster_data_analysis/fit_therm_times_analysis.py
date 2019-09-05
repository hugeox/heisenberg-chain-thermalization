import sys
import os
sys.path.append("..")
sys.path.append("../code")
import data_analysis_lib
import numpy as np
from scipy.ndimage.filters import convolve1d
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from math import cos, pi, sin, log,acos
from observables import bond_perp_mean,s_perp_var
import h5py

import s0_factory
import spinlib
import spin_solver
import scipy
import pickle

"""
To analyze observable thermalization time dependence on energy density

assumed directory structure:
    all data assumed stored in os.environ['CHAIN_PROJECT_DIR'] folder,
    in a hierarchical structure
    z.b.:
           if  (os.environ['CHAIN_PROJECT_DIR'] == ~/chain)
        then observables for N = 100, delta = 20 are stored in folder:
            ~/chain/N_100/delta_20/observables/
    
"""


pickle_name = "../data/var_triggers_all_new.pickle"

PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  

use_cache = True

deltas = [10,20,40,100,150,200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
          950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,
          1900,1960,1980,1990]
Ns =[100,250,500,1000]
number_of_batches = 200 #for N = 1000 - 200 batches, for others only 100, but it is not important
n_evals = 100
if os.path.isfile(pickle_name) and use_cache:
    print("Restoring cache named: ", pickle_name)
    with open(pickle_name, 'rb') as fp:
        t_odes,therm_times_all,lyaps, magnetizations,therm_sequence_all,angles= pickle.load(fp)
else:
    t_odes,therm_times_all,lyaps,magnetizations,therm_sequence_all,angles =\
                        data_analysis_lib.generate_data(Ns, deltas,
                                        n_evals, number_of_batches, OBS_TEMPLATE, LYAP_TEMPLATE)
    with open(pickle_name, 'wb') as fp:
        print("Caching into:", pickle_name)
        pickle.dump((t_odes,therm_times_all,lyaps,magnetizations,therm_sequence_all,angles),fp)

i = 0
Ns =[1000]
N=Ns[0]
deltas = t_odes[Ns[0]].keys()
print(t_odes[N])

cmap=plt.get_cmap('tab10')
data_analysis_lib.full()
#methods = ['abs','rel','','abs0.03','rel0.8','first']

if False:
    observable = "s_z_var"
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.02",
                              t_odes,fit_range = (2,16), multiplier =1000)
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.04",
                              t_odes,fit_range = (2,16), multiplier =100)
    #data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.01",
    #                          t_odes,fit_range = (3,10), multiplier =100)
    #data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.04",t_odes,fit_range = (5,13),multiplier =10)
    #data_analysis_lib.plot_smoothed(therm_sequence_all,500,observable,"abs0.01",t_odes,fit_range = (2,12))
    #data_analysis_lib.plot_means(therm_times_all,N,observable,"abs0.03",fit_range = (5,13), multiplier = 0.1)
    data_analysis_lib.plot_means(therm_times_all,N,observable,"first",fit_range = (3,15), multiplier = 0.01)
    #data_analysis_lib.plot_means(therm_times_all,N,observable,"abs",fit_range = (2,15), multiplier = 0.0001)
    plt.suptitle(r"$\operatorname{Var}_z$ ")
    #data_analysis_lib.plot_means(therm_times_all,500,observable,"rel",fit_range = (0,8))
else:
    observable = "bondz_mean"
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.02",
                              t_odes,fit_range = (2,11), multiplier =1)
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.04",t_odes,fit_range = (2,10),multiplier =1)
    plt.suptitle(r"$\epsilon_z$ ")
    


plt.grid()
plt.legend()
plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
plt.ylabel("Thermalization time ")

plt.show()

exit()

mags =[]
hams_lyap =[]
for delta in magnetizations[100].keys():
    mags.append(np.mean(magnetizations[100][delta]))
    if delta < 1000:
        hams_lyap.append(0.001*delta)
    else:
        hams_lyap.append(2-0.001*delta)

plt.scatter(hams_lyap,mags,label = "Magnetizations")
plt.ylabel(r"$\overline{|\vec{m}|}$")
plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
plt.legend()
plt.show()
hams_lyap = []
lyap_exps = []
mean_lyaps = []
lyap_errs = []
mags = []
for delta in deltas:
    N=100
    mags.append(np.mean(magnetizations[N][delta]))
    """Lyaps"""
    if delta < 1000:
        hams_lyap.append(0.001*delta)
    else:
        hams_lyap.append(2-0.001*delta)
    #mean_lyaps.append(0.1*np.mean(lyaps[N][delta]))
    print(delta, len(lyaps[N][delta]))
    mean_lyaps.append(0.01*np.reciprocal(np.mean(lyaps[N][delta])))
    lyap_errs.append(0.01*np.sqrt(np.var(np.reciprocal(lyaps[N][delta])))/len(lyaps[N][delta])**0.5)

"""Plotting lyapunovs"""
if True and (N==500 or N==100):
    
    #plt.scatter(hams_lyap,np.array(mean_lyaps), marker='x')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.errorbar(hams_lyap, np.array(mean_lyaps),lyap_errs, label = "mean of Lyapunov times for N = " + str(N),linestyle = '--')
plt.show()



