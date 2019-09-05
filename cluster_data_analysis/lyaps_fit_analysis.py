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
To analyze Lyapunovs for N = 100 
"""


pickle_name = "../data/100_more_lyaps.pickle"

PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  

use_cache = True

deltas = [10,20,40,80,100,120,140,150,180,200,250,300]
Ns =[100]
number_of_batches =100 #number of batches
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
Ns =[100]
N=Ns[0]
deltas = list(t_odes[Ns[0]].keys())
print(t_odes[N])

cmap=plt.get_cmap('tab10')
data_analysis_lib.full()
#methods = ['abs','rel','','abs0.03','rel0.8','first']

hams_lyap = []
lyap_exps = []
mean_lyaps = []
lyap_errs = []
mags = []
for delta in deltas[2:-2]:
    N=100
    mags.append(np.mean(magnetizations[N][delta]))
    """Lyaps"""
    if delta < 1000:
        hams_lyap.append(0.001*delta)
    else:
        hams_lyap.append(2-0.001*delta)
    print("Number of Lyaps for delta: ", delta, len(lyaps[N][delta]))
    mean_lyaps.append(np.mean(lyaps[N][delta]))
    lyap_errs.append(np.sqrt(np.var(lyaps[N][delta]))/len(lyaps[N][delta])**0.5)

"""Plotting lyapunovs"""
if True and (N==500 or N==100):
    
    #plt.scatter(hams_lyap,np.array(mean_lyaps), marker='x')
    
    start = 2
    end = None
    hams_lyap_log = np.log(hams_lyap)
    mean_lyaps_log = np.log(mean_lyaps)
    
    coefficients,residuals, a ,b ,c= np.polyfit(hams_lyap_log[start:end],
                                                mean_lyaps_log[start:end], 1, full = True)
    print(N, "Lyapunov residuals are ", residuals[0])
    polynomial = np.poly1d(coefficients)
    ys = polynomial(hams_lyap_log)
    plt.plot(hams_lyap[start:end], np.exp(ys[start:end]),color = cmap.colors[1])
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$\overline{\lambda}$")
    plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
    plt.grid()
    plt.errorbar(hams_lyap, np.array(mean_lyaps),lyap_errs,
            label = "mean maximal Lyapunov exponent for N = " + str(N) +", exponent " + str(coefficients[0])[:7],
            linestyle = '--')
plt.legend()
plt.show()
