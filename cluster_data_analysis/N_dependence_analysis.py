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
To analyze N dependence
"""


deltas  = [600,800]
Ns =[100,250,500,1000,2000,4000,6000,8000,10000,20000]
evals =100
n_evals = 100

pickle_name = "cache/n_dependence.pickle"
PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  

use_cache = True

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
#deltas = [10,20,40,100,150,200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
#        950]
i = 0
Ns =[100,250,500,1000,2000,4000,6000,8000,10000]
N=Ns[0]
deltas = t_odes[Ns[0]].keys()

data_analysis_lib.full()
cmap=plt.get_cmap('tab10')

observable = "bondz_mean"
observable = "s_z_var"

for delta in [0.6,0.8]:
    t_therms = []
    for N in [100,250,500,1000,2000,4000,6000,8000,10000]:
        #t_therms.append(data_analysis_lib.plot_smoothed(therm_sequence_all,
        #                                        N,observable,"abs0.02",t_odes,plot = False)[delta])
        if delta == 0.6:
            a = 9
        else:
            a =4.9
        t_therms.append(a-data_analysis_lib.plot_means(therm_times_all,
                                                N,observable,"abs",plot = False)[delta])

    plt.scatter(Ns,t_therms,label = r"$\overline{T_{therm}}, \delta = $" + str(delta))
    plt.plot(Ns,t_therms,color = cmap.colors[1])
for delta in [0.6,0.8]:
    t_therms = []
    for N in [100,250,500,1000,2000,4000,6000,8000,10000]:
        t_therms.append(data_analysis_lib.plot_smoothed(therm_sequence_all,
                                                N,observable,"abs0.02",t_odes,plot = False)[delta])

    plt.plot(Ns,t_therms, label = r"$T_{therm}$ from $\overline{\mathcal{O}(t)}, \delta = $" + str(delta))
    print(np.mean(t_therms))
plt.grid()
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel("Thermalization time ")

plt.show()
