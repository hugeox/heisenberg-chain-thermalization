"""
To look at magnetization dependence of perpendicular observables from MonteCarlo


"""
import sys
sys.path.append("../code")
sys.path.append("..")
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from math import cos, pi, sin, log

import spinlib
import pickle
import spin_solver
import s0_factory
import h5py


N = 1000
delta = 400
starting_counter = 400
n_evals = 100

for delta in [200,250,500,800][1:2]:
    f_in = h5py.File('../data/mc_observables_{}_{}.hdf5'.format(delta,0), 'r')
    s_perp_vars = np.array(f_in["s_perp_vars"])
    bond_perp_means = np.array(f_in["bond_perp_means"])
    mags = np.array(f_in["mags"])
    f_in.close()
    for k in range(1,5):
        if os.path.isfile('../data/mc_observables_{}_{}.hdf5'.format(delta,k)):
            f_in = h5py.File('../data/mc_observables_{}_{}.hdf5'.format(delta,k), 'r')
            s_perp_vars = np.concatenate((s_perp_vars,np.array(f_in["s_perp_vars"])))
            bond_perp_means = np.concatenate((bond_perp_means,np.array(f_in["bond_perp_means"])))
            mags = np.concatenate((mags,np.array(f_in["mags"])))
            f_in.close()
    print(delta, np.mean(mags))
    plt.hist(mags,range=(0,0.5),bins = 40, density = True)
    plt.grid()
    plt.show()
    quit()
    means = np.array(s_perp_vars)[:,0]
    #means = np.array(bond_perp_means)[:,0]

    f_in.close()
    magds = []
    ts =[]
    for i in range(20):
        step = 0.025
        means_low_mag = [x[0] for x in zip(means,mags) if (x[1]<step*i+step and x[1] >step*i)]
        if len(means_low_mag) > 600:
            ts.append(np.mean(means_low_mag))
            magds.append(i*step)
            print(len(means_low_mag))

    if delta ==250:
        plt.plot(magds,ts,label = r"$\delta = $" + str(0.001*delta) + ", N =100")
    else:
        plt.plot(magds,ts,label = r"$\delta = $" + str(0.001*delta) + ", N =1000")
    print(dict(zip(magds,ts)))
k =np.linspace(0,0.25,30)

plt.xlabel(r"$|\vec{m}|$")
plt.ylabel(r"$\operatorname{Var}_{perp}$")
plt.grid()
plt.legend()
plt.show()


