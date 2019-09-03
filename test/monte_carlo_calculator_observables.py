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
from observables import  s_z_var, bond_mean, bond_var, bond2_var, bond2_mean,bondz_mean, bondz_var, bondz2_var, bondz2_mean, bondx_var,bondy_var, s_perp_var,bond_perp_mean,bond_perp_var, s_perp,bond_perp
import h5py


N = 1000
delta = 400
starting_counter = 400
n_evals = 100
print("N = ", N,", starting_counter = ", starting_counter,  ", n_evals = ", n_evals)

OBSERVABLE_TEMPLATE = '{}_{}'
S0_TEMPLATE = 's0_{}'
SF_TEMPLATE = 'sf_{}'

PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
INIT_TEMPLATE = PATH + "/init_state/s0_{}_{}.hdf5"
f_in = h5py.File(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
observables = [s_perp_var,bond_perp_mean,bond_perp_var]
observables_str = [x.__name__ for x in observables]

results ={}
for obs in observables_str:
    results[obs]=[]
s0 = np.array(f_in[S0_TEMPLATE.format(starting_counter)])
betas = {400:2.44,200:4.81,100:10.5}
solver = spin_solver.SpinSolver(1, N,10 ,s0, beta = betas[delta])
f_in.close()

s_perp_vars = []
s_perps = []
s_perps_2 = []
s_perps_3 = []
bond_perps = []
bond_perp_means = []
s_perp_vars_2 = []
s_perp_vars_3 = []
mags = []
filename = "rand_delta200"
for delta in [100,400]:
    filename = "mc_cache_delta_" + str(delta) +".pickle"#- low magnetization + magnetization dependence
    #filename = "mc_cache_delta_100_long"
    if os.path.isfile(filename) and False:
        with open(filename, 'rb') as fp:
            s_perp_vars,s_perps,bond_perps, bond_perp_means,mags= pickle.load(fp)
    else:
        for l in range(100000):
            solver.mc_update()
            solver.mc_update()
        print(solver.hamilt/N +1)
        for l in range(5000000*100):
            solver.mc_update()
            solver.mc_update()
            if l%40==0 and np.abs(solver.hamilt+0.001*N*(1000-delta))<0.5:
                mags.append(spinlib.return_magnetization_norm(solver.s0))
                s_perp_vars.append(s_perp_var(solver.s0))
                s_perps.append(s_perp(solver.s0)) 
                bond_perps.append(bond_perp(solver.s0))
                bond_perp_means.append(bond_perp_mean(solver.s0))
        with open(filename, 'wb') as fp:
            pickle.dump((s_perp_vars,
                         s_perps,bond_perps,bond_perp_means,mags),fp)

            #obs = spinlib.return_observables(s0,observables)
            #for x, y  in zip(obs,observables_str):
            #    results[y].append(x)




    l = int(len(s_perps)*16/17)
    print(l)
    #plt.hist(np.array(s_perps).ravel(),bins = 30, histtype = 'step', density = True)
    #plt.hist(np.array(s_perps_2).ravel(),bins = 30, histtype = 'step', density = True)
    #plt.show()

    means = np.array(bond_perp_means)[:,1]
    means_low_mag = [3 * x[0] for x in zip(means,mags) if x[1]<0.1]
    print("perp mean",np.mean(means_low_mag))
    means = np.array(bond_perp_means)[:,0]
    means_low_mag = [3 * x[0] for x in zip(means,mags) if x[1]<0.1]
    print("xy mean",np.mean(means_low_mag))

    means = np.array(s_perp_vars)[:,1]
    means_low_mag = [x[0] for x in zip(means,mags) if x[1]<0.1]
    print("perp var",np.mean(means_low_mag))
    means = np.array(s_perp_vars)[:,0]
    means_low_mag = [x[0] for x in zip(means,mags) if x[1]<0.1]
    print("xy var",np.mean(means_low_mag))

    print(np.mean(np.array(bond_perp_means)[:,0]))
    print(np.mean(np.array(bond_perp_means)[:,1]))
    print(np.mean(np.array(bond_perp_means)[:,2]))
    print(np.mean(np.array(s_perp_vars)[:,0]))
    print(np.mean(np.array(s_perp_vars)[:,1]))
    print(np.mean(np.array(s_perp_vars)[:,2]))

    magds = []
    ts =[]
    print("Max mag", max(mags))
    for i in range(20):
        means_low_mag = [x[0] for x in zip(means,mags) if (x[1]<0.05*i+0.05 and x[1] >0.05*i)]
        ts.append(np.mean(means_low_mag))
        print("delta,", delta, "i", i ," len means", len(means_low_mag))
        magds.append(i*0.05)

    plt.plot(magds,ts)


    

#plt.hist(np.array(s_perp_vars)[:,0].ravel(), histtype = 'step',label = r'$\operatorname{Var}_{perp}$', density = True)
#plt.hist(np.array(bond_perp_means)[:,0].ravel(), histtype = 'step',label = r'$\epsilon_{perp}$', density = True)
plt.xlabel("Magnetization")
plt.ylabel(r'$\overline{\operatorname{Var}_{perp}}$')
#plt.hist(np.array(bond_perp_means)[:,1], histtype = 'step',label = 'end', density = True)
#plt.hist(s_perp_vars[l:], histtype = 'step',label = 'xy end', density = True)
#plt.hist(s_perp_vars_2[:-l], histtype = 'step',label = 'start', density = True)
#plt.hist(bond_perp_means, histtype = 'step',label = 'xy perp', density = True)
plt.legend()
plt.show()

