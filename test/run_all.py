
import sys
import os
sys.path.append("..")
sys.path.append("../code")
sys.path.append("../cluster_data_analysis")

import s0_factory
import numpy as np

import random
from math import sin,cos, acos
from observables import  s_z_var, bond_mean, bond_var, bond2_var, bond2_mean,bondz_mean, bondz_var, bondz2_mean

import matplotlib.pyplot as plt

import spinlib
import spin_solver
import time
import data_analysis_lib

import h5py

#betas calculated for N = 100 xy-model , assume same for different N
betas = {800: 0.396728515625, 450: 1.319580078125, 900: 0.181884765625, 100: 5.387500000000001, 950: 0.057373046875, 550: 0.975341796875, 200: 2.855224609375, 10: 53.6, 300: 2.061767578125, 750: 0.511474609375, 400: 1.485595703125, 40: 13.049999999999999, 850: 0.299072265625, 20: 26.6, 150: 3.599853515625, 600: 0.860595703125, 500: 1.192626953125, 250: 2.379150390625, 700: 0.614013671875, 650: 0.736083984375, 350: 1.744384765625, 1000: 0, 1050: -0.057373046875, 1100: -0.181884765625, 1150: -0.299072265625, 1200: -0.396728515625, 1250: -0.511474609375, 1300: -0.614013671875, 1350: -0.736083984375, 1400: -0.860595703125, 1450: -0.975341796875, 1500: -1.192626953125, 1550: -1.319580078125, 1600: -1.485595703125, 1650: -1.744384765625, 1700: -2.061767578125, 1750: -2.379150390625, 1800: -2.855224609375, 1850: -3.599853515625, 1900: -5.387500000000001, 1960: -13.049999999999999, 1980: -26.6, 1990: -53.6,80:6.6,120:4.37,140:3.856,180:3.06}

Ns = [50]
n_evals = 100
number_of_batches = 5

S0_TEMPLATE = 's0_{}'
SF_TEMPLATE = 'sf_{}'
OBSERVABLE_DATASET_TEMPLATE = '{}_{}'
OBS_TEMPLATE = 're_observables_N_{}_delta_{}_{}_{}.hdf5'

deltas = [200,300,400,500,600,700,800]
for batch_no in range(number_of_batches):
    starting_counter = batch_no * n_evals
    for N in Ns:
        for delta in deltas:
            s0_factory.get_s0_xy_mc(N = N, delta = delta, beta = betas[delta], starting_counter = starting_counter, n_evals =n_evals)

            t_ode = 200
            n_evals_ode = 400/t_ode

            f_in = h5py.File('s0_{}_{}.hdf5'.format(starting_counter,n_evals), 'r')
            f_out = h5py.File(OBS_TEMPLATE.format(N,delta, starting_counter,n_evals), 'w')
            f_out.attrs['N'] = N
            f_out.attrs['delta'] = f_in.attrs['delta']
            f_out.attrs['beta'] = f_in.attrs['beta']
            f_out.attrs['starting_counter'] = starting_counter
            f_out.attrs['n_evals'] = n_evals
            f_out.attrs['t_ode'] = t_ode
            f_out.attrs['n_evals_ode'] = n_evals_ode
            f_out.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            observables = [s_z_var, bondz_mean, bondz2_mean]
            observables_str = [x.__name__ for x in observables]

            for obs in observables_str:
                f_out.create_group(obs)
            f_out.create_group("s0")
            f_out.create_group("sf")
            for i in range(starting_counter, starting_counter + n_evals):
                s0 = np.array(f_in[S0_TEMPLATE.format(i)])
                solver = spin_solver.SpinSolver(1, N, t_ode,s0)
                obs, si, sf = solver.solve_return_observables(n_evals_ode,observables, rtol = 1e-6,atol = 1e-9, print_time = False)

                f_out["s0"].create_dataset(S0_TEMPLATE.format(i), data = s0)
                f_out["sf"].create_dataset(SF_TEMPLATE.format(i), data = sf)

                for x, y  in zip(obs,observables_str):
                    f_out[y].create_dataset(OBSERVABLE_DATASET_TEMPLATE.format(y,i), data = x)
            f_in.close()
            f_out.close()

t_odes,therm_times_all,lyaps,magnetizations,therm_sequence_all,angles =\
                        data_analysis_lib.generate_data(Ns, deltas,
                                        n_evals, number_of_batches, OBS_TEMPLATE)
N=Ns[0]
deltas = t_odes[Ns[0]].keys()

observable = "s_z_var"
plt.xscale("log")
plt.yscale("log")
data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.04",
                          t_odes,fit_range = (0,-1))

#plt.xscale("linear")
#plt.yscale("linear")

plt.grid()
plt.legend()
plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
plt.ylabel("Thermalization time ")

plt.grid()
plt.show()
