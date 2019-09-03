"""
Uses initial states stored in s0_{starting_counter}_{n_evals}.hdf5 file and computes observables
that are given in the list observables.
These are then stored in f_out, named re_observables_{starting_counter}_{n_evals}.hdf5

Command-line arguments:

N - length of spin chain
starting_counter - specifies at what index s0_factory starts generating initial states
n_evals - specifies how many initial states are generated
t_ode - integration time for ODE integration
n_evals_ode - number of evaluations per second for ODE integration
"""
import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from math import cos, pi, sin, log

import spinlib
import argparse
import spin_solver
import s0_factory
from observables import  s_z_var, bond_mean, bond_var, bond2_var, bond2_mean,bondz_mean, bondz_var, bondz2_var, bondz2_mean, bondx_var,bondy_var
import h5py



parser = argparse.ArgumentParser(
    description='N, index')
parser.add_argument('N', metavar='N', type=int, help='linear system size')
parser.add_argument('starting_counter', metavar='starting_counter',type = int, help = 'for which s0 to solve')
parser.add_argument('n_evals', metavar='n_evals',type = int, help = 'number of evals to compute')
parser.add_argument('t_ode', metavar='t_ode',type = float, help = 'integration time')
parser.add_argument('n_evals_ode', metavar='n_evals_ode',type = float, help = 'evaluations per second')
args = parser.parse_args()
N = args.N
starting_counter = args.starting_counter
t_ode = args.t_ode
n_evals = args.n_evals
n_evals_ode = args.n_evals_ode
print("N = ", N,", starting_counter = ", starting_counter, ", t_ode = ", t_ode, ", n_evals = ", n_evals,
", n_evals_ode", n_evals_ode)

OBSERVABLE_TEMPLATE = '{}_{}'
S0_TEMPLATE = 's0_{}'
SF_TEMPLATE = 'sf_{}'

f_in = h5py.File('s0_{}_{}.hdf5'.format(starting_counter,n_evals), 'r')
f_out = h5py.File('re_observables_{}_{}.hdf5'.format(starting_counter,n_evals), 'w')
f_out.attrs['N'] = N
f_out.attrs['delta'] = f_in.attrs['delta']
f_out.attrs['beta'] = f_in.attrs['beta']
f_out.attrs['starting_counter'] = starting_counter
f_out.attrs['n_evals'] = n_evals
f_out.attrs['t_ode'] = t_ode
f_out.attrs['n_evals_ode'] = n_evals_ode
f_out.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
observables = [s_z_var, bond2_mean,bondz_mean, bondz_var, bondz2_var, bondz2_mean]
observables_str = [x.__name__ for x in observables]

for obs in observables_str:
    f_out.create_group(obs)
f_out.create_group("s0")
f_out.create_group("sf")
st = time.time()
for i in range(starting_counter, starting_counter + n_evals):
    s0 = np.array(f_in[S0_TEMPLATE.format(i)])
    solver = spin_solver.SpinSolver(1, N, t_ode,s0)
    obs, si, sf = solver.solve_return_observables(n_evals_ode,observables, rtol = 1e-6,atol = 1e-9, print_time = False)

    f_out["s0"].create_dataset(S0_TEMPLATE.format(i), data = s0)
    f_out["sf"].create_dataset(SF_TEMPLATE.format(i), data = sf)

    for x, y  in zip(obs,observables_str):
        f_out[y].create_dataset(OBSERVABLE_TEMPLATE.format(y,i), data = x)

f_in.close()
f_out.close()
end = time.time()
print("Time taken: ", end-st)

