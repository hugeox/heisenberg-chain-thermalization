"""
Uses one of the initial states obtained in s0_{starting_counter}_{n_evals}.hdf5 to compute
a Lyapunov exponent. The index of that state is chosen randomly
Length of calculation is determined by variables qr_iters and t_ode
Command-line arguments:

N - length of spin chain
starting_counter - specifies at what index s0_factory starts generating initial states
n_evals - specifies how many initial states are generated


Input file:

s0_{starting_counter}_{n_evals}.hdf5

Output file:

lyaps_{i}.hdf5  (i is a randomly chosen index between starting_counter and starting_counter + n_evals -1)
"""
import sys
import numpy as np

import h5py
import random
import spin_solver
import matplotlib.pyplot as plt
import time
from math import cos, pi, sin, log, exp

import spinlib
import argparse


parser = argparse.ArgumentParser(
    description='N, index')
parser.add_argument('N', metavar='N', type=int, help='linear system size')
parser.add_argument('starting_counter', metavar='starting_counter',type = int, help = 'for which s0 to solve')
parser.add_argument('n_evals', metavar='n_evals',type = int, help = 'number of evals to compute')
args = parser.parse_args()
N = args.N
starting_counter = args.starting_counter
n_evals = args.n_evals

i = starting_counter + random.randrange(0,n_evals)

print("Lyapunov calculation, N = ", N, ", starting_counter = ", starting_counter, ", n_evals = ", n_evals, ", i = ", i)
qr_iters = 1500
t_ode = 5
n_vectors = 1

f_in = h5py.File('s0_{}_{}.hdf5'.format(starting_counter,n_evals), 'r')
f_out = h5py.File('lyap_{}.hdf5'.format(i), 'w')
f_out.attrs['N'] = N
f_out.attrs['delta'] = f_in.attrs['delta']
f_out.attrs['beta'] = f_in.attrs['beta']
f_out.attrs['starting_counter'] = starting_counter
f_out.attrs['n_evals'] = n_evals
f_out.attrs['t_ode'] = t_ode*qr_iters
f_out.attrs['qr_iters'] = qr_iters
f_out.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

S0_TEMPLATE ='s0_{}'
LYAP_TEMPLATE ='lyap_{}'

s0 = np.array(f_in[S0_TEMPLATE.format(i)])
f_out.create_group("s0")
f_out["s0"].create_dataset(S0_TEMPLATE.format(i), data = s0)

solver = spin_solver.SpinSolver(1,N, t_ode = t_ode, s0 = s0, n_vectors = n_vectors,qr_iterations = qr_iters)
lyap, lyaps= solver.solve( rtol = 1e-7,atol = 1e-10,max_step = np.inf, print_time = True, plot = False)

f_out.create_dataset(LYAP_TEMPLATE.format(i), data = lyaps)
f_in.close()
f_out.close()

