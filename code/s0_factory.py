"""
Generates xy aligned initial states of given energy density using elaborate Monte Carlo techniques.
Command-line arguments:

N - length of spin chain
delta = 1000 * (1+energy density)
beta - beta corresponding to energy density
starting_counter - specifies at what index s0_factory starts generating initial states
n_evals - specifies how many initial states are generated
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import random
from math import sin,cos, acos

import spinlib
import spin_solver
import argparse
import time

import h5py



def get_s0_twist(N, winding_number):
    """Returns a simple twist state (same angles between neighbors) with given winding number  
    """
    angle = 2*np.pi * winding_number / N
    s0x= [1]
    s0y = [0]
    s0z = [0]
    for i in range(1,N):
        s_old  = [s0x [-1],s0y [-1],s0z [-1] ]
        final_angle = angle
        sin_final = sin(final_angle)
        cos_final = cos(final_angle)
        s_new = [cos_final*s_old[0] +sin_final*s_old[1], -sin_final*s_old[0] +  cos_final*s_old[1],0]
        s0x.append(s_new[0])
        s0y.append(s_new[1])
        s0z.append(s_new[2])
        norm = np.linalg.norm([s0x [-1],s0y [-1],s0z [-1] ])
        s0x [-1] = s0x[-1] / norm
        s0y [-1] = s0y[-1] / norm
        s0z [-1] = s0z[-1] / norm

    return np.concatenate((s0x,s0y,s0z),axis = 0)
    
def get_s0_xy_equipartition(N, angle):
    """ state with neighboring angles of order angle
    
    there are different choices for neighboring angles commented out
    and this also ignores periodic bcs,
    """
    s0x= [1]
    s0y = [0]
    s0z = [0]
    for i in range(1,N):
        s_old  = [s0x [-1],s0y [-1],s0z [-1] ]
        #final_angle = random.choice([-1,1]) * np.radians(angle - 3 + 6*random.random())
        #final_angle = random.choice([-1,1]) * np.radians(angle * 2*random.random())
        final_angle = random.choice([-1,1]) * np.radians(random.gauss(0, 1.2 * angle))
        sin_final = sin(final_angle)
        cos_final = cos(final_angle)
        s_new = [cos_final*s_old[0] +sin_final*s_old[1], -sin_final*s_old[0] +  cos_final*s_old[1],0]
        s0x.append(s_new[0])
        s0y.append(s_new[1])
        s0z.append(s_new[2])
        norm = np.linalg.norm([s0x [-1],s0y [-1],s0z [-1] ])
        s0x [-1] = s0x[-1] / norm
        s0y [-1] = s0y[-1] / norm
        s0z [-1] = s0z[-1] / norm

    return np.concatenate((s0x,s0y,s0z),axis = 0)

def get_s0_random_xy(N):
    """random state in xy plane
    """
    s0x= []
    s0y = []
    s0z = []
    for i in range(N):
        s0z.append(0)
        r = 2 * np.pi * random.random()
        s0x.append(cos(r))
        s0y.append(sin(r))
        norm = np.linalg.norm([s0x [-1],s0y [-1],s0z [-1] ])
        s0x [-1] = s0x[-1] / norm
        s0y [-1] = s0y[-1] / norm
        s0z [-1] = s0z[-1] / norm
    return np.concatenate((s0x,s0y,s0z),axis = 0)

def get_s0_xy_positive(N, delta, beta,starting_counter, n_evals):
    """returns n_evals vectors of energy density 0.001*(delta) - 1, for negative beta

    This is a rather ad hoc function used to generate negative beta states from positive beta states.
    The way it works is it simply flips every other spin of a previously generated state
    """
    energy_per_bond = -1 + delta * 0.001 # delta in miliergs
    S0_TEMPLATE = 's0_{}'# format is S0_N_energy_per_bond_i
    #infinite temperature
    if delta ==1000:
        f = h5py.File('s0_{}_{}.hdf5'.format(starting_counter,n_evals), 'w')
        for i in range(starting_counter,starting_counter + n_evals):
            s0 = get_s0_random_xy(N)
            dset = f.create_dataset(S0_TEMPLATE.format(i), data = s0)
        f.attrs['N'] = N
        f.attrs['delta'] = delta
        f.attrs['beta'] = beta
        f.attrs['starting_counter'] = starting_counter
        f.attrs['n_evals'] = n_evals
        f.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        f.attrs['monte_carlo_method'] = "True random xy for delta =1000"
        f.close()
    else:
        f_in = h5py.File('s0_temp.hdf5', 'r')
        f = h5py.File('s0_{}_{}.hdf5'.format(starting_counter,n_evals), 'w')
        for i in range(starting_counter, starting_counter + n_evals):
            s0 = np.array(f_in[S0_TEMPLATE.format(i)])
            s0[::2]=-s0[::2]
            dset = f.create_dataset(S0_TEMPLATE.format(i), data = s0)
        f.attrs['N'] = N
        f.attrs['delta'] = delta
        f.attrs['beta'] = beta
        f.attrs['starting_counter'] = starting_counter
        f.attrs['n_evals'] = n_evals
        f.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        f.attrs['monte_carlo_method'] = "From -Beta, by flipping odd spins: 500 iters (every50th twist), then 100 iters(every10th twist) between samples, for delta<50 start with gaussian equipartition"
        f_in.close()
        f.close()

def get_s0_xy_mc(N, delta, beta,starting_counter, n_evals):
    """returns n_evals vectors of energy density 0.001*(delta) - 1

    Uses xy monte carlo method, this should be described in f.attrs['monte_carlo_method']
    delta is given in miliergs - to get actual energy density, use -1 + delta * 0.001
    """
    tolerance = 0.1 #energy tolerance
    energy_per_bond = -1 + delta * 0.001 # delta in miliergs

    S0_TEMPLATE = 's0_{}'# format is S0_N_energy_per_bond_i
    if delta < 50:# was 50
        angle = np.degrees(acos(0.99-delta*0.001))
        s0 = get_s0_xy_equipartition(N,angle)
    else:
        s0 = get_s0_random_xy(N)
    solver = spin_solver.SpinSolver(1, N, t_ode = 1,s0 = s0, beta = beta)

    #initial converging towards right energy using xy updates
    while abs(solver.hamilt - energy_per_bond * N) > tolerance:
        solver.mc_update_xy()

    #initial shuffling around, not generating any states yet
    iters = 0
    while iters < 500:
        iters +=1
        solver.mc_micro_sweep()
        for m in range(N):
            solver.mc_update_xy()
        if iters%50==0:
            solver.mc_twist()
    f = h5py.File('s0_{}_{}.hdf5'.format(starting_counter,n_evals), 'w')

    #generating initial states
    for i in range(n_evals):
        iters = 0
        while iters < 100:
            iters +=1
            if iters%10==0:
                solver.mc_twist()
            solver.mc_micro_sweep()
            for m in range(N):
                solver.mc_update_xy()
        # reaching desired energy
        while abs(solver.hamilt - energy_per_bond * N) > tolerance:
            solver.mc_update_xy()
        dset = f.create_dataset(S0_TEMPLATE.format(starting_counter+i), data = solver.s0)
    f.attrs['N'] = N
    f.attrs['delta'] = delta
    f.attrs['beta'] = beta
    f.attrs['starting_counter'] = starting_counter
    f.attrs['n_evals'] = n_evals
    f.attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    f.attrs['monte_carlo_method'] = "500 iters (every50th twist), then 100 iters(every10th twist) between samples, for delta<50 start with gaussian equipartition"
    f.close()

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='N, delta in miliergs, beta, starting_counter, n_evals')
    parser.add_argument('N', metavar='N', type=int, help='linear system size')
    parser.add_argument('delta', metavar='delta', type = int,
                         help='energy density in miliergs measured from GS')
    parser.add_argument('beta', metavar='beta',type = float, help = 'beta')
    parser.add_argument('starting_counter', metavar='STARTING_COUNTER',type = int, help = 'start at')
    parser.add_argument('n_evals', metavar='n_evals',type = int, help = 'number to evaluate')
    st = time.time()
    args = parser.parse_args()
    delta = args.delta
    n_evals = args.n_evals
    beta = args.beta
    starting_counter = args.starting_counter
    N = args.N
    print("N = ", N,", starting_counter = ", starting_counter, ", delta = ", delta, ", n_evals = ", n_evals,
    ", beta = ", beta)

    # for delta>950, one gets s0 by flipping every other spin of a previous run for -beta
    if delta>950:
        get_s0_xy_positive(N = N, delta = delta, beta = beta, starting_counter = starting_counter, n_evals =n_evals)
    else:
        get_s0_xy_mc(N = N, delta = delta, beta = beta, starting_counter = starting_counter, n_evals =n_evals)
    end = time.time()
    print("Time taken for generation of initial states: ", end - st, " s")
