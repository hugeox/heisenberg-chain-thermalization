"""
Calculates observables in perpendicular directions to magnetization by 3D Canonical Monte Carlo,
by first finding beta with given energy density and then sampling at that density
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
from observables import  s_z_var, bond_mean, bond_var, bond2_var, bond2_mean,bondz_mean, bondz_var, bondz2_var, bondz2_mean, bondx_var,bondy_var, s_perp_var,bond_perp_mean,bond_perp_var, s_perp,bond_perp
import h5py


#N=100,delta=250
#N=1000 - others
N = 100
starting_counter = 400
n_evals = 100
print("N = ", N,", starting_counter = ", starting_counter,  ", n_evals = ", n_evals)


for i in range(3): #compute iteratively
    for delta in [250]:
        print("delta: ", delta)
        s_perp_vars = []
        bond_perp_means = []
        mags = []
        s0 = s0_factory.get_s0_random_xy(N)
        solver = spin_solver.SpinSolver(1, N,10 ,s0, beta =1)
        solver.find_beta(0.001*N*(delta-1000))

        for l in range(100):
            solver.mc_update()
            solver.mc_update()
        print(solver.hamilt/N +1)
        for l in range(5000*1000*50):
            solver.mc_update()
            solver.mc_update()
            if l%10==0 and np.abs(solver.hamilt+0.001*N*(1000-delta))<0.5:
                mags.append(spinlib.return_magnetization_norm(solver.s0))
                s_perp_vars.append(s_perp_var(solver.s0))
                bond_perp_means.append(bond_perp_mean(solver.s0))
        f_out = h5py.File('../data/mc_observables_{}_{}.hdf5'.format(delta,i), 'w')
        f_out.create_dataset("mags",data = mags)
        f_out.create_dataset("s_perp_vars" ,data = s_perp_vars)
        f_out.create_dataset("bond_perp_means", data = bond_perp_means)
        f_out.attrs['beta'] = solver.beta
        f_out.close()

