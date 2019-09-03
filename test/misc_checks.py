import sys
import os
sys.path.append("..")
sys.path.append("../code")
sys.path.append("../cluster_data_analysis")
import data_analysis_lib


import numpy as np
from scipy.ndimage.filters import convolve1d
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from math import cos, pi, sin, log
import h5py

import spinlib
import spin_solver
import scipy
import pickle

"""
Misc checks 
"""

def test_1():
    """To test magnetization dependence of Lyapunovs"""
    N=500
    delta = 1990
    S0_TEMPLATE = 's0_{}'
    PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
    LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
    INIT_TEMPLATE = PATH_TEMPLATE + "/init_state/s0_{}_{}.hdf5"
    OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  
    evals = 20
    n_evals = 100
    mags = []
    lyaps = []

    for batch_no in range(evals):
        starting_counter = batch_no * n_evals
        f_obs = h5py.File(OBS_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
        for i in range(starting_counter, starting_counter + n_evals):
            s0 = np.array(f_obs["s0"][S0_TEMPLATE.format(i)])
            lyap_filename =LYAP_TEMPLATE.format(N,delta,i)
            if os.path.isfile(lyap_filename):
                f_lyap = h5py.File(lyap_filename, 'r')
                lyap_time_series = np.array(f_lyap["lyap_"+str(i)])
                plt.plot(lyap_time_series)
                plt.show()
                mags.append(np.linalg.norm(spinlib.return_magnetization(s0)))
                lyaps.append(lyap_time_series[-1])
                f_lyap.close()
        f_obs.close()
    print(len(lyaps))
    plt.hist(np.log(lyaps), bins = 10)
    plt.show()
    exit()
    plt.scatter(mags,lyaps)
    plt.xlabel("Magnetization")
    plt.ylabel("Lyapunov exponent")
    plt.grid()
    plt.legend()
    plt.show()
def test_2():
    deltas = [10,20,40,100,150,200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
    N = 500
    mean_speeds =[]
    evals = 100
    n_evals = 100
    S0_TEMPLATE = 's0_{}'
    PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
    LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
    INIT_TEMPLATE = PATH_TEMPLATE + "/init_state/s0_{}_{}.hdf5"
    OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  

    for delta in deltas:
        continue
        speeds = []
        for batch_no in range(evals):

            starting_counter = batch_no * n_evals
            if not os.path.isfile(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals)):
                continue
            f_s0 = h5py.File(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
            for i in range(starting_counter, starting_counter + n_evals):
                s0 = np.array(f_s0[S0_TEMPLATE.format(i)])
                speeds.append(np.var(spinlib.get_speed(s0)))
            f_s0.close()
        mean_speeds.append(np.mean(speeds))

    mean_speeds = np.load("mean_speeds.npy")
    plt.xscale("log")
    plt.yscale("log")
    coefficients,residuals, a ,b ,c= np.polyfit(np.log(deltas)[:3],np.log(mean_speeds)[:3], 1, full = True)
    print(N, "Fit times power law fit residuals are ", residuals[0])
    polynomial = np.poly1d(coefficients)
    ys = polynomial(np.log(deltas))
    plt.plot(deltas, np.exp(ys))#, label = "N=" + str(N)+ ",exponent = " + str(coefficients[0])[:7])
    plt.scatter(deltas,mean_speeds, label = r"Fit times: $T_{therm}$ N = " + str(N) + ", exponent " + str(coefficients[0])[:7])
    plt.legend()

    plt.show()




"""4th power test"""
def test_3():
    deltas = [10,20,40,100,150,200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
    N = 500
    mean_speeds =[]
    evals = 100
    n_evals = 100
    S0_TEMPLATE = 's0_{}'
    PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
    LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
    INIT_TEMPLATE = PATH_TEMPLATE + "/init_state/s0_{}_{}.hdf5"
    OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  

    for delta in deltas[:4]:
        speeds = []
        continue
        for batch_no in range(evals):

            starting_counter = batch_no * n_evals
            if not os.path.isfile(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals)):
                continue
            f_s0 = h5py.File(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
            for i in range(starting_counter, starting_counter + n_evals):
                s0 = np.array(f_s0[S0_TEMPLATE.format(i)])
                bondz = np.multiply(np.roll(spinlib.get_speed(s0),1),spinlib.get_speed(s0))
                speeds.append(np.var(bondz))
            f_s0.close()
        mean_speeds.append(np.mean(speeds))


    #np.save("mean_speeds.npy", mean_speeds)
    mean_speeds = np.load("mean_speeds.npy")
    plt.xscale("log")
    plt.yscale("log")
    coefficients,residuals, a ,b ,c= np.polyfit(np.log(deltas[:3]),np.log(mean_speeds[:3]), 1, full = True)
    print(N, "Fit times power law fit residuals are ", residuals[0])
    polynomial = np.poly1d(coefficients)
    ys = polynomial(np.log(deltas))
    plt.plot(deltas, np.exp(ys))#, label = "N=" + str(N)+ ",exponent = " + str(coefficients[0])[:7])
    plt.scatter(deltas[:3],mean_speeds[:3], label = r" Should go as square Fit times: $T_{therm}$ N = " + str(N) + ", exponent " + str(coefficients[0])[:7])
    plt.legend()

    plt.show()

def test_4():
    evals =100
    n_evals = 100

    deltas = [10,20,40,100,150,200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
            950,1000]
    for filename in [ "var_triggers_all.pickle"]:
        with open("../cluster_data_analysis/cache/"+filename, 'rb') as fp:
            t_odes,therm_times_all,lyaps, magnetizations,therm_sequence_all,angles= pickle.load(fp)
        observable = "bondz_mean"
        #observable = "s_z_var"
        therm_times=therm_times_all[observable]
        therm_sequence=therm_sequence_all[observable]
        i = 0
        for observable in [ "s_z_var","bondz_mean"]:
            for N in [100,1000]:
                therm_times=therm_times_all[observable]
                therm_sequence=therm_sequence_all[observable]
                #plt.xscale("log")
                #plt.yscale("log")
                for delta in [1100]:
                    #plt.hist(therm_times[N][delta],bins = 30,label = "$\delta$: " + str(0.001*delta)+ ", N:"+str(N)+ ", t_ode:" + str(t_odes[N][delta]), histtype="step" )
                    plt.plot(np.linspace(0,t_odes[N][delta],801),np.array(therm_sequence[N][delta]), label = observable+ " for N = " + str(N))
                    #plt.suptitle("Histogram of thermalization times at $\delta =$" + str(0.001*delta))
    #plt.plot([0,800],[0.3,0.3])
    #plt.plot([0,800],[0.296,0.296])
    plt.grid()
    plt.legend()
    #plt.suptitle("Average var_z plot")
    plt.xlabel("$T_{ODE}$")
    plt.ylabel(r"$\overline{\mathcal{O}(t)}$")
    #plt.xlabel("$\delta$")
    plt.show()
test_4()
