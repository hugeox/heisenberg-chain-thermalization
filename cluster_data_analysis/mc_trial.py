import sys
import os
sys.path.append("..")
sys.path.append("../code")
import data_analysis_lib
import numpy as np
from scipy.ndimage.filters import convolve1d 
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

def return_closest(n,value_set):
  return min(value_set, key=lambda x:abs(x-n))

def get_thermalization_time(t_ode,time_series,fixed_trigger = None):
    """ returns thermalization time of time_series

    fixed_trigger - time series considered thermalized when it reaches fixed_trigger 
                    if None: one uses average value through the second half of integration as trigger
    """

    if fixed_trigger:
        long_time_average = fixed_trigger
        if long_time_average<0:
            karel =np.where(time_series < long_time_average )[0]
        else:
            karel =np.where(time_series > long_time_average )[0]
        if len(karel)==0:
            index = len(time_series)-1
        else:
            index = karel[0]
    else:
        long_time_average = np.mean(time_series[int(len(time_series)/2):])
        index = np.where(time_series > long_time_average )[0][0]
    t_therm = t_ode * index / len(time_series)
    value_therm = time_series[index]
    return t_therm, value_therm

def generate_data(Ns, deltas, n_evals,evals,OBS_TEMPLATE, LYAP_TEMPLATE = "{}_{}_{}"):
    """
    Generates thermalization times for the observables bondz_mean and s_z_var using various methods

               !!! All paths should be absolute!!! 
    OBS_TEMPLATE - a template for observable file. 
            Is then formatted using .format(N,delta,starting_counter,n_evals)
    LYAP_TEMPLATE - a template for lyapunov file. 
            Is then formatted using .format(N,delta,i)
    evals - number of batches
    n_evals - length of a batch
    Ns - Ns to analyze
    deltas - deltas to analyze

    Methods:
        absFLOAT - time series considered thermalized when it reaches equilibrium_average-FLOAT
        relFLOAT - time series considered thermalized when it reaches equilibrium_average*FLOAT
        first - time series considered thermalized when it reaches equilibrium_average (special case of above)
        (empty_string) - time series considered thermalized when it 
                            first reaches its average value through the second half of integration
            NB the equilibrium average here is assumed to be the isotropic zero magnetization value
    """
    S0_TEMPLATE = 's0_{}'
    OBSERVABLE_DATASET_TEMPLATE = '{}_{}'
    observables =['s_z_var', 'bondz_mean']
    # methods for thermalization time calculation for individual runs
    methods = ['abs0.01','abs0.02','abs0.03','abs0.04','rel0.8','rel0.9','rel0.7','first','']

    data = {}
    t_odes ={}
    therm_times = {}
    therm_times_all = {}
    therm_sequence_all={}
    eq_obs_values = {}
    for obs in observables:
        for method in methods:
            therm_times_all[obs+method] = {}
        therm_sequence_all[obs]={}
        eq_obs_values[obs] = {}
    lyaps = {}
    angles={}
    magnetizations = {}
    for N in Ns:
        for obs in observables:
            for method in methods:
                therm_times_all[obs+method][N]={}
            therm_sequence_all[obs][N]={}
        t_odes[N] ={}
        lyaps[N] ={}
        angles[N]={}
        magnetizations[N] ={}
        for delta in deltas :
            for obs in observables:
                for method in methods:
                    therm_times_all[obs+method][N][delta]=[]
            lyaps[N][delta] = []
            magnetizations[N][delta] =[]
            angles[N][delta] =[]
    mag_vals = {0.0: 0.33332738590880195, 0.025: 0.33307956586891685, 0.05: 0.3326564798699867, 0.07500000000000001: 0.3320883979754039, 0.1: 0.3313545522832395, 0.125: 0.3302569822117752, 0.15000000000000002: 0.3290837899991117, 0.17500000000000002: 0.327625550776773, 0.2: 0.32559085983104563, 0.225: 0.32329927512893614 }
    for N in Ns:
        for delta in deltas:
            print("Processing observables for N: ", N, ", delta: ",delta)
            obses = {}
            for obs in observables:
                obses[obs]=[]

            for batch_no in range(evals):

                starting_counter = batch_no * n_evals
                if not os.path.isfile(OBS_TEMPLATE.format(N,delta,starting_counter,n_evals)):
                    continue
                f_obs = h5py.File(OBS_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
                t_ode = f_obs.attrs["t_ode"]
                t_odes[N][delta] = t_ode
                for i in range(starting_counter, starting_counter + n_evals):
                    s0 = np.array(f_obs["s0"][S0_TEMPLATE.format(i)])
                    #mag = np.linalg.norm(spinlib.return_magnetization(s0))
                    #if mag> 0.06:
                    #    continue
                    magnetizations[N][delta].append(np.linalg.norm(spinlib.return_magnetization(s0)))
                    angles[N][delta].append(spinlib.compute_angle(s0)/np.pi/2)
                    for observable in observables:
                        obs = np.array(f_obs[observable][OBSERVABLE_DATASET_TEMPLATE.format(observable,i)])
                        obses[observable].append(obs)
                        for method in methods:
                            if observable == "bondz_mean":
                                limit = (1 - 0.001*delta)/3
                            else:
                                mag = np.linalg.norm(spinlib.return_magnetization(s0))
                                limit = 0.33333
                                limit = mag_vals[return_closest(mag,mag_vals.keys())]
                            if method[:3]=="rel":
                                threshold = float(method[3:]) * limit
                                t, value = get_thermalization_time(t_ode,obs,threshold)
                            elif method[:3] =="abs":
                                threshold = limit - float(method[3:])
                                t, value = get_thermalization_time(t_ode,obs,threshold)
                            elif method =="first":
                                #first reaching of limiting value
                                threshold = limit
                                t, value = get_thermalization_time(t_ode,obs,threshold)
                            else:
                                #long term average method
                                t, value = get_thermalization_time(t_ode,obs,None)

                            therm_times_all[observable+method][N][delta].append(t)

                    # dealing with llyapunovs here
                    lyap_filename =LYAP_TEMPLATE.format(N,delta,i)
                    if (N==100 or N==500) and os.path.isfile(lyap_filename):
                        f_lyap = h5py.File(lyap_filename, 'r')
                        lyap_time_series = np.array(f_lyap["lyap_"+str(i)])
                        lyaps[N][delta].append(lyap_time_series[-1])
                        f_lyap.close()
                f_obs.close()
            for observable in observables:
                therm_sequence_all[observable][N][delta]=np.mean(obses[observable],axis=0)

    return t_odes,therm_times_all,lyaps,magnetizations,therm_sequence_all,angles



def plot_smoothed(data, N, observable, method, t_odes, fit_range=None, plot = True, mags = []):
    mag_vals = {0.0: 0.33332738590880195, 0.025: 0.33307956586891685, 0.05: 0.3326564798699867, 0.07500000000000001: 0.3320883979754039, 0.1: 0.3313545522832395, 0.125: 0.3302569822117752, 0.15000000000000002: 0.3290837899991117, 0.17500000000000002: 0.327625550776773, 0.2: 0.32559085983104563, 0.225: 0.32329927512893614 }
    multiplier = 1

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
    plt.ylabel("Thermalization time ")
    cmap=plt.get_cmap('tab10')
    therm_sequence = data[observable][N]
    prefix = r"$T_{therm}$ from $\overline{\mathcal{O}(t)}, $" + " {}, method: {}".format(observable,method)

    """Getting time at which average goes over threshold"""
    mean_therm_times =[]
    hams = []
    errs = []
    for delta in therm_sequence.keys():
        temp_sum = 0
        for mag in mags[N][delta]:
            temp_sum += mag_vals[return_closest(mag,mag_vals.keys())]
        true_average = temp_sum/len(mags[N][delta])
        if delta>1000:
            continue
        if observable == "bondz_mean":
            limit = (1 - 0.001*delta)/3
        else:
            limit = 0.3333333
            limit = true_average
        if method[:3]=="rel":
            threshold = limit * float(method[3:])
        else:
            threshold = limit - float(method[3:])
        if threshold >0:
            if therm_sequence[delta][-1] > threshold:
                index = np.where(therm_sequence[delta] > threshold) [0][0]
                t_therm = t_odes[N][delta] * (index) / len(therm_sequence[delta])
                mean_therm_times.append(t_therm)
            else:
                mean_therm_times.append(np.inf)
        else:
            mean_therm_times.append(np.inf)
        if delta > 1000:
            hams.append(2-0.001*delta)
        else:
            hams.append(0.001*delta)
        errs.append(t_odes[N][delta]  / len(therm_sequence[delta]))
    hams_log = np.log(np.array(hams))
    mean_therm_times_log = np.log(mean_therm_times)
    if plot:
        if fit_range:

            start,end = fit_range
            coefficients,residuals, a ,b ,c= np.polyfit(hams_log[start:end],
                                                        mean_therm_times_log[start:end], 1, full = True)
            print(N, "Thermalization time residuals are ", residuals[0])
            polynomial = np.poly1d(coefficients)
            ys = polynomial(hams_log)
            plt.plot(hams[start:end], multiplier*np.exp(ys[start:end]),color = cmap.colors[1])
            #, label = "N=" + str(N)+ ",exponent = " + str(coefficients[0])[:7])
            plt.scatter(hams,multiplier*np.array(mean_therm_times),
                        label = prefix + " N = " + str(N) +
                    ", exponent " + str(coefficients[0])[:7] + ", resiuduals " +str(residuals[0])[:7])
        else:
            plt.scatter(hams,mean_therm_times, label = prefix + " N = " + str(N))
    return dict(zip(hams,mean_therm_times))


pickle_name = "../data/mc_experiment.pickle"
#pickle_name = "../data/low_m_experiment.pickle"

PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"

LYAP_TEMPLATE = PATH_TEMPLATE + "/lyaps/lyap_{}.hdf5"
OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"  

use_cache = False

deltas = [20,40,100,150,200,250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
Ns =[1000]
number_of_batches =200 #number of batches
n_evals = 100
if os.path.isfile(pickle_name) and use_cache:
    print("Restoring cache named: ", pickle_name)
    with open(pickle_name, 'rb') as fp:
        t_odes,therm_times_all,lyaps, magnetizations,therm_sequence_all,angles= pickle.load(fp)
else:
    t_odes,therm_times_all,lyaps,magnetizations,therm_sequence_all,angles =\
                        generate_data(Ns, deltas,
                                        n_evals, number_of_batches, OBS_TEMPLATE, LYAP_TEMPLATE)
    with open(pickle_name, 'wb') as fp:
        print("Caching into:", pickle_name)
        pickle.dump((t_odes,therm_times_all,lyaps,magnetizations,therm_sequence_all,angles),fp)

i = 0
Ns =[1000]
N=Ns[0]
deltas = np.array(list(t_odes[Ns[0]].keys()))
print([(len(therm_sequence_all["s_z_var"][1000][delta]),delta) for delta in deltas])
print(t_odes[N])

print([len(magnetizations[1000][x]) for x in magnetizations[1000].keys()])

cmap=plt.get_cmap('tab10')
data_analysis_lib.full()
if False:
    observable = "s_z_var"

    #plot_smoothed(therm_sequence_all,N,observable,"abs0.02",t_odes,fit_range = (2,15),mags = magnetizations)
    #plot_smoothed(therm_sequence_all,N,observable,"abs0.04",t_odes,fit_range = (2,15),mags = magnetizations)
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.01",t_odes,fit_range = (3,15))
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.04",t_odes,fit_range = (1,15), multiplier =1)
    #data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.02", t_odes,fit_range = (1,15), multiplier =100)
    data_analysis_lib.plot_means(therm_times_all,N,observable,"first",fit_range = (3,14), multiplier = 1)
else:
    observable = "bondz_mean"
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.04",t_odes,fit_range = (2,10), multiplier =10)
    data_analysis_lib.plot_smoothed(therm_sequence_all,N,observable,"abs0.02", t_odes,fit_range = (2,9), multiplier =10)


plt.grid()
plt.legend()
plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
plt.ylabel("Thermalization time ")

plt.show()

mags =[]
hams_lyap =[]
for delta in magnetizations[1000].keys():
    mags.append(np.mean(magnetizations[1000][delta]))
    if delta < 1000:
        hams_lyap.append(0.001*delta)
    else:
        hams_lyap.append(2-0.001*delta)

plt.scatter(hams_lyap,mags,label = "Magnetizations")
plt.ylabel(r"$\overline{|\vec{m}|}$")
plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
plt.legend()
plt.show()

exit()
""" 
Checking difference betweeen triggering with 1/3 and triggering with Monte Carlo average
"""
pickle_name_2 = "cache/var_triggers_all_new.pickle" #this pickle's therm_times_all have been triggered with 1/3
with open(pickle_name_2, 'rb') as fp:
    t_odes,therm_times_all_2,lyaps, magnetizations,therm_sequence_all,angles= pickle.load(fp)
for delta in [200]:
    magds = []
    ts =[]
    mags = magnetizations[1000][delta]
    means = therm_times_all["s_z_varfirst"][1000][delta]
    for i in range(20):
        step = 0.05
        means_low_mag = [x[0] for x in zip(means,mags) if (x[1]<step*i+step and x[1] >step*i)]
        if len(means_low_mag)>30:
            ts.append(np.mean(means_low_mag))
            magds.append(i*step)

    plt.scatter(magds,ts,
            label = r"$\overline{T_{therm}}(|\vec{m}|)$, using $\mathcal{O}_{MC}(|\vec{m}|)$, $\delta = $" + str(0.001*delta)[:4])
for delta in [200]:
    magds = []
    ts =[]
    mags = magnetizations[1000][delta]
    means = therm_times_all_2["s_z_varfirst"][1000][delta]
    for i in range(20):
        step = 0.05
        means_low_mag = [x[0] for x in zip(means,mags) if (x[1]<step*i+step and x[1] >step*i)]
        if len(means_low_mag)>30:
            ts.append(np.mean(means_low_mag))
            magds.append(i*step)

    plt.scatter(magds,ts,label = r"$\overline{T_{therm}}(|\vec{m}|)$, using $\frac{1}{3}$ $\delta = $" + str(0.001*delta)[:4])
plt.grid()
plt.legend()
plt.ylabel(r"$T_{therm}$ ")
plt.xlabel(r"$\overline{|\vec{m}|}$")
plt.show()

