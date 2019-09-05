""" 
checking cluster data for individual ODE integrations
assumes data present in os.environ['CHAIN_PROJECT_DIR']
+ lot of other stuff that is not that important including correlations of states

"""
import sys
sys.path.append("../cluster_data_analysis")
sys.path.append("../code")
import data_analysis_lib
import matplotlib.pyplot as plt
import numpy as np
import spinlib
import h5py
import os



data_analysis_lib.half()
delta = 1000
N =100
n_evals = 100
S0_TEMPLATE = 's0_{}'
OBSERVABLE_DATASET_TEMPLATE = '{}_{}'
PATH_TEMPLATE = os.environ['CHAIN_PROJECT_DIR'] + "/N_{}/delta_{}"
INIT_TEMPLATE = PATH_TEMPLATE + "/init_state/s0_{}_{}.hdf5"
OBS_TEMPLATE = PATH_TEMPLATE + "/observables/re_observables_{}_{}.hdf5"

observable = 's_z_var'
#observable = "bondz_mean"
cor = False
correlations={}
for k in range(1,125):
    correlations[k]=[]
angles = []
mag_ls =[]
mags = {}
s_old = np.zeros(3*N)

obses=[]

for N in [100,250,500,1000]:
    for delta in [1200]:
        for batch_no in range(34,35):
            #plt.xscale("log")
            #plt.yscale("log")
            starting_counter = batch_no * n_evals
            if not os.path.isfile(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals)):
                continue
            f_in = h5py.File(INIT_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
            if not os.path.isfile(OBS_TEMPLATE.format(N,delta,starting_counter,n_evals)):
                continue
            f_obs = h5py.File(OBS_TEMPLATE.format(N,delta,starting_counter,n_evals), 'r')
            print(f_obs.attrs['t_ode'])
            #plt.plot(np.linspace(0,f_obs.attrs['t_ode'],801), np.array(f_obs[observable][OBSERVABLE_DATASET_TEMPLATE.format(observable,starting_counter)]))
            plt.plot(np.array(f_obs[observable][OBSERVABLE_DATASET_TEMPLATE.format(observable,starting_counter)]), label="N ={}, $\delta$ = {} against time, observable {}".format(N,0.001*delta,observable))
            plt.xlabel("Integration time")
            plt.ylabel(observable)
            f_in.close()
            f_obs.close()
            continue
            for i in range(starting_counter, starting_counter + n_evals):
                obses.append(np.array(f_obs[observable][OBSERVABLE_DATASET_TEMPLATE.format(observable,i)]))
                s0 = np.array(f_in[S0_TEMPLATE.format(i)])

                print("OLD . nEW",np.dot(s0,s_old))
                s_old = s0
                if cor:
                    for k in correlations.keys():
                        mean_c = np.mean(np.multiply(s0[:N],np.roll(s0[:N],k))+ \
                                np.multiply(s0[N:2*N],np.roll(s0[N:2*N],k)))
                        correlations[k].append(mean_c)
                angles.append(spinlib.compute_angle(s0)/np.pi/2)
                if int(round(spinlib.compute_angle(s0)/np.pi/2)) not in mags.keys():
                    mags[int(round(spinlib.compute_angle(s0)/np.pi/2))] = [np.linalg.norm(spinlib.return_magnetization(s0))]
                else:
                    mags[int(round(spinlib.compute_angle(s0)/np.pi/2))].append(np.linalg.norm(spinlib.return_magnetization(s0)))
                mag_ls.append(np.linalg.norm(spinlib.return_magnetization(s0)))
                if i%100==0:
                    print(np.linalg.norm(spinlib.return_magnetization(s0)))
            f_in.close()
            f_obs.close()
plt.grid()
plt.legend()
plt.show()

exit()
print(np.array(obses).shape)
a = np.array(obses[0])
for k in obses[1:]:
    a=np.add(a,k)
#plt.plot(np.linspace(0,14000,401), np.sum(obses,axis=0))
plt.plot(np.linspace(0,14000,401), np.abs(a-a[-1]))
#plt.plot(np.linspace(0,14000,401), 10000*obses[0])
plt.suptitle("N ={}, $\delta$ = {}, s_z_var against time".format(N,delta))
plt.xlabel("Integration time")
plt.ylabel("s_z_var")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()

plt.plot(mag_ls)
plt.show()
plt.hist(angles, bins = 200)
plt.show()
if cor:
    ks = []
    cs = []
    for m in correlations.keys():
        ks.append(m)
        cs.append(np.mean(correlations[m]))
    plt.scatter(ks,cs,label = "correlation against distance")
    plt.show()

for i in mags.keys():
    print(i, len(mags[i]))
    plt.hist(mags[i], range=(0,1),bins = 50, label = "Winding number" + str(i), histtype = 'step')
    plt.legend()
plt.hist(mag_ls,range=(0,1), bins = 50, label = "Cummulative hist", histtype = 'step')
plt.legend()
plt.show()

