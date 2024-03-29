import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#from cycler import cycler
#mpl.rcParams.update({'text.usetex': True})
from cycler import cycler
import os
import h5py
import spinlib

colors=['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2','#DECF3F', '#F15854', '#4D4D4D']
markers = [ 'o', '^','v', 's','x','8','D','P']
linestyles= ['-', '--', ':', '-.']

#mpl.rcParams.update({'axes.prop_cycle':(cycler('color', colors) *cycler('linestyle', ['-', '--', ':', '-.']))})
#mpl.rcParams.update({'axes.prop_cycle':(cycler('color', colors[0:4]) +cycler('linestyle', ['-', '--', ':', '-.']))})
#mpl.rcParams.update({'axes.prop_cycle':(cycler('color', colors) *cycler('markers', markers))})
#mpl.rcParams.update({'axes.prop_cycle':(cycler('color', colors[0:4]) *cycler(marker=markers))})
#mpl.rcParams.update({'axes.prop_cycle':(cycler(color= colors[0:4])+cycler(marker=markers)+cycler(linestyle=linestyles))})
#mpl.rcParams.update({'axes.prop_cycle':(cycler(color= colors[0:4])+cycler(marker=markers)+cycler(linestyle=linestyles))})

#A4 full page width
width=8.27
#Komascript A4 page with DIV=12
#width=6.181
#Figsize for APS Journals Single Column
#width=3.375
height = width/1.618
mpl.rcParams.update({'figure.figsize': [width,height]})

def half():
    mpl.rcParams.update({'figure.figsize': [width/2,height]})
def full():
    mpl.rcParams.update({'figure.figsize': [width,height]})
#Remove Whitespace
mpl.rcParams.update({'savefig.bbox':'tight'})
mpl.rcParams.update({'savefig.pad_inches':0.02})





#Default sequence of linestyles
#mpl.rcParams.update({'axes.prop_cycle':cycler('color',['r','g','b','y'])+cycler('linestyle'['-','--',':','-.' ]) })


mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.size':8})
mpl.rcParams.update({'legend.fontsize': 8})
mpl.rcParams.update({'axes.labelsize': 8})
mpl.rcParams.update({'xtick.labelsize': 8})
mpl.rcParams.update({'ytick.labelsize': 8})
#mpl.rcParams.update({'figure.autolayout': True})
#mpl.rcParams.update({'figure.autolayout': 'tight'})
#mpl.rcParams.update({'lines.linewidth': 2})
#mpl.rcParams.update({'lines.markersize'  : 2.0})

#mpl.rcParams.update({'axes.linewidth'     : 1.0})

#mpl.rcParams.update({'xtick.major.size'     : 4})    #4
#mpl.rcParams.update({'xtick.minor.size'     : 2})   #2
#mpl.rcParams.update({'xtick.major.width'    : 0.5}) #0.5
#mpl.rcParams.update({'xtick.minor.width'    : 0.5}) #0.5
#mpl.rcParams.update({'xtick.major.pad'      : 4})    #4
#mpl.rcParams.update({'xtick.minor.pad'      : 4})   #4
#mpl.rcParams.update({'ytick.major.size'     : 4})    #4
#mpl.rcParams.update({'ytick.minor.size'     : 2})   #2
#mpl.rcParams.update({'ytick.major.width'    : 0.5}) #0.5
#mpl.rcParams.update({'ytick.minor.width'    : 0.5}) #0.5
#mpl.rcParams.update({'ytick.major.pad'      : 4})    #4
#mpl.rcParams.update({'ytick.minor.pad'      : 4})   #4

def plot_means(data, N, observable, method, fit_range=None, plot = True,multiplier = 1):
    """ 
    plots (does not show) mean thermalization times on a log-log scale, which are assumed 
                        to be already calculated using generate_data
    further, returns mean thermalization times as a dictionary with format {delta: T_therm}

    data - dict {observable_method:{{N_1:{delta_1:[t_therm_1,t_therm_2,...],...},...},...} 
            in generate_data, this is therm_times_all
    method - which method out of those generated by generate_data to use for therm. times
    fit_range - if given as tuple (start,end),  a power law fit is performed on the given range of points
    plot - whether to actually plot (useful to set to False when one wants to look at N dependence)
    """
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
    plt.ylabel("Thermalization time ")

    cmap=plt.get_cmap('tab10')
    therm_times=data[observable+method][N]
    if multiplier ==1:
        prefix = str(multiplier) + r"$\overline{T_{therm}}$" + " {}, method: {}".format(observable,method)
    else:
        prefix =  r"$\overline{T_{therm}}$" + " {}, method: {}".format(observable,method)


    """Getting time at which average goes over threshold"""
    mean_therm_times =[]
    hams = []
    deltas = list(therm_times.keys())
    deltas.sort()
    for delta in deltas:
        if delta>1000:
            continue
        mean_therm_times.append(np.mean(therm_times[delta]))
        if delta > 1000:
            hams.append(2-0.001*delta)
        else:
            hams.append(0.001*delta)

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
            plt.scatter(hams,multiplier*np.array(mean_therm_times), label = prefix + " N = " + str(N) +
                    ", exponent " + str(coefficients[0])[:5]+ ", resiuduals " +str(residuals[0])[:5])
        else:
            plt.scatter(hams,mean_therm_times, label = prefix +", N = " + str(N))
    return dict(zip(hams,mean_therm_times))

def plot_smoothed(data, N, observable, method, t_odes, fit_range=None, plot = True, multiplier =1):
    """ 
    plots (does not show) thermalization times on a log-log scale,
                obtained from average time evolution, and using method
        also returns these

    data - dict {observable:{{N_1:{delta_1:averaged_time_series,...},...},...} 
                in generate_data this is time_sequence_all
    observable - which observable to use (either s_z_var or bondz_mean)
    t_odes - dict {N:{delta:t_ode,...},...} of integration times
    method - which method to use for getting thermalization time out of averaged evolution
            expected values :   relFLOAT
                                absFLOAT
    fit_range - if given as tuple (start,end),  a power law fit is performed on the given range of points
    plot - whether to actually plot (useful to set to False when one wants to look at N dependence)
    """
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\delta$ = 1 + $\epsilon$")
    plt.ylabel("Thermalization time ")
    cmap=plt.get_cmap('tab10')
    therm_sequence = data[observable][N]
    if multiplier ==1:
        prefix = r"$T_{therm}$ from $\overline{\mathcal{O}(t)}, $" + " {}, method: {}".format(observable,method)
    else:
        prefix = str(multiplier) + r"$ T_{therm}$ from $\overline{\mathcal{O}(t)}, $" + " {}, method: {}".format(observable,method)

    """Getting time at which average goes over threshold"""
    mean_therm_times =[]
    hams = []
    deltas = list(therm_sequence.keys())
    deltas.sort()
    for delta in deltas:
        if delta>1000:
            continue
        if observable == "bondz_mean":
            limit = (1 - 0.001*delta)/3
        else:
            limit = 0.3333333
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
            plt.scatter(hams,multiplier*np.array(mean_therm_times), label = prefix  +
                    ", exp:" + str(coefficients[0])[:5] + ", res:" +str(residuals[0])[:6])
        else:
            plt.scatter(hams,mean_therm_times, label = prefix + " N = " + str(N))
    return dict(zip(hams,mean_therm_times))


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
                    magnetizations[N][delta].append(np.linalg.norm(spinlib.return_magnetization(s0)))
                    angles[N][delta].append(spinlib.compute_angle(s0)/np.pi/2)
                    for observable in observables:
                        obs = np.array(f_obs[observable][OBSERVABLE_DATASET_TEMPLATE.format(observable,i)])
                        obses[observable].append(obs)
                        for method in methods:
                            if observable == "bondz_mean":
                                limit = (1 - 0.001*delta)/3
                            else:
                                limit = 0.33333
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
