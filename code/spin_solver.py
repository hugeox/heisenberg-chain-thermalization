import sys
import numpy as np
from scipy.linalg import expm

import random
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import time
from math import cos, pi, sin, log, exp, acos

import spinlib


class SpinSolver(object):
    """
    Class implementing Heisenberg Model ODE solver, Lyapunov solver and Monte Carlo updater
    """

    def __init__(self, J, N,t_ode, s0, n_vectors = 0, qr_iterations = 0, beta = 1 ):
        """
        n_vectors - number of Highest Lyapunovs to be computed
        qr_iterations - number of qr orthogonalizations used during Lyapunov calculation.  
                        Total integration time for Lyap calculation is therefore t_ode*qr_iterations
        J - Heisenberg coupling constant +ve for Ferromagnet, -ve for antiferromagnet
        t_ode - how long to integrate ODE
        s0 - our representation of a state of the model. Its format is: [x1,...,xN,y1,...,yN,z1,...,zN] if 
        n_vectors = 0. 
        If n_vectors is nonzero, it also includes the tangent vectors, as follows
        [sx1,...,sxN,...,szN,tan1x1,...,tan1zN,tan2x1,...,tan2xN,...]
        """
        vectors = spinlib.get_standard_basis(N, n_vectors)
        self.t_span = (0,t_ode)
        self.t_ode = t_ode
        self.J = J
        self.N = N
        self.n_vectors = n_vectors
        self.qr_iterations = qr_iterations
        # s0 is now a s0 + vectors
        # every time s0 is set, hamilt needs to be reset
        self.s0 = np.concatenate((s0, vectors),axis = 0)
        self.hamilt = self.hamiltonian(self.s0[:3*N]) # hamiltonian corresponding to s0. important to slice!
        self.beta = beta
        self.energies = []
        self.llyapunovs = []

    def set_s0(self,s0):
        self.s0[:3*self.N] = s0 
        self.hamilt = self.hamiltonian(s0)

    def f(self,t,svect):
        """s' = f(s) for ODE solver

        Including tangent space dynamics 
        svect = [sx1,...,sxN,...,szN,tan1x1,...,tan1zN,tan2x1,...,tan2xN,...]
        the tangent components evolve according to the jacobian - tan'= jac * tan
        """
        s1, vectors = spinlib.get_s_and_vectors(svect, self.N)
        heff = -spinlib.cycleLeft3D(s1,self.N) - spinlib.cycleRight3D(s1,self.N)
        
        # some transpose magic - basically want to matrix multiply the right things
        if len(vectors) == 0:
            retvect = vectors
        else:
            retvect = np.dot(self.jac(s1, heff), vectors).T.ravel()
        return  self.J * np.concatenate((-spinlib.cross(s1, heff), retvect), axis = 0)

    def jac(self, s, heff):
        """computes Jacobian for Heisenberg dynamics, setting J = 1
        """
        if len(s)%3!=0:
            raise Exception("Not 3D spins" + str(len(s)))
        N = int(len(s)/3)
        j = np.zeros((len(s),len(s)))
        for x in range(3):
            y = (x + 1) %3
            z = (x + 2) %3
            for i in range(N):
                iplus1 = (i + 1) % N
                iminus1 = (i - 1) % N
                j[N*x + i, N*y +i] = heff[N*z + i]
                j[N*x + i, N*z +i] = - heff[N*y + i]
                j[N*x + i, N*y + iplus1] = s[N*z + i]
                j[N*x + i, N*y + iminus1] = s[N*z + i]
                j[N*x + i, N*z + iplus1] = -s[N*y + i]
                j[N*x + i, N*z + iminus1] = - s[N*y + i]
        return j


    def mc_update_micro_fixed(self, i, xy = False):
        """Updating according to microcanonical Monte carlo, for either xy or 3D Heisenberg
        
        3D - rotating spin i around its local field by a random angle with 0.5 probability
        or flipping it (around the local field) with 0.5 probability.

        if xy - the routine assumes that an xy aligned state is being dealt with (s[2*N:3*N] =0).
        For this state, there is only one of the rotations from before that can be done that also keeps the 
        state in xy plane, and this is performed with certainty.
        """
        old_hamilt = self.hamilt
        # set new hamilt
        S_old = [self.s0[i], self.s0[i + self.N], self.s0[i + 2 * self.N]]
        iplus1 = (i+1)% self.N
        iminus1 = (i-1)% self.N
        axis = np.add([self.s0[iplus1], self.s0[iplus1 + self.N], self.s0[iplus1 + 2 * self.N]],[self.s0[iminus1], self.s0[iminus1 + self.N], self.s0[iminus1 + 2 * self.N]])
        if xy:
            theta = np.pi 
            S_rand = np.dot(spinlib.rotation_matrix(axis, theta),S_old)
            S_rand[2] = 0
            spinlib.set_spin(self.s0, i, S_rand, self.N)
        else:
            if random.random() < 0.5:
                # PERFORM RANDOM ROTATION
                theta =  2 * np.pi * random.random()
                S_rand = np.dot(spinlib.rotation_matrix(axis, theta),S_old)
                spinlib.set_spin(self.s0, i, S_rand, self.N)
                #self.hamilt = self.hamiltonian(self.s0[:3*self.N])
            else:
                # PERFORM REFLECTION
                theta = np.pi 
                S_rand = np.dot(spinlib.rotation_matrix(axis, theta),S_old)
                spinlib.set_spin(self.s0, i, S_rand, self.N)
                #self.hamilt = self.hamiltonian(self.s0[:3*self.N])
        #hamilt shouldn't change
        return True

    def mc_update_micro(self, xy = False):
        i = random.randint(0,self.N-1)
        return self.mc_update_micro_fixed(i, xy)

    def mc_micro_sweep(self):
        """Sweep of xy microcanonical flips
            
            Performed at every site with p = 0.7
        """ 
        for i in range(self.N):
            if random.random()>0.3:
                self.mc_update_micro_fixed(i,xy = True)

    def mc_twist(self):
        """ Global twist update
        
        Relevant only in xy, this is a global twist update that changes winding number by +-1
        """
        old_hamilt = self.hamilt 
        winding_number = random.choice([-1,1])
        s_new = spinlib.twist(self.s0[:3*self.N], winding_number) 
        new_hamilt = self.hamiltonian_sub(s_new, self.N, 0) 
        if np.log(random.random()) <  (-(new_hamilt-old_hamilt)*self.beta):
            # set s0 to twisted, slice assignement
            self.s0[:3*self.N] = s_new
            self.hamilt = new_hamilt
            return True
        else:
            return False

    def mc_update_fixed(self, i, xy = False):
        """Updates s0 according to the Metropolis Monte Carlo Method using local update at site i

        For the xy model, if the energy density is higher than -cos(0.1*pi), 
        the proposed update is a random rotation of spin i.
        For energy densities lower than -cos(0.1*pi), the proposed update is 
        a rotation by an angle chosen uniformly from [-2 * acos(- epsilon),2*acos(- epsilon)]

        """
        old_hamilt = self.hamilt 
        # set new hamilt
        S_old = [self.s0[i], self.s0[i + self.N], self.s0[i + 2 * self.N]]
        if xy == True:
            theta = acos(- old_hamilt/self.N) 
            if abs(2*theta)> 2 * np.pi : 
                r = 2* np.pi  * random.random()
            else:
                if abs(theta)< np.pi*0.1:
                    #theta = np.pi* 0.1
                    theta = 2 * theta
                r = 2 * theta * random.random() - theta
            #S_rand = [cos(r),sin(r),0]
            S_rand = [cos(r)*S_old[0] +sin(r)*S_old[1], -sin(r)*S_old[0] +  cos(r)*S_old[1],0]
        else:
            S_rand = spinlib.new_random_spin()
        new_hamilt = old_hamilt - self.spin_energy(i,S_old) + self.spin_energy(i,S_rand)
        if np.log(random.random()) <  (-(new_hamilt-old_hamilt)*self.beta):
            #accept change:
            self.hamilt = new_hamilt
            spinlib.set_spin(self.s0, i, S_rand, self.N)
            return True
        else:
            return False

    def spin_energy(self, i, S):
        """ returns energy of spin at site i (which is set to S = len3 array)"""
        iplus1 = (i + 1) % self.N
        iminus1 = (i - 1) % self.N
        return  - self.J * np.dot(spinlib.get_spin_at_index(self.s0,iplus1)+ \
                                spinlib.get_spin_at_index(self.s0,iminus1),
                            S)
    

    def mc_sweep(self):
        for i in range(self.N):
            self.mc_update_fixed(i,self.N)

    def mc_update(self):
        """Update random site using mc_update_fixed for 3D model
        """
        i = random.randint(0,self.N-1)
        return self.mc_update_fixed(i)
    def mc_update_xy(self):
        """Update random site using mc_update_fixed for xy model
        """
        i = random.randint(0,self.N-1)
        return self.mc_update_fixed(i,xy = True)

    def hamiltonian(self,s):
        """Energy of s
        """
        heff = -spinlib.cycleLeft3D(s,self.N) - spinlib.cycleRight3D(s,self.N)
        return self.J * 0.5*np.dot(s[:3*self.N],heff)

    def hamiltonian_sub(self, s, N_s = 10, starting_at = 0):
        """hamiltonian of a subsystem from starting_at up to starting_at + N_s-1 (lenght N_s)
            
        """
        if N_s == self.N and starting_at==0:
            return self.hamiltonian(s[:3*self.N])
        s1 = s.copy()
        N = self.N
        s1 = spinlib.cycle3D(s, N, -starting_at)
        s1[N_s:N].fill(0)
        s1[N+N_s:2*N].fill(0)
        s1[2*N+N_s:].fill(0)
        return self.hamiltonian(s1[:3*self.N])

    def single_bond_hamiltonians(self, s):
        bond_energies = np.zeros(self.N) 
        for i in range(self.N):
            bond_energies[i] = self.hamiltonian_2(s, starting_at = i)
        return bond_energies

    def solve(self,plot = False, naive_check = False, max_step = np.inf, atol = 1e-6, rtol = 1e-4, print_time = False):
        """Solves ODE and returns self.n_vectors largest Lyap exponents and 
            their evolution through qr iterations

        see: Geist, Karlheinz, Ulrich Parlitz, and Werner Lauterborn. 
            "Comparison of different methods for computing Lyapunov exponents."
            Progress of theoretical physics 83.5 (1990): 875-893, Section 3.1 
            for description of method used

        """
        start = time.time()
        t_eval = [self.t_ode]
        print('\n Hamiltonian before solve',self.hamilt)
        solution = solve_ivp(self.f, self.t_span,self.s0, t_eval = t_eval, max_step = max_step, atol=atol,rtol=rtol)
        rel_error = 0.001
        s = solution.y.T[0] 
        #reorthogonalizing every t_ode seconds, so that vectors don't get too big
        s, r = spinlib.orthogonalize(s,self.N)
        logsums = np.log(np.array([r]))
        llyaps = np.array([logsums[0]/self.t_ode])
        for i in range(1,self.qr_iterations):
            #atol = 1e-8 good enough to keep energy conserved up to 10-4 rel error
            s = solve_ivp(self.f,self.t_span,s, t_eval = t_eval, max_step =  max_step ,rtol = rtol, atol = atol).y.T[0]
            s, r = spinlib.orthogonalize(s,self.N)
            logsums = np.append(logsums, [np.log(r)+logsums[-1]], axis = 0)
            llyaps = np.append(llyaps, [logsums[i]/(self.t_ode*i)], axis = 0)

            #naive check for convergence
            if naive_check and all(abs(t-s)/abs(s) < rel_error for t,s in zip(llyaps[-1],llyaps[-2]) ):
                print("Naive check passed, breaking cycle")
                break

        if print_time:
            end = time.time()
            print("Computational time for llyaps to converge:",end-start)
        print('\n Hamiltonian  after solve ',self.hamiltonian(s))

        if plot:
            for i in range(0,self.n_vectors):
                plt.plot(llyaps.T[i], label = str(i) + '. Llyap exp. with H = ' 
                        + str(self.hamiltonian(self.s0[:3*self.N])))
            plt.plot(shortterm_lyaps,label= "short termlyaps")
            plt.plot(immediate_lyaps,label= "immediate_lyypas")
            plt.legend()
            plt.show()
        return llyaps[-1], llyaps

    def solve_ode(self, n_evals, atol = 1e-6, rtol = 1e-4, print_time = False,max_step = np.inf):
        """Solves ode
        
        with initial condition self.s0, with n_evals per unit time and integration time self.t_ode
        """
        st = time.time()
        t_eval = np.linspace(0,self.t_ode,int(n_evals * self.t_ode) + 1 ).ravel()
        solution = solve_ivp(self.f, self.t_span,self.s0[:3*self.N], t_eval = t_eval, rtol = rtol, atol = atol, max_step = max_step )
        end = time.time()
        if print_time:
            print("Solution of ODE with n_evals: " + str(n_evals) +", t_ode: " + str(self.t_ode) 
                    + ", took: " + str(end-st) + " s.")
        return solution.y.T 

    def solve_return_observables(self, n_evals, observables,  atol = 1e-6, rtol = 1e-4, print_time = False):
        """solves ODE and returns values of each observable in observables at every evaluation point
        """
        s = self.solve_ode(n_evals,atol,rtol, print_time)
        return spinlib.return_observables(s, observables)

    def find_beta(self, hs_mean):
        beta = 2.5
        for l in range(11):
            self.beta = beta 
            hs_mc = []
            for i in range(600000):
                self.mc_update()
            for i in range(600000):
                self.mc_update()
                hs_mc.append(self.hamilt)
            hs_mc_mean = np.mean(hs_mc)
            if hs_mc_mean < hs_mean:
                beta = beta - 1.25 * 2.0**(-l)
            else:
                beta = beta + 1.25* 2** (-l)
        self.beta = beta
        print(hs_mc_mean)
        print(hs_mean)
        return beta, hs_mc

if  __name__ == '__main__':
    print("Hi")
