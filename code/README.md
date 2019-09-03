For more detailed documentation of each module, use pydoc $MODULE_NAME

# spin_solver 

Class implementing Heisenberg Model ODE solver, Lyapunov solver and Monte Carlo updater

# observables

Contains functions that return various observables. 
These functions are then passed to 
SpinSolver.solve_return_observables in calculations

# spinlib

Contains helper methods for dealing with spins in our representation [x1,...,xN,y1,...,zN]

# s0_factory

Generates xy aligned initial states of given energy density using elaborate Monte Carlo techniques.
Command-line arguments:

N - length of spin chain
delta = 1000 * (1+energy density)
beta - beta corresponding to energy density
starting_counter - specifies at what index s0_factory starts generating initial states
n_evals - specifies how many initial states are generated

Output file:

s0_{starting_counter}_{n_evals}.hdf5

# calculator_observables

Uses initial states stored in s0_{starting_counter}_{n_evals}.hdf5 file and computes observables
that are given in the list observables.
These are then stored in f_out, named re_observables_{starting_counter}_{n_evals}.hdf5

Command-line arguments:

N - length of spin chain
starting_counter - specifies at what index s0_factory starts generating initial states
n_evals - specifies how many initial states are generated
t_ode - integration time for ODE integration
n_evals_ode - number of evaluations per second for ODE integration

Input file:

s0_{starting_counter}_{n_evals}.hdf5

Output file:

re_observables_{starting_counter}_{n_evals}.hdf5

# lyapunov_calculator

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
