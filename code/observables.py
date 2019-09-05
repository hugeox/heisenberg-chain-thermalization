"""
contains functions that return various observables 
These functions are then passed to 
SpinSolver.solve_return_observables in calculations
"""
import numpy as np
import random
import math
import spinlib

# norm squared of magnetic moment
def mzsquared(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
        return np.dot(s[2*N:],s[2*N:])

# norm squared of magnetic moment
def mzsquared_3_spins(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
        return np.dot(s[2*N:2*N +3],s[2*N:2*N +3])

def mzsquared_30_spins(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
        return np.dot(s[2*N:2*N +30],s[2*N:2*N +30])
def one_bond(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    res = 0
    for i in range(3):
        res = res + s[N*i] * s[N*i+1]
        return res
def ten_bonds(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    res = 0
    for l in range(10):
        for i in range(3):
            res = res + s[N*i + l] * s[N*i+1 + l]
    return res
def thirty_bonds_squared(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    res = 0
    for l in range(30):
        for i in range(3):
            res = res + (s[N*i + l] * s[N*i+1 + l])**2
    return res
def sixty_bonds_squared(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    res = 0
    for l in range(60):
        for i in range(3):
            res = res + (s[N*i + l] * s[N*i+1 + l])**2
    return res
def ten_z(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    return sum(s[:10])
    """ returns s_z if M_z = 0 toherwise returns a perpendicular component to the magnetization lying in the xy plane"""
def s_z_var(s):
    N = int(len(s)/3)
    s_z = s[2*N:]
    return np.var(s_z)
def bond(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond = i_iplus1[:N] + i_iplus1[N:2*N] + i_iplus1[2*N:]
    return bond
def bond_mean(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond = i_iplus1[:N] + i_iplus1[N:2*N] + i_iplus1[2*N:]
    return np.mean(bond)
def bond_var(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond = i_iplus1[:N] + i_iplus1[N:2*N] + i_iplus1[2*N:]
    #bond =  i_iplus1[2*N:]

    return np.var(bond)
def bond2_var(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus2 = np.multiply(s, spinlib.cycle3D(s,N, 2))
    bond = i_iplus2[:N] + i_iplus2[N:2*N] + i_iplus2[2*N:]
    #bond = i_iplus2[2*N:]
    return np.var(bond)
def bond2_mean(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus2 = np.multiply(s, spinlib.cycle3D(s,N, 2))
    bond = i_iplus2[:N] + i_iplus2[N:2*N] + i_iplus2[2*N:]
    return np.mean(bond)
def bondz_mean(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond = i_iplus1[2*N:]
    return np.mean(bond)
def bondz_var(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond =  i_iplus1[2*N:]

    return np.var(bond)
def bondx_var(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond =  i_iplus1[:N]

    return np.var(bond)
def bondy_var(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus1 = np.multiply(s, spinlib.cycle3D(s,N, 1))
    bond =  i_iplus1[N:2*N]

    return np.var(bond)
def bondz2_var(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus2 = np.multiply(s, spinlib.cycle3D(s,N, 2))
    bond = i_iplus2[2*N:]
    return np.var(bond)
def bondz2_mean(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    i_iplus2 = np.multiply(s, spinlib.cycle3D(s,N, 2))
    bond = i_iplus2[2*N:]
    return np.mean(bond)


#CODE FOR MONTE CARLO CALCULATIONS FOLLOWS
def s_perp(s):
    """ two random perpendicular components of s to magnetization
        and the parallel component to magnetization
    """
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        N = int(len(s)/3)
    m= spinlib.return_magnetization_direction(s)
    dir1 = spinlib.normalize(np.array([m[1],-m[0],0]), N=1)
    dir2 = np.cross(dir1,m)

    theta = np.pi * random.random()
    matrix = spinlib.rotation_matrix(m,theta)
    dir1 = np.dot(matrix,dir1)
    dir2 = np.dot(matrix,dir2)
    

    sx,sy,sz = spinlib.getSlices(s)
    sperp = dir1[0]*sx + dir1[1] * sy  + dir1[2] * sz
    sperp2 = dir2[0]*sx + dir2[1] * sy + dir2[2] * sz
    sperp3 = m[0]*sx + m[1] * sy + m[2] * sz
    return sperp,sperp2,sperp3
def s_perp_var(s):
    """variances of  two random perpendicular components of s to magnetization
     and the parallel component
    """
    sperp1,sperp2,sperp3 = s_perp(s)
    return np.var(sperp1),np.var(sperp2),np.var(sperp3)

def bond_perp(s):
    """energies stored in the  two random perpendicular components to magnetization
     and in the parallel ( to magnetization) component of spins
    """
    sperp1,sperp2,sperp3 = s_perp(s)
    return np.multiply(sperp1,np.roll(sperp1,1)),np.multiply(sperp2,np.roll(sperp2,1)),\
                np.multiply(sperp3,np.roll(sperp3,1))

def bond_perp_mean(s):
    """means for the energies stored in the  two random perpendicular components to magnetization
     and in the parallel ( to magnetization) component of spins
    """
    e1,e2,e3 = bond_perp(s)
    return np.mean(e1),np.mean(e2),np.mean(e3)
def bond_perp_var(s):
    """variances for the energies stored in the  two random perpendicular components to magnetization
     and in the parallel ( to magnetization) component of spins
    """
    e1,e2,e3 = bond_perp(s)
    return np.var(e1),np.var(e2),np.var(e3)

if __name__ == '__main__':
    print(bond_mean([1,2,3,4,5,6]), 44)
    print(bond_var([1,2,3,4,5,6]), 0)
    print(bond2_mean([1,2,3,4,5,6,1,2,3]), 96/3)
    print(bond2_var([1,2,3,4,5,6,1,2,3]), 56)
    print(bond_perp_mean(np.array([1,1,2,-2,-3,3]))," should be -4,-9,1")
