"""
Contains helper methods for dealing with spins in our representation [x1,...,xN,y1,...,zN]
"""

import numpy as np
import time
import random
import math
import s0_factory
import observables


def getSlices(s):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        n = int(len(s)/3)
        return [s[:n],s[n:n+n],s[n+n:n+n+n]]
def cross(s, heff):
    """Cross product s x heff, performed site by site
    """
    [sx, sy, sz] = getSlices(s)
    [hx, hy, hz] = getSlices(heff)
    return np.concatenate((np.multiply(sy,hz)-np.multiply(sz,hy),
                np.multiply(sz,hx)-np.multiply(sx,hz),
                np.multiply(sx,hy)-np.multiply(sy,hx)), axis=0)

def cycle3D(s, N, amount):
    """np.roll by amount, working on our 3D representation [sx,sy,sz]

    """
    if amount == 0 and len(s)== 3*N:
        return s
    return np.concatenate(
                    (np.roll(s[:N],amount),
                    np.roll(s[N:2*N],amount),
                    np.roll(s[2*N:3*N],amount)),
                axis = 0)

def cycleLeft3D(s, N):
    return cycle3D(s,N,-1)

def cycleRight3D(s, N):
    return cycle3D(s,N,1)

def get_spin_at_index(s, index):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    else:
        n = int(len(s)/3)
        return np.array([s[index],s[index + n],s[index + n + n]])


def get_standard_basis(N, n_vectors):
    """returns first n vectors of standard R^3N basis
    
    concatenated in one array, using our sx,sy,sz representation
    """
    v = np.zeros(3 * N * n_vectors)
    for i in range(n_vectors):
        v[3*N*i + i] = 1
    return v

def normalize(s, N):
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    norm = np.sqrt(s[:N]**2 + s[N:2*N]**2 + s [2*N:3*N] **2)
    s_ret = np.concatenate((np.true_divide(s[:N],norm),
            np.true_divide(s[N:2*N],norm),
            np.true_divide(s[2*N:3*N],norm)),axis = 0)
    return s_ret


def get_s_and_vectors(svect, N):
    """ from svect, which is concatenated s and vectors, separates the two and returns

    Used for Lyapunov solver, where vectors are tangent vector and s is the state of our system
    """
    s1 = np.array(svect[:3 * N])
    vectors = np.array(svect[3*N:])
    if len(vectors)%(3*N)!=0:
        raise Exception("not a set of vectors,want" + str(3*N) + "got" + str(len(vectors)))
    x = int(len(vectors)/(3*N))
    y = 3 * N
    if len(vectors) == 0:
        return s1, vectors
    else:
        return s1, vectors.reshape(x,y).T

def orthogonalize(svect, N):
    """returns - re-orthogonalzied svect and diagonal entries of r coming form QR

    see: Karlheinz GEIST, Ulrich PARLITZ and Werner LAUTER BORN,
        Progress of Theoretical Physics, Vol. 83, No.5, May 1990, Section 3.1
        for the Lyapunov computation method which this implements
    """
    s, vectors = get_s_and_vectors(svect, N)
    if len(vectors)==0:
        return s, []
    else:
        q, r = np.linalg.qr(vectors)
        #more Transpose magic - q is now (3*N, n_vect), to unravel it into 3*N * n_vect we need to first trnspose 
        # need to ravel s aswell for some reason to make it (3N,) array rather than (3N, 1)
        return  np.concatenate((s.ravel(),q.T.ravel()), axis = 0), np.abs(np.diag(r))

def new_random_spin():
    """ return New random spin direction on the unit sphere

    """
    squ = 2
    while squ >= 1:
        x1 = 1 - 2 * random.random()
        x2 = 1 - 2 * random.random()
        squ = x1**2 + x2**2
    Sx = 2 * x1 * math.sqrt(1-squ)
    Sy = 2 * x2 * math.sqrt(1-squ)
    Sz = 1-2*squ
    return [Sx,Sy,Sz]

def set_spin(s0, i, S, N):
    """ set i-th spin of s0(of length N) to S
    """
    s0[i] = S[0]
    s0[i+N] = S[1]
    s0[i+N+N] = S[2]

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def return_observables(s , observables):
    """evaluates each of the observables on each datapoint in s

    """
    si = s[0]
    sf = s[-1]
    values = []
    for observable in observables:
        values.append(np.apply_along_axis(observable, axis = 1, arr = s))
    return values, si, sf

def return_magnetization_direction(s):
    """returns magnetization vector, normalized
    """
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    sx, sy, sz = getSlices(s)
    return normalize(np.array([np.sum(sx),np.sum(sy),np.sum(sz)]),1)

def return_magnetization(s):
    """
    Returns magnetization vector
    """
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    N = len(s)/3
    sx, sy, sz = getSlices(s)
    return np.array([np.sum(sx),np.sum(sy),np.sum(sz)])/N
def return_magnetization_norm(s):
    """
    Returns magnitude of magnetization
    """
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    N = len(s)/3
    sx, sy, sz = getSlices(s)
    return np.linalg.norm(np.array([np.sum(sx),np.sum(sy),np.sum(sz)])/N)
def reflect_across_xy(axis,S_old):
    perp = np.array([axis[1],-axis[0],0])
    perp = normalize(perp,1)
    axis = normalize(axis,1)
    return np.dot(axis,S_old)*axis - np.dot(perp,S_old) * perp

def compute_angle(s):
    """Computes the twist angle associated with s

    """
    if len(s)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    N = int(len(s)/3)
    i_iplus1 = np.multiply(s, cycle3D(s,N, 1))
    bond = i_iplus1[:N] + i_iplus1[N:2*N] + i_iplus1[2*N:]
    crosses = cross(s,cycle3D(s,N,1))
    ang = 0
    for i in range(N):
        if abs(bond[i]) >1:
            print(bond[i], "BOND", i, "OUT OF RANGE ignoring")
        else:
            if crosses[2*N+i]>0:
                ang += math.acos(bond[i])
            else:
                ang -= math.acos(bond[i])

    return ang

def twist(s0, winding_number):
    """applies a twist of given winding number to s0 and returns
    
    """
    if len(s0)%3!=0:
        raise Exception("Not 3D spins" + str(len(s)))
    N = int(len(s0)/3)

    angle = 2*np.pi * winding_number / N
    s0x= [s0[0]]
    s0y = [s0[N]]
    s0z = [s0[2*N]]
    for i in range(1,N):
        final_angle = i * angle
        sin_final = math.sin(final_angle)
        cos_final = math.cos(final_angle)
        s_new = [cos_final*s0[i] +sin_final*s0[i+N], -sin_final*s0[i] +  cos_final*s0[N + i],0]
        s0x.append(s_new[0])
        s0y.append(s_new[1])
        s0z.append(s_new[2])
        norm = np.linalg.norm([s0x [-1],s0y [-1],s0z [-1] ])
        s0x [-1] = s0x[-1] / norm
        s0y [-1] = s0y[-1] / norm
        s0z [-1] = s0z[-1] / norm

    return np.concatenate((s0x,s0y,s0z),axis = 0)

if __name__ == '__main__':
    #Various tests

    print(cycleRight([1,2,3]))
    print(cycle3D([1,2,3,4,5,6,7,8,9], 3, 1))
    print(cycle3D([1,2,3,4,5,6,7,8,9], 3, 2))
    print(cycle3D([1,2,3,4,5,6,7,8,9], 3, -2))
    print(reflect_across_xy(np.array([2,0,0]),np.array([0,1,0])))
    print(normalize(np.array([0,0,0,0,0,0]),2))
    print("should be -1")
    print(compute_angle(s0_factory.get_s0_twist(200,-1))/np.pi/2)
    print("should be 1")
    print(compute_angle(s0_factory.get_s0_twist(200,1))/np.pi/2)
    print("should be integer")
    print(compute_angle(s0_factory.get_s0_xy_equipartition(300,14))/np.pi/2)
    print(compute_angle(twist([1,1,1,0,0,0,0,0,0],1))/np.pi/2)
    print(compute_angle(twist([1,0,-1,0,1,0,0,0,0],1))/np.pi/2)
    print(compute_angle([1,0,-1,0,1,0,0,0,0])/np.pi/2)

    st = time.time()
    for i in range(1000):
        twist(np.ones(750),1)
    end = time.time()
    print('time',end-st)
