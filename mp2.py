######################################
#  Moller-Plesset Second Order
#   Perturbation Theory (MP2)
######################################

import numpy as np

######################################
#  Functions
######################################

def eint(a, b, c, d):
    """Return compound index given four indices
    """    
    if a > b:
        ab = a * (a + 1) / 2 + b
    else:
        ab = b * (b + 1) / 2 + a
    if c > d:
        cd = c * (c + 1) / 2 + d
    else:
        cd = d * (d + 1) / 2 + c
    if ab > cd:
        abcd = ab * (ab + 1) / 2 + cd
    else:
        abcd = cd * (cd + 1) / 2 + ab
    return abcd

# 
def teimo(a, b, c, d):
    """Return Value of spatial MO two electron integral
    Example: (12\vert 34) = tei(1,2,3,4)
    """
    return ttmo.get(eint(a, b, c, d), 0.0e0)


######################################
#  Initialize orbital energies and 
#  transformed two-electron integrals
######################################

Nelec = 2  # 2 electrons in HeH+
dim = 2  # two spatial basis functions in STO-3G
E = [-1.52378656, -0.26763148]
ttmo = {
    5.0: 0.94542695583037617,
    12.0: 0.17535895381500544,
    14.0: 0.12682234020148653,
    17.0: 0.59855327701641903,
    19.0: -0.056821143621433257,
    20.0: 0.74715464784363106,
}

######################################
#  Convert spatial orbital to 
#  spin orbital MOs
######################################

# This makes the spin basis double bar integral 
# (physicists' notation).
# We double the dimension (each spatial orbital 
# is now two spin orbitals)

dim = dim * 2
ints = np.zeros((dim, dim, dim, dim))
for p in range(1, dim + 1):
    for q in range(1, dim + 1):
        for r in range(1, dim + 1):
            for s in range(1, dim + 1):
                value1 = (
                    teimo((p + 1) // 2, (r + 1) // 2, (q + 1) // 2, (s + 1) // 2)
                    * (p % 2 == r % 2)
                    * (q % 2 == s % 2)
                )
                value2 = (
                    teimo((p + 1) // 2, (s + 1) // 2, (q + 1) // 2, (r + 1) // 2)
                    * (p % 2 == s % 2)
                    * (q % 2 == r % 2)
                )
                ints[p - 1, q - 1, r - 1, s - 1] = value1 - value2

######################################
#  Spin basis fock matrix eigenvalues
######################################

fs = np.zeros((dim))
for i in range(0, dim):
    fs[i] = E[i // 2]
# fs = np.diag(fs)

######################################
#  MP2 Calculation
######################################

# First two loops sum over occupied spin orbitals
# which is equal to the number of electrons. 
# The last two loops loop over the virtual orbitals.

EMP2 = 0.0
for i in range(0, Nelec):
    for j in range(0, Nelec):
        for a in range(Nelec, dim):
            for b in range(Nelec, dim):
                EMP2 += 0.25 * ints[i, j, a, b] * ints[i, j, a, b] / (fs[i] + fs[j] - fs[a] - fs[b])

print("E(MP2) Correlation Energy = ", EMP2, " Hartrees")
