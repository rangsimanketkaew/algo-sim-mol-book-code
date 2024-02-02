"""
Algorithms for Computer Simulation of Molecular Systems
Copyright (c) 2023 Rangsiman Ketkaew

License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import numpy as np

from math import sqrt, pi
from timeit import default_timer as timer

start_time = timer()


def readoneint(filename, naos, t):
    text = open(filename, "r").read()
    splitfile = text.strip().split("\n")
    # print splitfile
    k = 0
    for i in range(naos):
        for j in range(i + 1):
            # print i, j
            line = splitfile[k]
            t[i, j] = line.split()[2]
            t[j, i] = t[i, j]
            k = k + 1
    return


def readtwoint(filename, naos, eri):
    text = open(filename, "r").read()
    splitfile = text.strip().split("\n")
    ntwoints = len(splitfile)
    for k in range(ntwoints):
        line = splitfile[k]
        m = int(line.split()[0]) - 1
        n = int(line.split()[1]) - 1
        k = int(line.split()[2]) - 1
        l = int(line.split()[3]) - 1
        eri[m, n, k, l] = line.split()[4]
        eri[n, m, k, l] = eri[m, n, k, l]
        eri[m, n, l, k] = eri[m, n, k, l]
        eri[n, m, l, k] = eri[m, n, k, l]
        eri[k, l, m, n] = eri[m, n, k, l]
        eri[k, l, n, m] = eri[m, n, k, l]
        eri[l, k, m, n] = eri[m, n, k, l]
        eri[l, k, n, m] = eri[m, n, k, l]
    return


# Read one/two electron intergrals
def readint(naos, t, s, v, eri):
    readoneint("STO-3G/t.dat", naos, t)
    readoneint("STO-3G/s.dat", naos, s)
    readoneint("STO-3G/v.dat", naos, v)
    readtwoint("STO-3G/eri.dat", naos, eri)
    return


# Builds new (i)th fock matrix from the (i-1)th density matrix
def build_new_fock(denmat, naos, hcore):
    fmat = np.zeros((naos, naos))
    for m in range(naos):
        for n in range(naos):
            fmat[m, n] = hcore[m, n]
            for k in range(naos):
                for l in range(naos):
                    fmat[m, n] += denmat[k, l] * (
                        2 * erimat[m, n, k, l] - erimat[m, k, n, l]
                    )
    return fmat


# Diagonalises (i)th fock matrix and obtain (i)th density matrix,
# electronic energy, etc (check line 92)
def diagonalize_fock(fmat, smat_half, hcore):
    fmatp = np.dot(
        smat_half, np.dot(fmat, smat_half)
    )  # transforming to canonical AO basis

    epsilon, cmatp = np.linalg.eig(fmatp)  # diagonalising fmatp
    idx = epsilon.argsort()[::1]  # sorting the eigenvalues
    epsilon = epsilon[idx]
    cmatp = cmatp[:, idx]
    cmat = np.dot(smat_half, cmatp)
    # only first 5 MOs are occupied in case of water
    cmat_occ = cmat[:, 0:5]

    denmat = np.dot(
        cmat_occ, cmat_occ.transpose()
    )  # building density matrix and calculating electronic energy
    var1 = hcore + fmat
    eelec = np.trace(np.dot(denmat, var1))

    return denmat, eelec, fmatp, cmat, epsilon


# builds a new error matrix based on (e = FDS-SDF)
def build_error_vector(fmat, denmat, smat):
    errvec = (
        np.dot(fmat, np.dot(denmat, smat)) - np.dot(smat, np.dot(denmat, fmat))
    ).reshape(naos * naos, 1)
    norm_errvec = sqrt(np.dot(errvec.transpose(), errvec)[0, 0])
    return errvec, norm_errvec


# builds the B-matrix and solve for DIIS coefficient
def get_diis_coeff(errvec_subspace):
    b_mat = np.zeros((len(errvec_subspace) + 1, len(errvec_subspace) + 1))
    b_mat[-1, :] = -1
    b_mat[:, -1] = -1
    b_mat[-1, -1] = 0

    rhs = np.zeros((len(errvec_subspace) + 1, 1))
    rhs[-1, -1] = -1

    for i in range(len(errvec_subspace)):
        for j in range(i + 1):
            b_mat[i, j] = np.dot(errvec_subspace[i].transpose(), errvec_subspace[j])
            b_mat[j, i] = b_mat[i, j]

    *diis_coeff, _ = np.linalg.solve(b_mat, rhs)

    return diis_coeff


# Builds an extrapolated fock matrix by a linear combination
# (These won't be a part of fmat_subspace)
def extrapolated_fock(fmat_subspace, diis_coeff):
    extrapolated_fmat = np.zeros((naos, naos))

    for i in range(len(fmat_subspace)):
        extrapolated_fmat += fmat_subspace[i] * diis_coeff[i]

    return extrapolated_fmat


# Defining a class = Subspace for storing error vectors,
# fock, and density matrices
class Subspace(list):
    def append(self, item):
        list.append(self, item)
        if len(self) > dimSubspace:
            del self[0]


print("=" * 58)
print(f"* Output for SCF energy calculation using DIIS algorithm *")
print("=" * 58, "\n\n")


# Bunch of global variables defined; initalizing some arrays

naos = 7  # STO-3G has 7 AOs for H2O

tmat = np.zeros((naos, naos))
smat = np.zeros((naos, naos))
vmat = np.zeros((naos, naos))
erimat = np.zeros((naos, naos, naos, naos))
denmat = np.zeros((naos, naos))

with open("STO-3G/enuc.dat", "r") as f:
    enuc = float(f.read())

readint(naos, tmat, smat, vmat, erimat)

# This will be initial guess for fock matrix
hcore = tmat + vmat

iterations = 12  # maximum number of iterations

scftol = 1e-12  # energy convergence criterion
dentol = 1e-12  # density convergence criterion
errtol = 1e-15  # error vector should be as close to 0

print(f"Molecule = H2O")
print(f"Basis Set = STO-3G")
print(f"Total number of basis functions = {naos}\n\n")

print(f"CONVERGENCE CRITERIA:\n~~~~~~~~~~~~~~~~~~~~~")
print(f"Energy convergence criterion: {scftol}")
print(f"Norm of density matrix convergence criterion: {dentol}")
print(f"Norm of DIIS-error vector convergence criterion: {errtol}\n\n")


# S^(-1/2) is used to diagonalise Fock matrix
s_eigval, s_eigvec = np.linalg.eig(smat)
s_halfeig = np.zeros((naos, naos))

for i in range(naos):
    s_halfeig[i, i] = s_eigval[i] ** (-0.5)

a = np.dot(s_eigvec, s_halfeig)  # a = L*s^(-1/2)
b = s_eigvec.transpose()
smat_half = np.dot(a, b)


####################
# DIIS Code begins #
####################

# Usually, around 6-8 to give a reasonable result
dimSubspace = 6

# Creating instances of the class=Subspace.
# The maximum dimension if defined above.
errvec_subspace = Subspace()
fmat_subspace = Subspace()
denmat_subspace = Subspace()

# Initilizing these variables to check convergence
old_energy, old_denmat = 0, np.zeros((naos, naos))

# Providing the guess fock matrix to begin with
# (This will not be a part of fmat_subspace)
fmat = np.zeros((naos, naos))
for m in range(naos):
    for n in range(naos):
        fmat[m, n] = hcore[m, n]

print(f"SCF ITERATIONS BEGIN:\n~~~~~~~~~~~~~~~~~~~~~")

# Begining the DIIS-SCF iterations
for i in range(iterations):
    print(f"Iteration {i+1}:")

    if i <= 1:
        denmat = diagonalize_fock(fmat, smat_half, hcore)[0]
        denmat_subspace.append(denmat)

        energy = diagonalize_fock(fmat, smat_half, hcore)[1]
        print(f"    * Electronic energy = {energy} Eh")

        fmat = build_new_fock(denmat, naos, hcore)
        fmat_subspace.append(fmat)

        errvec, norm_errvec = build_error_vector(fmat, denmat, smat)
        errvec_subspace.append(errvec)
        print(f"    * Norm of DIIS error vector = {norm_errvec}")

    else:
        # Start building and solving B-matrix to obatined DIIS coefficient
        # after we have at least 2 error vectors
        diis_coeff = get_diis_coeff(errvec_subspace)

        # Building extrapolated fock matrices
        extrapolated_fmat = extrapolated_fock(fmat_subspace, diis_coeff)

        # Density matrix from extrapolated fock
        denmat = diagonalize_fock(extrapolated_fmat, smat_half, hcore)[0]
        denmat_subspace.append(denmat)

        # Next entry in the fock subspace obtained
        # from the recent density matrix obtained
        fmat = build_new_fock(denmat, naos, hcore)
        fmat_subspace.append(fmat)

        # Electronic energy
        energy = diagonalize_fock(fmat, smat_half, hcore)[1]
        print(f"    * Electronic energy = {energy} Eh")

        # Error vector calculation
        errvec, norm_errvec = build_error_vector(fmat, denmat, smat)
        errvec_subspace.append(errvec)
        print(f"    * Norm of DIIS error vector = {norm_errvec}")
        # DIIS algorithm ends here ####

    ### Checking for various convergence criteria
    deltaE = energy - old_energy
    print(f"    * Energy convergence = {deltaE} Eh")

    denmat_vector, olddenmat_vector = np.reshape(denmat, (naos * naos, 1)), np.reshape(
        old_denmat, (naos * naos, 1)
    )
    norm_denmat = denmat_vector - olddenmat_vector
    denconv = (np.dot(norm_denmat.transpose(), norm_denmat)) ** 0.5
    print(f"    * Norm of density matrix convergence = {denconv[0,0]}\n")

    # Stop the iterations when convergence is reached
    if abs(deltaE) < scftol and abs(denconv[0, 0]) < dentol:
        print(f"SCF ENERGY AND DENSITY CONVERGED WITHIN {i+1} ITERATIONS!\n\n")
        MOenergies = diagonalize_fock(fmat, smat_half, hcore)[4]
        print(f"ORBITAL ENERGIES:\n~~~~~~~~~~~~~~~~~\n{MOenergies}\n\n")
        print(f"FINAL RESULT:\n~~~~~~~~~~~~~")
        print(f"    * Electronic energy = {energy} Eh")
        print(f"    * Total energy = {energy+enuc} Eh")
        print(f"    * HOMO-LUMO gap = {MOenergies[5]-MOenergies[4]} Eh")
        break
    else:
        old_denmat = denmat
        old_energy = energy

print(f"\n")
end_time = timer()
print(f"Wall time = {end_time-start_time} seconds\n")
