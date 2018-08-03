#!/usr/bin/env python

import numpy as np, os
import numpy.linalg as npl

h0 = 6.626070040*10.**-34.
c0 = 299792458
grams = 1.66053873*10.**-27.
cm = 0.5291772083 * 10.**-10.

molInput = input("Which File should be read?\n")
molGeom = open('.\\Input_Files\\'+molInput+'.txt', 'r')
geomCont = molGeom.readlines()


atomCount = int(geomCont[0].replace('\n', ''))
geomCont = geomCont[1:]

def pos(content):

	posMatrix = np.array([])

	for i in content:

		i = i.split('\t')
		i[-1] = i[-1].replace('\n','')
		i = i[1:]
		i = [float(x) for x in i]
		posMatrix = np.append(posMatrix, i)

	posMatrix = np.reshape(posMatrix, (len(content),3))
	return posMatrix

def atomicMass(content):

	an = np.array([])

	for i in content:

		i = i.split('\t')
		i = i[:1]
		i = [float(x) for x in i]
		an = np.append(an,i)

	for x in range(len(an)):
		if(an[x] == 6.):
			an[x] = 12.
		elif(an[x] == 8):
			an[x] = 16

	return an

def v_(atom1, atom2, positions = pos(geomCont)):

	v_ij = (positions[atom2]-positions[atom1])
	return v_ij

def bondLenghts(positions):

	distances = np.array([])
	posDummy = positions.copy()
	bondPartnerCount = 0			#correctly counts the index of the bonding partner as posDummy gets shorter while looping
	
	for i in range(len(positions)):

		posDummy = posDummy[1:]

		bondPartnerCount = i+1

		for k in range(len(posDummy)):

			distances = np.append(distances, i)
			distances = np.append(distances, bondPartnerCount)
			distances = np.append(distances, np.around(np.abs(npl.norm(positions[i]-posDummy[k])), 3))
			bondPartnerCount += 1
	
	distances = np.reshape(distances, (-1,3))
	return distances

def bondAngles():

	angles = np.array([])

	for j in range(1,atomCount-1):
		for i in range(0,j):

			if(j == i):
				continue

			for k in range(j,atomCount):

				if(k == j or k == i):
					continue

				elif(npl.norm(v_(j,i)) <= 4. and npl.norm(v_(j,k)) <= 4.):
					e_ji = v_(j,i)/npl.norm(v_(j,i))
					e_jk = v_(j,k)/npl.norm(v_(j,k))

					angle = np.arccos(np.dot(e_ji,e_jk))*180/np.pi
					angles = np.append(angles, [i,j,k,np.around(angle,5)])
					angles = np.reshape(angles, (-1,4))
						
	return angles

def outOfPlaneAngles():

	oOPAngles = np.array([])

	for l in range(atomCount):
		for j in range(atomCount):
			for k in range(atomCount):
				for i in range(0,k):
					if(l != j and l != i and l != k and k != j and k != i and j != i and npl.norm(v_(j,i)) <= 4. and npl.norm(v_(j,k)) <= 4. and npl.norm(v_(j,l)) <= 4.):
						e_ji = v_(j,i)/npl.norm(v_(j,i))
						e_jk = v_(j,k)/npl.norm(v_(j,k))
						e_jl = v_(j,l)/npl.norm(v_(j,l))

						c_ji_jk = np.cross(e_ji,e_jk)

						angle_jkl = np.arccos(np.dot(e_ji,e_jk))

						oopa = np.dot(c_ji_jk,e_jl)/np.sin(angle_jkl)*180/np.pi

						oOPAngles = np.append(oOPAngles, [l,i,j,k,oopa])
						oOPAngles = np.reshape(oOPAngles, (-1,5))


	oOPAngles = np.reshape(oOPAngles, (-1,5))
	return oOPAngles

def torsionAngles():

	tAngles = np.array([])

	for i in range(atomCount):
		for j in range(i):
			for k in range(j):
				for l in range(k):
					if(l != j and l != i and l != k and k != j and k != i and j != i and npl.norm(v_(i,j)) <= 4. and npl.norm(v_(j,k)) <= 4. and npl.norm(v_(k,l)) <= 4.):
						e_ij = v_(i,j)/npl.norm(v_(i,j))
						e_jk = v_(j,k)/npl.norm(v_(j,k))
						e_kl = v_(k,l)/npl.norm(v_(k,l))

						c_ij_jk = np.cross(e_ij,e_jk)
						c_jk_kl = np.cross(e_jk,e_kl)

						d_ijkl = np.dot(c_ij_jk, c_jk_kl)

						angle_ijk = np.arccos(np.dot(e_ij,e_jk))
						angle_jkl = np.arccos(np.dot( e_jk,e_kl))

						ta = d_ijkl / (np.sin(angle_ijk) * np.sin(angle_jkl))

						if(ta <= -1.):
							ta = np.pi
						elif(ta >= 1.):
							ta = 0
						else:
							ta = np.arccos(ta)

						tAngles = np.append(tAngles, [i,j,k,l,np.around(ta*180/np.pi,3)])
						tAngles = np.reshape(tAngles, (-1,5))

	return tAngles

def centerOfMass(masses,positions):

	CoM = np.array([])
	totalMass = 0.

	for i in range(len(masses)):

		CoM = np.append(CoM, masses[i] * positions[i])
		totalMass += masses[i]

	CoM = CoM / totalMass
	CoM = np.reshape(CoM,(-1,3))
	CoM = np.sum(CoM, axis = 0)

	return CoM

def posCOM(r, CoM):

	r = [r[i]-CoM for i in range(len(r))]
	r = np.reshape(r, (-1,3))

	return r

def intertiaTensor(m,r):			# args(masses, positions)

	I = np.reshape(np.zeros(9), (3,3))
	
	for i in range(len(m)):			# len(m) is equal to the total amount of atoms
		
		I += m[i] * (np.dot(r[i],r[i]) * np.identity(3) - np.outer(r[i],r[i]))
		
	return I

def PMI(I):

	w,v = npl.eig(I)
	w = np.sort(w)
	w = [np.around(w[i]* grams * cm**2, 50) for i in range(len(w))]

	return w

def rotorType(PMI):

	rotType = np.select([PMI[0] == PMI[1] == PMI[2], PMI[1]-PMI[0] > 10**-4  and PMI[1] == PMI[2], PMI[0] == PMI[1] < PMI[2], PMI[0] < PMI[1] == PMI[2], PMI[0] != PMI[1] != PMI[2]], \
						["Spherical Top", "Linear", "Oblate Symmetric Top", "Prolate Symmetric Top", "Asymmetric Top"])
	
	return rotType

def rotConsts(PMI):

	rc = np.array(np.zeros(3))
	rc = [np.around(h0 / (8 * np.pi* np.pi * c0 * 100 * PMI[i]), 6) for i in range(len(PMI))]

	return rc


def main(positions):

	posMain = pos(positions)
	masses = atomicMass(geomCont)
	lenghts = bondLenghts(posMain)
	angles = bondAngles()
	oop = outOfPlaneAngles()
	torsionA = torsionAngles()
	CoM = centerOfMass(masses, posMain)
	posCoM = posCOM(posMain,CoM)
	iTens = intertiaTensor(masses, posCoM)
	PMIval = PMI(iTens)
	rotor = rotorType(PMIval)
	rotC = rotConsts(PMIval)

	print("Number of atoms: ")
	print(atomCount)
	print("\nPositions:")
	print(posMain)
	print("\nBond lenghts:")
	print(lenghts)
	print("\nBond Angles:")
	print(angles)
	print("\nOut of Plane Angles:")
	print(oop)
	print("\nTorsion Angles:")
	print(torsionA)
	print("\nCenter of Mass:")
	print(CoM)
	print("\nMoment of Inertia Tensor")
	print(iTens)
	print("\nPrincipal Moments of Intertia:")
	print(PMIval)
	print("\nMolecular Rotor Type:")
	print(rotor)
	print("\nRotational Constants")
	print(rotC)

	return None

main(geomCont)
molGeom.close()