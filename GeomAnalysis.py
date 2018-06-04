#!/usr/bin/env python

import numpy as np, os
import numpy.linalg as npl

molGeom = open('.\\Input_Files\\geom.txt', 'r')
geomCont = molGeom.readlines()


atomCount = int(geomCont[0].replace('\n', ''))
geomCont = geomCont[1:]

def pos(positions):

	posMatrix = np.array([])

	for i in positions:

		i = i.split('\t')
		i[-1] = i[-1].replace('\n','')
		i = i[1:]
		i = [float(x) for x in i]
		posMatrix = np.append(posMatrix, i)

	posMatrix = np.reshape(posMatrix, (len(positions),3))
	return posMatrix

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
			distances = np.append(distances, np.abs(npl.norm(positions[i]-posDummy[k])))
			bondPartnerCount += 1
	
	distances = np.reshape(distances, (-1,3))
	return distances

def v_(atom1, atom2, positions = pos(geomCont)):

	v_ij = (positions[atom2]-positions[atom1])
	return v_ij

def bondAngles(positions):

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
					angles = np.append(angles, [i,j,k,angle])
					angles = np.reshape(angles, (-1,4))
						
	return angles

def outOfPlaneAngles(positions):

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

						oOPAngles = np.append(oOPAngles, [int(l),int(i),int(j),int(k),oopa])
						oOPAngles = np.reshape(oOPAngles, (-1,5))


	oOPAngles = np.reshape(oOPAngles, (-1,5))
	return oOPAngles

def main(positions):

	posMain = pos(positions)
	lenghts = bondLenghts(posMain)
	angles = bondAngles(posMain)
	oop = outOfPlaneAngles(posMain)

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

	return None

main(geomCont)
molGeom.close()