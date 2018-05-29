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



def bondAngles(positions, distances):

	#doesn't always give the angle between neighboring atoms eg ijk = 2,3,4 gives the angle between the oxygen and two hydrogens even though oxygen is not bound to either hydrogen
	#TODO: Get bond angles for all atom combinations with d_ij and d_jk < 4
	for j in range(1,atomCount-1):

		for z in range(len(distances)):

			if((j == distances[z][0] or j == distances[z][1]) and (j-1 == distances[z][0] or j-1 == distances[z][1])):
				D_ij = distances[z][2]

			if((j == distances[z][0] or j == distances[z][1]) and (j+1 == distances[z][0] or j+1 == distances[z][1])):
				D_jk = distances[z][2]

		e_ij = (positions[1]-positions[0])/D_ij
		e_jk = (positions[2]-positions[1])/D_jk

		angle = 180-np.arccos(np.dot(e_ij,e_jk))*180/np.pi
		print(str(j-1) +" "+ str(j) +" "+ str(j+1) + " " + str(angle))


	"""
	for z in range(len(distances)):

		if((1. == distances[z][0] or 1. == distances[z][1]) and (0. == distances[z][0] or 0. == distances[z][1])):
			D_ij = distances[z][2]

		if((1. == distances[z][0] or 1. == distances[z][1]) and (2. == distances[z][0] or 2. == distances[z][1])):
			D_jk = distances[z][2]

	e_ij = (positions[1]-positions[0])/D_ij
	e_jk = (positions[2]-positions[1])/D_jk

	angle = np.arccos(np.dot(e_ij,e_jk))
	print(angle)
	"""

	return None


def main(positions):

	posMain = pos(positions)
	lenghts = bondLenghts(posMain)

	print("Number of atoms: ")
	print(atomCount)
	print("\nPositions:")
	print(posMain)
	print("\nBond lenghts:")
	print(lenghts)

	return None

bondAngles(pos(geomCont),bondLenghts(pos(geomCont)))

#main(geomCont)
molGeom.close()